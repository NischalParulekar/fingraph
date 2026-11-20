"""The FinGraph HTTP service.

    uvicorn fingraph.api.app:app --port 8000

Endpoints:
- GET  /healthz                    liveness: Redis, Postgres and the arms on offer
- GET  /metrics                    Prometheus metrics
- GET  /questions                  the dev-subset questions the demo can ask
- GET  /reports/{uid}              the table and paragraphs a question is about
- POST /ask                        answer a question with one arm
- GET  /runs/{thread_id}           a run's result, or its pending review
- POST /runs/{thread_id}/resume    a reviewer's decision on a paused run

Arm C (`graph_verifier`) runs in API mode: when its answer still fails the evidence
check after its retries, the graph pauses at human_review (LangGraph `interrupt()`),
its state is checkpointed in Postgres under the run's thread_id, and /ask returns
`pending_review`. POST /runs/{thread_id}/resume with approve, edit or reject finishes it,
from any process, since the state is in Postgres.

Finished answers are cached in Redis by (arm, report, normalised question). Each client
address has a token bucket, and model spend is capped per UTC day across all clients
(the pattern is tableqa's service).
"""

import asyncio
import dataclasses
import logging
import math
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from langgraph.types import Command
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from redis.asyncio import Redis
from redis.exceptions import RedisError

from fingraph.api.config import Settings
from fingraph.api.metrics import Metrics
from fingraph.api.store import Store, answer_key
from fingraph.data.tatqa import Example
from fingraph.eval.offline_verify import check, inputs_from_trace
from fingraph.llm import Spend
from fingraph.scales import SCALES
from fingraph.tools.retriever import backend

log = logging.getLogger("fingraph.api")

ARM_NAMES = {
    "single_agent": "A: single agent",
    "graph": "B: graph",
    "graph_verifier": "C: graph + verifier",
}
MAX_EVIDENCE = 60


class AskRequest(BaseModel):
    uid: str  # a question from the dev subset; its report is what gets searched
    arm: str
    question: str | None = None  # ask something else about the same report


class ResumeRequest(BaseModel):
    action: Literal["approve", "edit", "reject"]
    answer: list[str] | None = None  # for edit
    scale: str | None = None  # for edit


def load_examples() -> dict[str, Example]:
    from fingraph.data.splits import load_split

    return {e.uid: e for e in load_split("dev")}


def load_arms(names: list[str], checkpointer) -> dict:
    """Build each configured arm; one that cannot be built is left out and logged."""
    from dotenv import load_dotenv

    load_dotenv()
    arms = {}
    for name in names:
        try:
            if name == "single_agent":
                from fingraph.arms.single_agent import SingleAgent

                arms[name] = SingleAgent()
            elif name == "graph":
                from fingraph.arms.graph import GraphArm

                arms[name] = GraphArm()
            elif name == "graph_verifier":
                if checkpointer is None:
                    raise RuntimeError("needs Postgres for its checkpointer")
                from fingraph.arms.graph import GraphArm

                arms[name] = GraphArm(verifier=True, mode="api", checkpointer=checkpointer)
        except Exception:
            log.exception("arm %s is unavailable", name)
    return arms


def postgres_checkpointer(url: str):
    """A PostgresSaver on a connection pool (graph calls run on worker threads), or None."""
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(
            url,
            max_size=8,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=True,
            timeout=5,
        )
        saver = PostgresSaver(pool)
        saver.setup()
        return saver, pool
    except Exception:
        log.exception("Postgres is not reachable; Arm C is off")
        return None, None


def _report_dict(report) -> dict:
    return {
        "passed": report.passed,
        "supported": [dataclasses.asdict(c) for c in report.supported],
        "unsupported": [dataclasses.asdict(c) for c in report.unsupported],
        "scale_problems": report.scale_problems,
    }


def create_app(
    settings: Settings | None = None,
    redis: Redis | None = None,
    arms: dict | None = None,
    examples: dict[str, Example] | None = None,
    checkpointer=None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = app.state
        state.redis = redis or Redis.from_url(settings.redis_url, decode_responses=True)
        state.store = Store(state.redis, settings.cache_ttl_s, settings.run_ttl_s)
        state.examples = examples if examples is not None else load_examples()
        pool = None
        saver = checkpointer
        if arms is None and saver is None:
            saver, pool = postgres_checkpointer(settings.database_url)
        state.checkpointer = saver
        state.arms = arms if arms is not None else load_arms(settings.arms, saver)
        state.limit = asyncio.Semaphore(settings.concurrency)
        metrics.start_at_zero(state.arms)
        try:
            metrics.spend_today.set(await state.store.spend_today())
        except RedisError:
            log.warning("Redis is not reachable at startup")
        yield
        if redis is None:
            await state.redis.aclose()
        if pool is not None:
            pool.close()

    app = FastAPI(title="FinGraph", lifespan=lifespan)
    app.state.metrics = metrics
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(RedisError)
    async def redis_down(request: Request, exc: RedisError):
        log.error("Redis error: %s", exc)
        return JSONResponse(
            {"detail": "The service's store is unavailable. Try again shortly."}, status_code=503
        )

    def client_of(request: Request) -> str:
        if settings.client_ip_header:
            value = request.headers.get(settings.client_ip_header, "").strip()
            if value:
                return value
        return request.client.host if request.client else "unknown"

    def example_for(state, uid: str, question: str | None = None) -> Example:
        example = state.examples.get(uid)
        if example is None:
            raise HTTPException(404, f"No question with uid {uid!r}. See GET /questions.")
        if question is None:
            return example
        question = question.strip()
        if not question or len(question) > settings.max_question_chars:
            raise HTTPException(
                400, f"Ask a question of 1 to {settings.max_question_chars} characters."
            )
        return dataclasses.replace(example, question=question)

    def result_of(arm_name: str, prediction, example: Example) -> dict:
        """What the client sees of a finished run: answer, evidence and the verifier."""
        record = {"answer": prediction.answer, "scale": prediction.scale, "trace": prediction.trace}
        state = prediction.trace[-1] if prediction.trace else {}
        reports = state.get("verifier_reports") if isinstance(state, dict) else None
        if reports:  # Arm C: its own last verdict
            verifier = {k: reports[-1][k] for k in ("passed", "supported", "unsupported")}
            verifier["scale_problems"] = reports[-1]["scale_problems"]
            verifier["source"] = "arm"
        else:  # A and B: the same verifier, run after the fact
            verifier = {**_report_dict(check(record, example)), "source": "offline"}
        evidence, calculations = inputs_from_trace(prediction.trace)
        return {
            "answer": prediction.answer,
            "scale": prediction.scale,
            "abstained": not prediction.answer,
            "error": prediction.error,
            "evidence": [dataclasses.asdict(e) for e in evidence[:MAX_EVIDENCE]],
            "calculations": [dataclasses.asdict(c) for c in calculations],
            "verifier": verifier,
            "llm_calls": prediction.llm_calls,
            "tokens": prediction.tokens,
            "cost_usd": prediction.cost_usd,
            "arm_latency_ms": prediction.latency_ms,
        }

    def account(arm_name: str, spend_or_prediction) -> None:
        p = spend_or_prediction
        calls = getattr(p, "llm_calls", None) or getattr(p, "calls", 0)
        metrics.llm_calls.labels(arm_name).inc(calls)
        metrics.tokens.labels(arm_name).inc(p.tokens)
        metrics.cost.labels(arm_name).inc(p.cost_usd)

    @app.get("/healthz")
    async def healthz(request: Request):
        state = request.app.state
        redis_ok = await state.store.ping()
        body = {
            "status": "ok" if redis_ok else "degraded",
            "redis": redis_ok,
            "postgres": state.checkpointer is not None,
            "embedder": backend(),
            "arms": sorted(state.arms),
        }
        return JSONResponse(body, status_code=200 if redis_ok else 503)

    @app.get("/metrics")
    async def prometheus_metrics():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/arms")
    async def list_arms(request: Request):
        return [{"arm": a, "name": ARM_NAMES.get(a, a)} for a in request.app.state.arms]

    @app.get("/questions")
    async def list_questions(request: Request):
        return [
            {
                "uid": e.uid,
                "question": e.question,
                "answer_from": e.answer_from,
                "answer_type": e.answer_type,
            }
            for e in request.app.state.examples.values()
        ]

    @app.get("/reports/{uid}")
    async def report(uid: str, request: Request):
        e = example_for(request.app.state, uid)
        return {
            "uid": e.uid,
            "context_uid": e.context_uid,
            "question": e.question,
            "table": [list(row) for row in e.table],
            "paragraphs": [{"order": p.order, "text": p.text} for p in e.paragraphs],
            "gold": {"answer": e.gold["answer"], "scale": e.scale},
        }

    @app.post("/ask")
    async def ask(body: AskRequest, request: Request):
        state = request.app.state
        started = time.perf_counter()
        arm = state.arms.get(body.arm)
        if arm is None:
            raise HTTPException(400, f"Unknown arm. Choose one of: {', '.join(state.arms)}.")
        example = example_for(state, body.uid, body.question)

        key = answer_key(body.arm, example.context_uid, example.question)
        cached = await state.store.get_answer(key)
        allowed, wait = await state.store.take_tokens(
            client_of(request),
            1 if cached else settings.fresh_cost,
            settings.rate_capacity,
            settings.rate_refill_per_s,
        )
        if not allowed:
            metrics.asks.labels(body.arm, "rate_limited").inc()
            raise HTTPException(
                429,
                "Too many questions from this address. Wait a moment and try again.",
                headers={"Retry-After": str(max(1, math.ceil(wait)))},
            )
        if cached:
            metrics.asks.labels(body.arm, "cached").inc()
            metrics.latency.labels(body.arm, "hit").observe(time.perf_counter() - started)
            return {**cached, "cached": True, "cost_usd": 0.0}

        spent = await state.store.spend_today()
        metrics.spend_today.set(spent)
        if spent >= settings.daily_spend_cap_usd:
            metrics.asks.labels(body.arm, "spend_capped").inc()
            raise HTTPException(
                503, "Today's spending limit for the model is reached. Try again tomorrow (UTC)."
            )

        thread_id = uuid.uuid4().hex
        base = {
            "thread_id": thread_id,
            "arm": body.arm,
            "uid": example.uid,
            "context_uid": example.context_uid,
            "question": example.question,
        }
        verifier_arm = body.arm == "graph_verifier"
        async with state.limit:
            try:
                if verifier_arm:
                    spend = Spend()
                    run = asyncio.to_thread(
                        arm.run,
                        example,
                        {"question": example.question, "evidence": []},
                        spend,
                        thread_id,
                    )
                    final = await asyncio.wait_for(run, settings.ask_timeout_s)
                else:
                    run = asyncio.to_thread(arm.answer, example)
                    prediction = await asyncio.wait_for(run, settings.ask_timeout_s)
            except TimeoutError:
                metrics.asks.labels(body.arm, "failed").inc()
                raise HTTPException(504, "The arm took too long to answer.") from None
            except Exception as e:
                log.exception("arm %s failed", body.arm)
                metrics.asks.labels(body.arm, "failed").inc()
                raise HTTPException(502, f"The arm failed ({type(e).__name__}).") from None

        elapsed = time.perf_counter() - started
        metrics.latency.labels(body.arm, "miss").observe(elapsed)
        if verifier_arm:
            account(body.arm, spend)
            cost = spend.cost_usd
            for report in final.get("verifier_reports") or []:
                metrics.verifier.labels("pass" if report["passed"] else "fail").inc()
            if "__interrupt__" in final:
                pending = final["__interrupt__"][0].value
                run_record = {
                    **base,
                    "status": "pending_review",
                    "review": pending,
                    "cost_usd": cost,
                    "llm_calls": spend.calls,
                    "latency_ms": elapsed * 1000,
                }
                metrics.asks.labels(body.arm, "pending_review").inc()
                metrics.pending.inc()
                if cost:
                    metrics.spend_today.set(await state.store.add_spend(cost))
                await state.store.put_run(thread_id, run_record)
                return {**run_record, "cached": False}
            prediction = arm.prediction(final, spend, elapsed)
        else:
            account(body.arm, prediction)
            cost = prediction.cost_usd

        result = result_of(body.arm, prediction, example)
        run_record = {**base, "status": "done", **result, "latency_ms": elapsed * 1000}
        outcome = "abstained" if result["abstained"] else "answered"
        metrics.asks.labels(body.arm, outcome).inc()
        if cost:
            metrics.spend_today.set(await state.store.add_spend(cost))
        await state.store.put_run(thread_id, run_record)
        await state.store.put_answer(key, run_record)
        return {**run_record, "cached": False}

    @app.get("/runs/{thread_id}")
    async def get_run(thread_id: str, request: Request):
        run = await request.app.state.store.get_run(thread_id)
        if run is None:
            raise HTTPException(404, "No run with that thread_id. It may have expired.")
        return run

    @app.post("/runs/{thread_id}/resume")
    async def resume(thread_id: str, body: ResumeRequest, request: Request):
        state = request.app.state
        run = await state.store.get_run(thread_id)
        if run is None:
            raise HTTPException(404, "No run with that thread_id. It may have expired.")
        if run["status"] != "pending_review":
            raise HTTPException(409, f"This run is {run['status']}, not waiting for review.")
        arm = state.arms.get(run["arm"])
        if arm is None or arm.pending(thread_id) is None:
            raise HTTPException(409, "This run's checkpoint is not waiting for review.")
        decision = {"action": body.action}
        if body.action == "edit":
            if not body.answer or any(not a.strip() for a in body.answer):
                raise HTTPException(400, "An edit needs a non-empty answer.")
            if body.scale is not None and body.scale not in SCALES:
                raise HTTPException(400, f"scale must be one of {list(SCALES)}.")
            decision.update(answer=body.answer, scale=body.scale or run["review"]["scale"])

        example = example_for(state, run["uid"], run["question"])
        spend = Spend()
        started = time.perf_counter()
        command = Command(resume=decision)
        final = await asyncio.to_thread(arm.run, example, command, spend, thread_id)
        prediction = arm.prediction(final, spend, time.perf_counter() - started)
        result = result_of(run["arm"], prediction, example)
        done = {
            **{k: run[k] for k in ("thread_id", "arm", "uid", "context_uid", "question")},
            "status": "done",
            **result,
            "review": {**decision, "reviewed_draft": run["review"]["answer"]},
            "cost_usd": run["cost_usd"] + spend.cost_usd,
            "latency_ms": run["latency_ms"],
        }
        await state.store.put_run(thread_id, done)
        metrics.reviews.labels(body.action).inc()
        metrics.pending.dec()
        return done

    return app


def __getattr__(name: str):
    # `uvicorn fingraph.api.app:app` builds the app on first access, so importing this
    # module (as the tests do) has no side effects.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
