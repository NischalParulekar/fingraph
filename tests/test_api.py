import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from fingraph.api.app import create_app
from fingraph.api.config import Settings
from fingraph.arms.graph import GraphArm
from fingraph.arms.single_agent import SingleAgent
from fingraph.llm import LLM
from tests.test_arm_c import GOOD_PLAN, answer
from tests.test_single_agent import EXAMPLE, FakeClient, response, tool_use
from tests.test_tools import keyword_embedder

A_TURNS = [
    response(tool_use("t1", "query_table", {"sql": "SELECT c1, c2 FROM t WHERE row = 2"})),
    response(tool_use("t2", "calculate", {"expression": "(1320 - 1086) / 1086 * 100"})),
    response(tool_use("t3", "final_answer", {"answer": ["21.55"], "scale": "percent"})),
]


def make_client(a_turns=(), c_turns=(), **settings) -> TestClient:
    checkpointer = InMemorySaver()
    arms = {
        "single_agent": SingleAgent(LLM(client=FakeClient(a_turns)), embed=keyword_embedder),
        "graph_verifier": GraphArm(
            LLM(client=FakeClient(c_turns)),
            verifier=True,
            max_retries=0,
            mode="api",
            checkpointer=checkpointer,
            embed=keyword_embedder,
        ),
    }
    app = create_app(
        Settings(**settings),
        redis=fakeredis.aioredis.FakeRedis(decode_responses=True),
        arms=arms,
        examples={EXAMPLE.uid: EXAMPLE},
        checkpointer=checkpointer,
    )
    return TestClient(app)


@pytest.fixture
def client():
    with make_client(A_TURNS, [GOOD_PLAN, answer(["99"])]) as c:
        yield c


def test_health_questions_and_report(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok" and health["arms"] == ["graph_verifier", "single_agent"]
    assert client.get("/questions").json()[0]["uid"] == EXAMPLE.uid
    report = client.get(f"/reports/{EXAMPLE.uid}").json()
    assert report["table"][2][0] == "Wireless" and report["gold"]["answer"] == 21.55
    assert client.get("/reports/nope").status_code == 404


def test_ask_answers_with_evidence_and_verifier_then_caches(client):
    first = client.post("/ask", json={"uid": EXAMPLE.uid, "arm": "single_agent"}).json()
    assert first["status"] == "done" and first["answer"] == ["21.55"]
    assert first["verifier"]["passed"] and first["verifier"]["source"] == "offline"
    assert first["calculations"][0]["expression"] == "(1320 - 1086) / 1086 * 100"
    assert any(e["text"] == "1,320" for e in first["evidence"])
    assert client.get(f"/runs/{first['thread_id']}").json()["answer"] == ["21.55"]

    # Same question, any spacing or case: from the cache, at no cost, no model call.
    again = client.post(
        "/ask",
        json={
            "uid": EXAMPLE.uid,
            "arm": "single_agent",
            "question": EXAMPLE.question.upper() + "  ",
        },
    ).json()
    assert again["cached"] is True and again["cost_usd"] == 0.0
    metrics = client.get("/metrics").text
    assert 'fingraph_ask_total{arm="single_agent",outcome="cached"} 1.0' in metrics
    assert 'fingraph_llm_calls_total{arm="single_agent"} 3.0' in metrics


def test_unsupported_answer_pauses_for_review_and_resumes_with_an_edit(client):
    pending = client.post("/ask", json={"uid": EXAMPLE.uid, "arm": "graph_verifier"}).json()
    assert pending["status"] == "pending_review"
    assert pending["review"]["answer"] == ["99"]
    assert not pending["review"]["verifier_report"]["passed"]
    thread = pending["thread_id"]
    assert client.get(f"/runs/{thread}").json()["status"] == "pending_review"

    bad = client.post(f"/runs/{thread}/resume", json={"action": "edit", "answer": []})
    assert bad.status_code == 400

    done = client.post(
        f"/runs/{thread}/resume", json={"action": "edit", "answer": ["21.55"], "scale": "percent"}
    ).json()
    assert done["status"] == "done" and done["answer"] == ["21.55"]
    assert done["review"]["action"] == "edit" and done["review"]["reviewed_draft"] == ["99"]
    assert client.get(f"/runs/{thread}").json()["status"] == "done"
    assert client.post(f"/runs/{thread}/resume", json={"action": "approve"}).status_code == 409
    assert 'fingraph_reviews_total{action="edit"} 1.0' in client.get("/metrics").text


def test_a_rejected_review_is_an_abstention():
    with make_client(c_turns=[GOOD_PLAN, answer(["99"])]) as c:
        thread = c.post("/ask", json={"uid": EXAMPLE.uid, "arm": "graph_verifier"}).json()[
            "thread_id"
        ]
        done = c.post(f"/runs/{thread}/resume", json={"action": "reject"}).json()
        assert done["abstained"] and done["answer"] == []


def test_bad_requests(client):
    assert client.post("/ask", json={"uid": EXAMPLE.uid, "arm": "nope"}).status_code == 400
    assert client.post("/ask", json={"uid": "nope", "arm": "single_agent"}).status_code == 404
    long = {"uid": EXAMPLE.uid, "arm": "single_agent", "question": "x" * 500}
    assert client.post("/ask", json=long).status_code == 400
    assert client.get("/runs/nope").status_code == 404


def test_rate_limit_and_spend_cap():
    with make_client(A_TURNS, rate_capacity=5, fresh_cost=5, rate_refill_per_s=0.001) as c:
        assert c.post("/ask", json={"uid": EXAMPLE.uid, "arm": "single_agent"}).status_code == 200
        other = {"uid": EXAMPLE.uid, "arm": "single_agent", "question": "Something else?"}
        limited = c.post("/ask", json=other)
        assert limited.status_code == 429 and int(limited.headers["Retry-After"]) >= 1

    with make_client(A_TURNS, daily_spend_cap_usd=0.0) as c:
        capped = c.post("/ask", json={"uid": EXAMPLE.uid, "arm": "single_agent"})
        assert capped.status_code == 503


def test_behind_cloudfront_the_rate_limit_keys_on_the_address_not_the_port():
    settings = {"rate_capacity": 5, "fresh_cost": 5, "rate_refill_per_s": 0.001}
    with make_client(A_TURNS, client_ip_header="CloudFront-Viewer-Address", **settings) as c:
        first = {"uid": EXAMPLE.uid, "arm": "single_agent"}
        headers = {"CloudFront-Viewer-Address": "203.0.113.9:51234"}
        assert c.post("/ask", json=first, headers=headers).status_code == 200
        # Same viewer, new connection port: still the same client, so limited.
        other = {**first, "question": "Something else?"}
        again = c.post(
            "/ask", json=other, headers={"CloudFront-Viewer-Address": "203.0.113.9:60001"}
        )
        assert again.status_code == 429
