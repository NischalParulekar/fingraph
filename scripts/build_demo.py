"""Build the recorded demo for the static site from the committed dev runs. No model calls.

    uv run python scripts/build_demo.py

Writes frontend/public/demo.json: a seeded pick of dev-subset questions (dev, never
held-out), each with its report and what every arm answered in the committed runs, in
the same shape the live API returns (fingraph.api.view.run_view). Every answer shown is
exactly what the run recorded, right or wrong.

The pick favours questions that tell the story: some where the arms disagree, some
that Arm C abstained on (in the live API those pause for a reviewer), and the rest
drawn at random.
"""

import json
import random
from pathlib import Path

from fingraph.api.view import run_view
from fingraph.arms.base import Prediction
from fingraph.data.splits import load_split

OUT = Path("frontend/public/demo.json")
RUNS = {  # the dev runs of the frozen arms
    "single_agent": "results/dev/single_agent-v2.predictions.jsonl",
    "graph": "results/dev/graph-v2.predictions.jsonl",
    "graph_verifier": "results/dev/graph_verifier.predictions.jsonl",
}
ARMS = [
    {"arm": "single_agent", "name": "Single agent", "short": "A"},
    {"arm": "graph", "name": "Graph", "short": "B"},
    {"arm": "graph_verifier", "name": "Graph + verifier", "short": "C"},
]
SEED = 7
DISAGREE, ABSTAINED, OTHERS = 5, 2, 7


def load_runs() -> dict[str, dict[str, dict]]:
    runs = {}
    for arm, path in RUNS.items():
        with open(path, encoding="utf-8") as f:
            runs[arm] = {r["uid"]: r for r in map(json.loads, f)}
    return runs


def prediction(record: dict) -> Prediction:
    return Prediction(
        answer=record["answer"],
        scale=record["scale"],
        latency_ms=record["latency_ms"],
        cost_usd=record["cost_usd"],
        tokens=record["tokens"],
        llm_calls=record["llm_calls"],
        error=record["error"],
        trace=record["trace"],
    )


def pick(examples, runs) -> list:
    rng = random.Random(SEED)
    uids = [e.uid for e in examples]
    disagree = [u for u in uids if len({runs[a][u]["em"] for a in RUNS}) > 1]
    abstained = [u for u in uids if not runs["graph_verifier"][u]["answered"]]
    chosen = rng.sample(abstained, ABSTAINED)
    chosen += rng.sample([u for u in disagree if u not in chosen], DISAGREE)
    chosen += rng.sample([u for u in uids if u not in chosen], OTHERS)
    order = {u: i for i, u in enumerate(uids)}
    return sorted(chosen, key=order.get)


def main() -> None:
    examples = load_split("dev")
    by_uid = {e.uid: e for e in examples}
    runs = load_runs()
    questions, reports, answers = [], {}, {}
    for uid in pick(examples, runs):
        e = by_uid[uid]
        questions.append(
            {
                "uid": uid,
                "question": e.question,
                "answer_from": e.answer_from,
                "answer_type": e.answer_type,
            }
        )
        reports[uid] = {
            "uid": uid,
            "question": e.question,
            "table": [list(row) for row in e.table],
            "paragraphs": [{"order": p.order, "text": p.text} for p in e.paragraphs],
            "gold": {"answer": e.gold["answer"], "scale": e.scale},
        }
        answers[uid] = {}
        for arm in RUNS:
            r = runs[arm][uid]
            view = run_view(prediction(r), e)
            state = r["trace"][-1] if r["trace"] else {}
            reports_c = state.get("verifier_reports") or []
            if arm == "graph_verifier" and view["abstained"] and reports_c:
                # What a reviewer would have seen: the last draft and why it failed.
                last = reports_c[-1]
                view["draft"] = {
                    "answer": last["answer"],
                    "scale": last["scale"],
                    "verifier_report": last,
                }
            answers[uid][arm] = {
                **view,
                "status": "done",
                "exact_match": r["em"] == 1,
                "latency_ms": r["latency_ms"],
            }
    bundle = {
        "source": "Committed dev-subset runs of the frozen arms: " + ", ".join(RUNS.values()),
        "arms": ARMS,
        "questions": questions,
        "reports": reports,
        "answers": answers,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bundle, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"{OUT}: {len(questions)} questions, {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
