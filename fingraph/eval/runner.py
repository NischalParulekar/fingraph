"""Run one arm over a frozen split and score it with the official TAT-QA scorer.

    python -m fingraph.eval.runner --arm single_agent --split dev --max-cost 0.20
    python -m fingraph.eval.runner --arm single_agent --split dev --limit 10 --max-cost 0.05

Writes to results/<split>[-first<limit>]/:
- <arm>.predictions.jsonl  one line per question: the answer, its scores and the full
                           trace (every model call and tool call), so anyone can re-score
                           it or audit a single answer;
- <arm>.summary.json       the arm's config and the metrics defined in METRICS.md, with
                           Wilson 95% intervals, overall and per answer_from/answer_type.

The run stops before the next question once --max-cost is spent; a stopped run is
marked incomplete in its summary.
"""

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from fingraph.arms.base import Arm, Prediction
from fingraph.data.splits import load_split
from fingraph.data.tatqa import Example
from fingraph.eval.scorer import score
from fingraph.eval.stats import wilson

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _single_agent() -> Arm:
    from dotenv import load_dotenv

    from fingraph.arms.single_agent import SingleAgent

    load_dotenv()
    return SingleAgent()


ARMS: dict[str, Callable[[], Arm]] = {"single_agent": _single_agent}


def record(example: Example, prediction: Prediction) -> dict:
    em, f1 = score(example.gold, prediction.answer, prediction.scale)
    answered = bool(prediction.answer)
    return {
        "uid": example.uid,
        "answer_from": example.answer_from,
        "answer_type": example.answer_type,
        "question": example.question,
        "gold_answer": example.gold["answer"],
        "gold_scale": example.scale,
        "em": em,
        "f1": f1,
        "answered": answered,
        "confident_wrong": answered and em == 0,
        **asdict(prediction),
    }


def summarize(records: list[dict]) -> dict:
    """The METRICS.md metrics for a list of scored records."""
    n = len(records)
    if n == 0:
        return {"n": 0}
    answered = [r for r in records if r["answered"]]
    correct = sum(r["em"] == 1 for r in records)
    confident_wrong = sum(r["confident_wrong"] for r in records)
    latencies = sorted(r["latency_ms"] for r in records)

    def proportion(k: int, total: int) -> dict:
        low, high = wilson(k, total)
        return {"value": k / total if total else None, "k": k, "n": total, "ci95": [low, high]}

    return {
        "n": n,
        "em": proportion(correct, n),
        "f1": sum(r["f1"] for r in records) / n,
        "confident_wrong": proportion(confident_wrong, n),
        "abstention": proportion(n - len(answered), n),
        "selective_accuracy": proportion(correct, len(answered)),
        "cost_per_1k_usd": 1000 * sum(r["cost_usd"] for r in records) / n,
        "latency_ms_p50": _percentile(latencies, 50),
        "latency_ms_p95": _percentile(latencies, 95),
        "llm_calls_per_question": sum(r["llm_calls"] for r in records) / n,
        "errors": dict(_count(r["error"] for r in records if r["error"])),
    }


def _count(values) -> dict:
    counts: dict = defaultdict(int)
    for v in values:
        counts[v] += 1
    return dict(sorted(counts.items()))


def _percentile(ordered: list[float], q: float) -> float:
    position = (len(ordered) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def run(arm: Arm, examples: list[Example], max_cost: float, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    records, spent = [], 0.0
    with open(out / f"{arm.name}.predictions.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i, example in enumerate(examples):
            if spent >= max_cost:
                print(f"stopped at --max-cost ${max_cost} after {i} questions", file=sys.stderr)
                break
            r = record(example, arm.answer(example))
            spent += r["cost_usd"]
            records.append(r)
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            print(f"{i + 1}/{len(examples)} em={r['em']:.0f} ${spent:.4f}", file=sys.stderr)

    by_group = {
        field: {
            value: summarize([r for r in records if r[field] == value])
            for value in sorted({r[field] for r in records})
        }
        for field in ("answer_from", "answer_type")
    }
    summary = {
        "arm": arm.name,
        "config": arm.config,
        "complete": len(records) == len(examples),
        "planned_questions": len(examples),
        "overall": summarize(records),
        **by_group,
    }
    (out / f"{arm.name}.summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--split", choices=["dev", "heldout"], required=True)
    parser.add_argument("--limit", type=int, help="only the first N questions of the split")
    parser.add_argument("--max-cost", type=float, required=True, help="USD; stop once spent")
    args = parser.parse_args()

    examples = load_split(args.split)
    name = args.split
    if args.limit:
        examples, name = examples[: args.limit], f"{args.split}-first{args.limit}"
    summary = run(ARMS[args.arm](), examples, args.max_cost, RESULTS_DIR / name)
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
