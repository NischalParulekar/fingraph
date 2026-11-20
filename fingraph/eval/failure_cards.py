"""Sample failures per arm and print each as a compact card for labelling.

    python -m fingraph.eval.failure_cards results/heldout --n 50 > cards.md

A failure is a question with EM 0, abstentions included. The sample is drawn with a
fixed seed per arm, so `error_labels.csv` can always be traced back to the same
questions. Each card shows the question, the gold answer and its derivation, the
prediction, and what the arm read and computed, which is enough to pick a bucket.
"""

import argparse
import json
import random
from pathlib import Path

from fingraph.data.tatqa import load

ARMS = ("single_agent", "graph", "graph_verifier")
SEED = 42
CUT = 260


def sample(records: list[dict], n: int, arm: str) -> list[dict]:
    failures = sorted((r for r in records if r["em"] != 1), key=lambda r: r["uid"])
    return random.Random(f"{SEED}-{arm}").sample(failures, min(n, len(failures)))


def _short(text, limit=CUT) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def card(r: dict, example) -> list[str]:
    g = example.gold
    lines = [
        f"### {r['uid']}  [{r['answer_from']}/{r['answer_type']}]",
        f"Q: {r['question']}",
        f"GOLD: {g['answer']} scale={g['scale']!r} deriv={_short(g.get('derivation', ''), 120)!r} "
        f"rel_paras={g.get('rel_paragraphs')}",
        f"PRED: {r['answer']} scale={r['scale']!r} f1={r['f1']} error={r['error']}",
    ]
    for p in example.paragraphs:
        if str(p.order) in g.get("rel_paragraphs", []):
            lines.append(f"  gold para {p.order}: {_short(p.text)}")
    for step in r["trace"]:
        if step.get("label") == "graph_state":
            plan = step.get("plan") or {}
            cells = [(c["name"], c["row"], c["column"]) for c in plan.get("table_cells", [])]
            lookups = [c["what"][:60] for c in plan.get("text_lookups", [])]
            lines.append(
                f"  plan: cells={cells} lookups={lookups} "
                f"expr={plan.get('expression')!r} scale={plan.get('scale')!r}"
            )
            final_attempt = step.get("attempt") or 0
            for e in step.get("evidence") or []:
                if e.get("attempt", 0) != final_attempt:
                    continue
                where = (
                    f"r{e['row']}c{e['column']}" if e["source"] == "table" else f"p{e['paragraph']}"
                )
                lines.append(
                    f"  ev {e['name']} [{where}]: {_short(e['text'], 100)!r} {e.get('error') or ''}"
                )
            calc = step.get("calculation") or {}
            if calc:
                expression = calc.get("substituted") or calc.get("expression")
                lines.append(
                    f"  calc: {expression} = {calc.get('result')} {calc.get('error') or ''}"
                )
            for rep in step.get("verifier_reports") or []:
                bad = [c["value"] for c in rep["unsupported"]] + rep["scale_problems"]
                verdict = "pass" if rep["passed"] else "FAIL " + _short(bad, 150)
                lines.append(
                    f"  verifier[{rep['attempt']}] {rep['answer']} {rep['scale']!r}: {verdict}"
                )
            continue
        for call in step.get("tool_calls", []):
            if call["name"] == "final_answer":
                continue
            arg = next(iter(call["input"].values()), "")
            lines.append(
                f"  {call['name']}({_short(arg, 120)}) -> {_short(call.get('result', ''), 200)}"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    examples = load("test")
    for arm in [args.arm] if args.arm else ARMS:
        with open(args.directory / f"{arm}.predictions.jsonl", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        print(f"## {arm}\n")
        for r in sample(records, args.n, arm):
            print("\n".join(card(r, examples[r["uid"]])) + "\n")


if __name__ == "__main__":
    main()
