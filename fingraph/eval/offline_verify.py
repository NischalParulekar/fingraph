"""Run the verifier offline over a finished run, for any arm.

    python -m fingraph.eval.offline_verify results/dev/single_agent-v2.predictions.jsonl

Every arm leaves the evidence it gathered in its trace; this turns that into the
verifier's inputs, so the unsupported-number rate (METRICS.md) is computed with the same
code for every arm:

- Arm A: every value a query_table call returned, every paragraph retrieve_text
  returned, and every calculate call (expression and result).
- Arms B/C: the evidence items and the calculation in the final graph state.

Unit hints are the table cells and paragraph sentences that name a unit ("In
thousands", "$'000", ...), taken from the question's report, the same for every arm.

Also reports how the verifier's verdict lines up with correctness, which says how much
it would wrongly reject (correct answers that fail) and catch (wrong answers that fail).
"""

import argparse
import json
import re
from pathlib import Path

from fingraph.data.tatqa import load
from fingraph.verifier import Calculation, Evidence, Report, verify

_UNIT_HINT = re.compile(r"thousand|million|billion|'000|\$000|%|percent", re.IGNORECASE)
SPLIT_SOURCE = {"dev": "dev", "heldout": "test"}


def unit_hints(grid, paragraphs=()) -> list[str]:
    """Table cells and paragraph sentences that name a unit."""
    hints = [cell for row in grid for cell in row if cell and _UNIT_HINT.search(cell)]
    for p in paragraphs:
        hints += [s for s in re.split(r"(?<=[.;])\s+", p.text) if _UNIT_HINT.search(s)]
    return hints


def inputs_from_trace(trace: list[dict]) -> tuple[list[Evidence], list[Calculation]]:
    evidence, calculations = [], []
    for step in trace:
        if step.get("label") == "graph_state":
            for e in step.get("evidence") or []:
                if e.get("text"):
                    where = (
                        f"table row {e['row']}, column c{e['column']}"
                        if e["source"] == "table"
                        else f"paragraph {e['paragraph']}"
                    )
                    evidence.append(Evidence(str(e["text"]), where))
            calc = step.get("calculation") or {}
            if calc.get("result") is not None:
                calculations.append(Calculation(calc["expression"], float(calc["result"])))
            continue
        for call in step.get("tool_calls", []):
            if call.get("error") or "result" not in call:
                continue
            if call["name"] == "query_table":
                for row in json.loads(call["result"])["rows"]:
                    evidence += [
                        Evidence(str(v), "table query result") for v in row if v is not None
                    ]
            elif call["name"] == "retrieve_text":
                for hit in json.loads(call["result"]):
                    evidence.append(Evidence(hit["text"], f"paragraph {hit['paragraph']}"))
            elif call["name"] == "calculate":
                result = float(json.loads(call["result"])["result"])
                calculations.append(Calculation(call["input"]["expression"], result))
    return evidence, calculations


def check(record: dict, example) -> Report:
    evidence, calculations = inputs_from_trace(record["trace"])
    hints = unit_hints(example.table, example.paragraphs)
    return verify(record["answer"], record["scale"], evidence, calculations, hints)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--split", choices=list(SPLIT_SOURCE), default="dev")
    args = parser.parse_args()
    examples = load(SPLIT_SOURCE[args.split])
    records = [json.loads(line) for line in open(args.predictions, encoding="utf-8")]
    answered = [r for r in records if r["answered"]]
    counts = {"correct_pass": 0, "correct_fail": 0, "wrong_pass": 0, "wrong_fail": 0}
    unsupported_number = 0
    for r in answered:
        report = check(r, examples[r["uid"]])
        unsupported_number += report.has_unsupported_number
        key = ("correct" if r["em"] == 1 else "wrong") + ("_pass" if report.passed else "_fail")
        counts[key] += 1
    correct = counts["correct_pass"] + counts["correct_fail"]
    wrong = counts["wrong_pass"] + counts["wrong_fail"]
    print(
        json.dumps(
            {
                "answered": len(answered),
                "unsupported_number": unsupported_number,
                "unsupported_number_rate": unsupported_number / len(answered) if answered else None,
                **counts,
                "false_reject_rate": counts["correct_fail"] / correct if correct else None,
                "catch_rate": counts["wrong_fail"] / wrong if wrong else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
