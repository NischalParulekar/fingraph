"""Arm C on dev through the direct API vs through Bedrock, from committed run files.

    python -m fingraph.eval.bedrock_compare results/bedrock-dev

Reads graph_verifier-api.* and graph_verifier-bedrock.* (made by `make bedrock-dev`) and
writes report.json and report.md next to them. Dev subset only: this is not the frozen
held-out evaluation.

Interpretation rule (fixed before the runs, from the Bedrock spec): identical dev runs
flip about 3% of questions, so an EM or confident-wrong difference under 3 points is
reported as no measurable difference.
"""

import argparse
import json
from pathlib import Path

from fingraph.eval.report import ci, cost_views, pct
from fingraph.eval.runner import summarize
from fingraph.eval.stats import paired
from fingraph.llm import prices

RUNS = {"api": "Anthropic API", "bedrock": "Amazon Bedrock"}
NOISE = 0.03


def load(directory: Path, run: str) -> tuple[dict, list[dict]]:
    stem = directory / f"graph_verifier-{run}"
    summary = json.loads(stem.with_suffix(".summary.json").read_text(encoding="utf-8"))
    with open(f"{stem}.predictions.jsonl", encoding="utf-8") as f:
        return summary, [json.loads(line) for line in f]


def build(directory: Path) -> dict:
    runs = {run: load(directory, run) for run in RUNS}
    out = {"runs": {}}
    for run, (summary, records) in runs.items():
        steps = [s for r in records for s in r["trace"] if "input_tokens" in s]
        views = cost_views(records, one_off_prompts=False, price=prices(summary["config"]))
        out["runs"][run] = {
            "config": summary["config"],
            "complete": summary["complete"],
            "summary": summarize(records),
            "cache_read_tokens_per_question": views["cache_read_tokens_per_question"],
            "llm_calls": len(steps),
            "retried_calls": sum(1 for s in steps if s.get("retries")),
            "retries": sum(s.get("retries", 0) for s in steps),
        }
    api, bedrock = ({r["uid"]: r for r in runs[k][1]} for k in RUNS)
    if set(api) != set(bedrock):
        raise ValueError("the two runs did not answer the same questions")
    em = paired({u: api[u]["em"] == 1 for u in api}, {u: bedrock[u]["em"] == 1 for u in api})
    not_cw = paired(
        {u: not api[u]["confident_wrong"] for u in api},
        {u: not bedrock[u]["confident_wrong"] for u in api},
    )
    a, b = out["runs"]["api"]["summary"], out["runs"]["bedrock"]["summary"]
    out["em"] = {**em, "difference": b["em"]["value"] - a["em"]["value"]}
    out["confident_wrong"] = {
        **not_cw,
        "difference": b["confident_wrong"]["value"] - a["confident_wrong"]["value"],
    }
    out["flipped"] = sum(api[u]["em"] != bedrock[u]["em"] for u in api)
    return out


def verdict(difference: float) -> str:
    if abs(difference) < NOISE:
        return "no measurable difference"
    direction = "higher" if difference > 0 else "lower"
    return f"{direction} on Bedrock by {100 * abs(difference):.1f} points"


def markdown(r: dict) -> str:
    rows = r["runs"]
    n = rows["api"]["summary"]["n"]
    lines = [
        f"Arm C (graph + verifier) on the {n}-question dev subset, once through each "
        "provider, one after the other. **Dev subset only, not the frozen held-out "
        "evaluation.** EM, confident-wrong and abstention with Wilson 95% intervals.",
        "",
        "| Provider | EM | F1 | Confident-wrong | Abstain | p50 / p95 latency | Cost / 1k q "
        "| Cache-read tokens / q | Errors and abstentions | Retried calls |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run, label in RUNS.items():
        x, s = rows[run], rows[run]["summary"]
        errors = ", ".join(f"{k}: {v}" for k, v in s["errors"].items()) or "none"
        lines.append(
            f"| {label} | {ci(s['em'])} | {s['f1']:.3f} | {ci(s['confident_wrong'])} "
            f"| {ci(s['abstention'])} "
            f"| {s['latency_ms_p50'] / 1000:.1f} / {s['latency_ms_p95'] / 1000:.1f} s "
            f"| ${s['cost_per_1k_usd']:.2f} | {x['cache_read_tokens_per_question']:,.0f} "
            f"| {errors} | {x['retried_calls']} of {x['llm_calls']} |"
        )
    em, cw = r["em"], r["confident_wrong"]
    lines += [
        "",
        f"- **EM: {verdict(em['difference'])}** ({pct(rows['bedrock']['summary']['em']['value'])} "
        f"vs {pct(rows['api']['summary']['em']['value'])}). Paired: {em['wins']} questions "
        f"right only on Bedrock, {em['losses']} right only on the API "
        f"(sign test p = {em['p_value']:.3g}); {r['flipped']} of {em['n']} changed.",
        f"- **Confident-wrong: {verdict(cw['difference'])}** "
        f"({pct(rows['bedrock']['summary']['confident_wrong']['value'])} vs "
        f"{pct(rows['api']['summary']['confident_wrong']['value'])}).",
        f"- Bedrock: `{rows['bedrock']['config'].get('bedrock_model_id')}` from "
        f"`{rows['bedrock']['config'].get('aws_region')}`; "
        f"{rows['bedrock']['retries']} retries after throttling or a transient error.",
        "",
        "Differences under 3 points are within the run-to-run noise measured on dev "
        "(about 3% of questions flip between identical runs).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = build(args.directory)
    (args.directory / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    text = markdown(report)
    (args.directory / "report.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
