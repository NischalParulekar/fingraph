"""The held-out results table and the tests of H1-H3, from committed run files.

    python -m fingraph.eval.report results/heldout

Reads <arm>.predictions.jsonl for single_agent, graph and graph_verifier and writes
report.json and report.md next to them. Everything here is computed from those files,
so `make report` reproduces the README's results section.

Tests (as fixed in the README before any held-out run):
- H1: A vs B on EM, paired two-sided exact sign test; supported if B > A with p < 0.05.
- H2: B vs C on confident-wrong; supported if C has fewer with p < 0.05. Also reports
  the abstention rate and tokens per question it costs.
- H3: B vs A cost per question (descriptive), and the ratio.
"""

import argparse
import json
from pathlib import Path

from fingraph.eval.runner import summarize
from fingraph.eval.stats import paired
from fingraph.llm import CACHE_WRITE, PRICES

ARMS = {"A": "single_agent", "B": "graph", "C": "graph_verifier"}
LABELS = {"A": "A: single agent", "B": "B: graph", "C": "C: graph + verifier"}


def load(directory: Path) -> dict[str, dict[str, dict]]:
    runs = {}
    for key, name in ARMS.items():
        path = directory / f"{name}.predictions.jsonl"
        with open(path, encoding="utf-8") as f:
            runs[key] = {r["uid"]: r for r in map(json.loads, f)}
    uids = set(runs["A"])
    if any(set(r) != uids for r in runs.values()):
        raise ValueError("the runs did not answer the same questions")
    return runs


def cost_views(records: list[dict], one_off_prompts: bool) -> dict:
    """Cost per 1k questions three ways, from the token counts in the traces.

    - billed: as METRICS.md defines it (cache reads and writes at their own rates);
    - standalone: what the run would be billed on its own. For the graph arms
      (one_off_prompts) every cache read is repriced as a cache write: their prompts are
      one-off, so a read can only come from another run sending the identical prompt at
      the same time (B and C ran in parallel). Arm A's reads come from its own loop
      re-sending its conversation, so for A this equals billed;
    - no_cache: every input token at the plain input price, as with caching switched off.
    """
    totals = {"in": 0, "out": 0, "read": 0, "write": 0}
    for r in records:
        for step in r["trace"]:
            if "input_tokens" in step:
                totals["in"] += step["input_tokens"]
                totals["out"] += step["output_tokens"]
                totals["read"] += step["cache_read_tokens"]
                totals["write"] += step["cache_write_tokens"]
    model = records[0].get("model") or "claude-haiku-5-5"
    price_in, price_out = PRICES[model]
    out = totals["out"] * price_out
    per_1k = 1000 / len(records) / 1_000_000
    standalone = (totals["in"] + (totals["read"] + totals["write"]) * CACHE_WRITE) * price_in
    no_cache = (totals["in"] + totals["read"] + totals["write"]) * price_in
    billed = sum(r["cost_usd"] for r in records) * 1000 / len(records)
    return {
        "billed": billed,
        "standalone": (standalone + out) * per_1k if one_off_prompts else billed,
        "no_cache": (no_cache + out) * per_1k,
        "cache_read_tokens_per_question": totals["read"] / len(records),
    }


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def ci(p: dict) -> str:
    low, high = p["ci95"]
    return f"{pct(p['value'])} ({100 * low:.1f}-{100 * high:.1f})"


def build(runs: dict) -> dict:
    summaries = {k: summarize(list(r.values())) for k, r in runs.items()}
    tokens = {k: sum(x["tokens"] for x in r.values()) / len(r) for k, r in runs.items()}
    em = {k: {u: x["em"] == 1 for u, x in r.items()} for k, r in runs.items()}
    not_cw = {k: {u: not x["confident_wrong"] for u, x in r.items()} for k, r in runs.items()}
    h1 = paired(em["A"], em["B"])
    h2 = paired(not_cw["B"], not_cw["C"])
    cost = {k: s["cost_per_1k_usd"] for k, s in summaries.items()}
    views = {k: cost_views(list(r.values()), one_off_prompts=k != "A") for k, r in runs.items()}
    groups = {}
    for field in ("answer_from", "answer_type"):
        values = sorted({x[field] for x in runs["A"].values()})
        groups[field] = {
            v: {k: summarize([x for x in r.values() if x[field] == v]) for k, r in runs.items()}
            for v in values
        }
    return {
        "n": len(runs["A"]),
        "summaries": summaries,
        "tokens_per_question": tokens,
        "groups": groups,
        "h1": {**h1, "supported": h1["wins"] > h1["losses"] and h1["p_value"] < 0.05},
        "h2": {**h2, "supported": h2["wins"] > h2["losses"] and h2["p_value"] < 0.05},
        "h3": {
            "cost_per_1k_usd": cost,
            "b_over_a": cost["B"] / cost["A"],
            "c_over_a": cost["C"] / cost["A"],
            "supported": cost["B"] > cost["A"],
        },
        "cost_views": views,
        "a_vs_c_em": paired(em["A"], em["C"]),
        "a_vs_c_confident_wrong": paired(not_cw["A"], not_cw["C"]),
    }


def markdown(r: dict) -> str:
    s = r["summaries"]
    lines = [
        f"Held-out set: {r['n']} questions from the TAT-QA test split. EM, confident-wrong "
        "and abstention with Wilson 95% intervals.",
        "",
        "| Arm | EM | F1 | Confident-wrong | Abstain | Selective acc. | Cost / 1k q "
        "| Tokens / q | LLM calls / q | p50 / p95 latency |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k in ARMS:
        x = s[k]
        lines.append(
            f"| {LABELS[k]} | {ci(x['em'])} | {x['f1']:.3f} | {ci(x['confident_wrong'])} "
            f"| {ci(x['abstention'])} | {pct(x['selective_accuracy']['value'])} "
            f"| ${x['cost_per_1k_usd']:.2f} | {r['tokens_per_question'][k]:,.0f} "
            f"| {x['llm_calls_per_question']:.2f} "
            f"| {x['latency_ms_p50'] / 1000:.1f} / {x['latency_ms_p95'] / 1000:.1f} s |"
        )

    def verdict(h):
        return "supported" if h["supported"] else "not supported"

    h1, h2, h3 = r["h1"], r["h2"], r["h3"]
    lines += [
        "",
        f"- **H1 (B beats A on EM): {verdict(h1)}.** Paired over {h1['n']} questions: B right "
        f"and A wrong on {h1['wins']}, A right and B wrong on {h1['losses']}, {h1['ties']} "
        f"ties; exact sign test p = {h1['p_value']:.3g}.",
        f"- **H2 (C has fewer confident-wrong answers than B): {verdict(h2)}.** "
        f"{h2['wins']} questions confidently wrong in B but not in C, {h2['losses']} the "
        f"other way, {h2['ties']} ties; p = {h2['p_value']:.3g}. The cost: abstention "
        f"{pct(s['C']['abstention']['value'])} vs {pct(s['B']['abstention']['value'])}, "
        f"{r['tokens_per_question']['C']:,.0f} vs {r['tokens_per_question']['B']:,.0f} "
        "tokens per question.",
        f"- **H3 (B costs more than A): {verdict(h3)}.** B costs {h3['b_over_a']:.2f}x A per "
        f"question (${h3['cost_per_1k_usd']['B']:.2f} vs ${h3['cost_per_1k_usd']['A']:.2f} "
        f"per 1k), C {h3['c_over_a']:.2f}x as billed (see the cost table below).",
        f"- A vs C, not a pre-registered test: EM {r['a_vs_c_em']['wins']} C-only right vs "
        f"{r['a_vs_c_em']['losses']} A-only right (p = {r['a_vs_c_em']['p_value']:.3g}); "
        f"confident-wrong {r['a_vs_c_confident_wrong']['wins']} fewer in C vs "
        f"{r['a_vs_c_confident_wrong']['losses']} more (p = "
        f"{r['a_vs_c_confident_wrong']['p_value']:.3g}).",
        "",
        "Cost per 1k questions, three ways (see `cost_views` in fingraph/eval/report.py):",
        "",
        "| Arm | Billed | Billed if run alone | With caching off | Cache-read tokens / q |",
        "|---|---|---|---|---|",
        *[
            f"| {LABELS[k]} | ${v['billed']:.3f} | ${v['standalone']:.3f} "
            f"| ${v['no_cache']:.3f} | {v['cache_read_tokens_per_question']:,.0f} |"
            for k, v in r["cost_views"].items()
        ],
        "",
        "EM by question type (n per row):",
        "",
        "| Group | n | A | B | C |",
        "|---|---|---|---|---|",
    ]
    for field, values in r["groups"].items():
        for value, by_arm in values.items():
            n = by_arm["A"]["n"]
            cells = " | ".join(pct(by_arm[k]["em"]["value"]) for k in ARMS)
            lines.append(f"| {field} = {value} | {n} | {cells} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = build(load(args.directory))
    (args.directory / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    text = markdown(report)
    (args.directory / "report.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
