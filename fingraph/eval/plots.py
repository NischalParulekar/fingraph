"""The two held-out plots, from the committed run files. No model calls.

    python -m fingraph.eval.plots results/heldout

- em_by_type.png: EM per arm, by answer_type and by answer_from.
- coverage.png and coverage.json: Arm C's coverage (share answered) against selective
  accuracy (EM among answered), as the retry budget and the verifier's strictness vary.

The coverage points are replayed from Arm C's traces, not rerun: attempt k of a
question is the same whatever the budget, so a budget of r retries answers with the
first attempt <= r that passed and abstains otherwise. The strict verifier (strict=True,
which adds the missing-scale rule) is replayed at budget 0 only: a strict rejection
would trigger a retry whose outcome the traces do not hold. Arm B, with no verifier,
is the 100%-coverage point.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fingraph.data.tatqa import load  # noqa: E402
from fingraph.eval.offline_verify import inputs_from_trace  # noqa: E402
from fingraph.eval.scorer import score  # noqa: E402
from fingraph.verifier import unit_hints, verify  # noqa: E402

# Reference palette (dataviz skill), fixed order, validated for the light surface.
SURFACE, INK, INK_2, MUTED, GRID, BASE = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
SERIES = {"single_agent": "#2a78d6", "graph": "#eb6834", "graph_verifier": "#1baf7a"}
LABELS = {
    "single_agent": "A: single agent",
    "graph": "B: graph",
    "graph_verifier": "C: graph + verifier",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load_runs(directory: Path) -> dict[str, list[dict]]:
    runs = {}
    for arm in SERIES:
        with open(directory / f"{arm}.predictions.jsonl", encoding="utf-8") as f:
            runs[arm] = [json.loads(line) for line in f]
    return runs


def em_by_type(runs: dict, out: Path) -> None:
    groups = [("answer_type", v) for v in ("arithmetic", "span", "multi-span", "count")]
    groups += [("answer_from", v) for v in ("table", "table-text", "text")]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150, facecolor=SURFACE)
    _style(ax)
    width = 0.26
    for i, arm in enumerate(SERIES):
        values = []
        for field, value in groups:
            rows = [r for r in runs[arm] if r[field] == value]
            values.append(100 * sum(r["em"] for r in rows) / len(rows))
        xs = [g + (i - 1) * (width + 0.02) for g in range(len(groups))]
        bars = ax.bar(xs, values, width, color=SERIES[arm], label=LABELS[arm], zorder=2)
        for bar, v in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 1,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK_2,
            )
    n = {g: sum(r[g[0]] == g[1] for r in runs["single_agent"]) for g in groups}
    ax.set_xticks(range(len(groups)), [f"{v}\nn={n[(f, v)]}" for f, v in groups])
    ax.axvline(3.5, color=GRID, linewidth=1)
    ax.text(1.5, 103, "by answer type", ha="center", fontsize=9, color=MUTED)
    ax.text(5, 103, "by answer source", ha="center", fontsize=9, color=MUTED)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Exact match (%)", color=INK_2, fontsize=9)
    ax.set_title(
        "Held-out exact match by question type (500 questions)",
        loc="left",
        color=INK,
        fontsize=11,
        pad=18,
    )
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0, 1.1),
        fontsize=9,
        labelcolor=INK_2,
    )
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def coverage_points(runs: dict) -> list[dict]:
    examples = load("test")
    points = []
    graph = runs["graph"]
    points.append(_point("B: no verifier", [(r["em"], True) for r in graph]))
    c = runs["graph_verifier"]
    for budget in (0, 1, 2):
        outcomes = []
        for r in c:
            reports = r["trace"][-1].get("verifier_reports") or []
            first_pass = next(
                (rep for rep in reports if rep["passed"] and rep["attempt"] <= budget), None
            )
            if first_pass is None:
                outcomes.append((0.0, False))
            else:
                em, _ = score(examples[r["uid"]].gold, first_pass["answer"], first_pass["scale"])
                outcomes.append((em, True))
        points.append(
            _point(f"C: {budget} retr{'y' if budget == 1 else 'ies'}", outcomes, budget=budget)
        )
    strict = []
    for r in c:
        state = r["trace"][-1]
        reports = state.get("verifier_reports") or []
        if not (reports and reports[0]["passed"] and len(reports) == 1):
            strict.append((0.0, False))
            continue
        example = examples[r["uid"]]
        evidence, calculations = inputs_from_trace(r["trace"])
        hints = unit_hints(
            [x for row in example.table for x in row], [p.text for p in example.paragraphs]
        )
        rep = reports[0]
        report = verify(rep["answer"], rep["scale"], evidence, calculations, hints, strict=True)
        strict.append(
            (score(example.gold, rep["answer"], rep["scale"])[0], True)
            if report.passed
            else (0.0, False)
        )
    points.append(_point("C: 0 retries, strict", strict, budget=0, strict=True))
    return points


def _point(name: str, outcomes: list[tuple[float, bool]], budget=None, strict=False) -> dict:
    answered = [em for em, a in outcomes if a]
    return {
        "name": name,
        "retry_budget": budget,
        "strict": strict,
        "n": len(outcomes),
        "answered": len(answered),
        "coverage": len(answered) / len(outcomes),
        "selective_accuracy": sum(answered) / len(answered) if answered else None,
        "em": sum(answered) / len(outcomes),
        "confident_wrong": sum(1 for em, a in outcomes if a and em == 0) / len(outcomes),
    }


# Where each point's label sits, so the close-together retry points do not collide.
LABEL_OFFSETS = {
    "B: no verifier": ((-10, -3), "right"),
    "C: 0 retries": ((0, -14), "center"),
    "C: 1 retry": ((-8, -12), "right"),
    "C: 2 retries": ((-6, 8), "right"),
}


def coverage_plot(points: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150, facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    c = [p for p in points if p["name"].startswith("C") and not p["strict"]]
    ax.plot(
        [100 * p["coverage"] for p in c],
        [100 * p["selective_accuracy"] for p in c],
        color=SERIES["graph_verifier"],
        linewidth=2,
        marker="o",
        markersize=8,
        zorder=3,
        label="C: retry budget 0, 1, 2",
    )
    for p in points:
        color = SERIES["graph"] if p["name"].startswith("B") else SERIES["graph_verifier"]
        marker = "s" if p["name"].startswith("B") else ("D" if p["strict"] else "o")
        ax.plot(
            100 * p["coverage"],
            100 * p["selective_accuracy"],
            marker=marker,
            markersize=8,
            color=color,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            zorder=4,
        )
        offset, align = LABEL_OFFSETS.get(p["name"], ((8, -3), "left"))
        ax.annotate(
            p["name"],
            (100 * p["coverage"], 100 * p["selective_accuracy"]),
            textcoords="offset points",
            xytext=offset,
            ha=align,
            fontsize=8,
            color=INK_2,
        )
    ax.set_xlabel("Coverage: questions answered (%)", color=INK_2, fontsize=9)
    ax.set_ylabel("Selective accuracy: EM among answered (%)", color=INK_2, fontsize=9)
    ax.set_title(
        "Arm C: coverage against accuracy among answered", loc="left", color=INK, fontsize=11
    )
    xs = [100 * p["coverage"] for p in points]
    ys = [100 * p["selective_accuracy"] for p in points]
    ax.set_xlim(min(xs) - 2, 101.5)
    ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    runs = load_runs(args.directory)
    plots = args.directory / "plots"
    plots.mkdir(exist_ok=True)
    em_by_type(runs, plots / "em_by_type.png")
    points = coverage_points(runs)
    (args.directory / "coverage.json").write_text(
        json.dumps(points, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    coverage_plot(points, plots / "coverage.png")
    for p in points:
        print(
            f"{p['name']:24} coverage {100 * p['coverage']:5.1f}%  selective "
            f"{100 * p['selective_accuracy']:5.1f}%  EM {100 * p['em']:5.1f}%  "
            f"confident-wrong {100 * p['confident_wrong']:5.1f}%"
        )


if __name__ == "__main__":
    main()
