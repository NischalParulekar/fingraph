"""How well retrieval finds the paragraphs TAT-QA marks as relevant. No model calls.

    python -m fingraph.eval.retrieval --split dev-official

For every question with gold relevant paragraphs (`rel_paragraphs`), the question
itself is the query, and three retrievers are compared at k = 1 and k = 3:
- memory-context:   the in-memory ContextRetriever over the question's report (the arms);
- pgvector-context: pgvector filtered to the question's report;
- pgvector-corpus:  pgvector over all 2,635 paragraphs of the dev and test reports.

hit@k is the share of questions with at least one gold paragraph in the top k;
recall@k the share of gold paragraphs found in the top k. A corpus hit counts only if
it is from the question's own report. Also reports how often the two context-mode
retrievers return the same top-k paragraphs in the same order.
"""

import argparse
import json
from pathlib import Path

from fingraph.data.splits import load_split
from fingraph.data.tatqa import load
from fingraph.eval.stats import wilson
from fingraph.tools.pgvector_store import PgVectorRetriever, store
from fingraph.tools.retriever import ContextRetriever

RESULTS = Path(__file__).resolve().parents[2] / "results" / "retrieval"
KS = (1, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "dev-official"], default="dev-official")
    args = parser.parse_args()
    examples = load_split("dev") if args.split == "dev" else list(load("dev").values())
    examples = [e for e in examples if e.gold.get("rel_paragraphs")]

    vector_store = store()
    counts = {
        m: {k: {"hit": 0, "found": 0} for k in KS} for m in ("memory", "pg_context", "pg_corpus")
    }
    gold_total, same_order = 0, 0
    memory_cache: dict[str, ContextRetriever] = {}
    for e in examples:
        gold = {int(p) for p in e.gold["rel_paragraphs"]}
        gold_total += len(gold)
        if e.context_uid not in memory_cache:
            memory_cache = {e.context_uid: ContextRetriever(e.paragraphs)}
        memory = [p.order for p, _ in memory_cache[e.context_uid].search(e.question, max(KS))]
        pg_context = [
            p.order
            for p, _, _ in PgVectorRetriever(vector_store, e.context_uid).search(
                e.question, max(KS)
            )
        ]
        pg_corpus = [
            p.order if c == e.context_uid else None
            for p, _, c in PgVectorRetriever(vector_store, e.context_uid, "corpus").search(
                e.question, max(KS)
            )
        ]
        same_order += memory == pg_context
        for name, ranked in (
            ("memory", memory),
            ("pg_context", pg_context),
            ("pg_corpus", pg_corpus),
        ):
            for k in KS:
                top = set(ranked[:k]) - {None}
                counts[name][k]["hit"] += bool(top & gold)
                counts[name][k]["found"] += len(top & gold)

    n = len(examples)
    results = {}
    for name, by_k in counts.items():
        results[name] = {}
        for k in KS:
            hits = by_k[k]["hit"]
            results[name][f"hit@{k}"] = {"value": hits / n, "ci95": wilson(hits, n)}
            results[name][f"recall@{k}"] = by_k[k]["found"] / gold_total
    summary = {
        "split": args.split,
        "questions": n,
        "gold_paragraphs": gold_total,
        "context_rankings_identical": {"k": same_order, "n": n},
        "results": results,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{args.split}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
