# Metric definitions

Shared with the planned false-completion benchmark, which reuses the confident-wrong
definition. Change a definition only with a dated note here, never silently.

Every question ends in exactly one of three outcomes:

- **abstained**: the system returned no answer (an empty answer, an explicit abstention,
  a budget or API failure). Failures count as abstentions and are also counted
  separately by cause.
- **answered and correct**: official TAT-QA exact match (EM) = 1.
- **answered and wrong**: answered, EM = 0.

With N questions in the set:

| Metric | Definition |
|---|---|
| EM | Official TAT-QA exact match, averaged over all N (abstentions score 0). Scale-aware. |
| F1 | Official TAT-QA numeracy-focused F1, averaged over all N. For arithmetic and count questions the official script sets F1 = EM. |
| Confident-wrong rate | answered and wrong ÷ N. |
| Abstention rate | abstained ÷ N. |
| Selective accuracy | answered and correct ÷ answered. Undefined if nothing is answered. |
| Unsupported-number rate | answers containing a number that is neither in the evidence the system gathered nor the result of a logged calculator call, ÷ answered. Computed offline with the same verifier code for every arm. |
| Cost per 1k questions | Sum over questions of billed tokens × list price, × 1000 ÷ N. Cache reads and writes are priced at their own rates. |
| Latency p50 / p95 | Wall-clock time per question, end to end. |
| LLM calls per question | Mean number of model requests per question. |

Proportions are reported with Wilson 95% intervals. Paired comparisons between arms use
the two-sided exact sign test over questions where the two arms differ.

## Notes

- 2026-11-20: the held-out report (`fingraph/eval/report.py`) also shows cost per 1k
  questions with caching off, and "billed if run alone" for the graph arms (their cache
  reads, which can only come from a concurrent run, repriced as writes). Cost per 1k as
  defined above is unchanged and is what H3 is judged on.
