# Development runs (150-question dev subset)

Dev is used freely for building, so these numbers are optimistic: the prompts were
changed after looking at these very questions. Only held-out runs go in the main README.

| Run | Change | EM | F1 | Confident-wrong | Cost / 1k q |
|---|---|---|---|---|---|
| `single_agent` (v1) | first version | 103/150 = 68.7% | 0.773 | 31.3% | $0.45 |
| `single_agent-v2` | answer rules v2 (scale, span copying, sign), always search the text | 116/150 = 77.3% | 0.851 | 22.7% | $0.55 |
| `single_agent-v2-rerun` | none: v2 run again, to measure noise | 114/150 = 76.0% | 0.838 | 24.0% | $0.55 |
| `graph` (Arm B v1) | first version of the graph, same answer rules v2 | 88/150 = 58.7% | 0.637 | 41.3% | $0.84 |

v1 → v2, paired on EM: 17 wins, 4 losses, 129 ties, sign test p = 0.007.

**Run-to-run noise.** Haiku 5.5 takes no temperature, so two identical runs differ. v2
and its unchanged rerun gave a different answer or scale on 24 of 150 questions, but
only 4 changed correctness (1 win, 3 losses, p = 0.63): 113 right in both, 117 right in
at least one. So about 3% of questions flip between identical runs, and a difference
of 2 points in EM between two arms on 150 questions means nothing. The v1 → v2 gain
holds against the rerun as well (15 wins, 4 losses, p = 0.019).

The v2 rules are shared by every arm (`fingraph/arms/base.py`), so they do not favour
Arm A. The conventions behind them were measured on the whole official dev split, never
on held-out.

**Arm B v1 vs Arm A v2** (paired on EM): 4 wins, 32 losses, 114 ties, p < 0.0001. The
losses are in questions that need the text (table-text EM 41% vs 78%, text 37% vs 51%;
table alone 81% vs 90%). Two design faults in the graph, not in the model: the planner,
which cannot see the paragraphs, asked for a text lookup on only 31 of 150 questions
although 81 need the text; and it sometimes named empty cells (8 failed calculations).
Arm B v1 has had no iteration yet, while Arm A has had one round.
