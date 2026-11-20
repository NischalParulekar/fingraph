# Development runs (150-question dev subset)

Dev is used freely for building, so these numbers are optimistic: the prompts were
changed after looking at these very questions. Only held-out runs go in the main README.

| Run | Change | EM | F1 | Confident-wrong | Cost / 1k q |
|---|---|---|---|---|---|
| `single_agent` (v1) | first version | 103/150 = 68.7% | 0.773 | 31.3% | $0.45 |
| `single_agent-v2` | answer rules v2 (scale, span copying, sign), always search the text | 116/150 = 77.3% | 0.851 | 22.7% | $0.55 |
| `single_agent-v2-rerun` | none: v2 run again, to measure noise | 114/150 = 76.0% | 0.838 | 24.0% | $0.55 |
| `graph` (Arm B v1) | first version of the graph, same answer rules v2 | 88/150 = 58.7% | 0.637 | 41.3% | $0.84 |
| `graph-v2` (Arm B v2) | planner sees the top paragraphs and labelled table rows | 105/150 = 70.0% | 0.790 | 30.0% | $0.80 |
| `graph_verifier` (Arm C) | Arm B v2 + verifier, 2 retries, abstain in eval mode | 108/150 = 72.0% | 0.790 | 24.0% | $0.97 |

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

**Arm B v2** fixes both faults: B v1 → v2 is 26 wins, 9 losses (p = 0.006); table-text
EM rose from 41% to 74%, and failed calculations fell from 8 to 3. Against Arm A v2 it is
still behind: 5 wins, 16 losses (p = 0.027; against A's rerun, 4 and 13, p = 0.049),
at 1.46x the cost per question ($0.80 vs $0.55 per 1k) and p50 4.7 s vs 3.3 s. The
remaining gap is mostly in text-only questions (43% vs 51%) and single spans (58% vs
61%). Each arm has now had one round of dev iteration.

**Arm C vs Arm B v2.** Confident-wrong fell from 30.0% to 24.0% (paired: 15 questions
no longer confidently wrong, 6 newly so; p = 0.078, not significant on 150), at an
abstention rate of 4.0% (6/150, all six would have been wrong) and 1.21x Arm B's cost
($0.97 vs $0.80 per 1k; p95 latency 15.3 s, from retries). 21 questions were retried;
9 of them ended correct. EM is 72.0% vs 70.0% (12 wins, 9 losses, p = 0.66), and
selective accuracy 75.0%. Every answer Arm C gave passed the verifier, yet 36 were
wrong: the verifier catches unsupported numbers and units, not a wrong-but-grounded
answer. Against Arm A v2, Arm C is level on confident-wrong (24.0% vs 22.7%, p = 0.79)
and still behind on EM (6 wins, 14 losses, p = 0.12). Note that B and C are separate
runs, so about 3% of the flips between them are run-to-run noise.
