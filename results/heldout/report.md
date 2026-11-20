Held-out set: 500 questions from the TAT-QA test split. EM, confident-wrong and abstention with Wilson 95% intervals.

| Arm | EM | F1 | Confident-wrong | Abstain | Selective acc. | Cost / 1k q | Tokens / q | LLM calls / q | p50 / p95 latency |
|---|---|---|---|---|---|---|---|---|---|
| A: single agent | 71.6% (67.5-75.4) | 0.810 | 28.4% (24.6-32.5) | 0.0% (0.0-0.8) | 71.6% | $0.56 | 6,411 | 2.45 | 3.1 / 4.2 s |
| B: graph | 69.2% (65.0-73.1) | 0.784 | 30.8% (26.9-35.0) | 0.0% (0.0-0.8) | 69.2% | $0.82 | 4,794 | 2.28 | 4.5 / 7.5 s |
| C: graph + verifier | 71.4% (67.3-75.2) | 0.794 | 26.2% (22.5-30.2) | 2.4% (1.4-4.1) | 73.2% | $0.53 | 5,357 | 2.54 | 4.6 / 10.8 s |

- **H1 (B beats A on EM): not supported.** Paired over 500 questions: B right and A wrong on 28, A right and B wrong on 40, 432 ties; exact sign test p = 0.182.
- **H2 (C has fewer confident-wrong answers than B): supported.** 36 questions confidently wrong in B but not in C, 13 the other way, 451 ties; p = 0.0014. The cost: abstention 2.4% vs 0.0%, 5,357 vs 4,794 tokens per question.
- **H3 (B costs more than A): supported.** B costs 1.48x A per question ($0.82 vs $0.56 per 1k), C 0.96x as billed (see the cost table below).
- A vs C, not a pre-registered test: EM 36 C-only right vs 37 A-only right (p = 1); confident-wrong 41 fewer in C vs 30 more (p = 0.235).

Cost per 1k questions, three ways (see `cost_views` in fingraph/eval/report.py):

| Arm | Billed | Billed if run alone | With caching off | Cache-read tokens / q |
|---|---|---|---|---|
| A: single agent | $0.555 | $0.555 | $0.767 | 3,165 |
| B: graph | $0.824 | $0.824 | $0.720 | 3 |
| C: graph + verifier | $0.532 | $0.928 | $0.812 | 3,444 |

EM by question type (n per row):

| Group | n | A | B | C |
|---|---|---|---|---|
| answer_from = table | 222 | 81.1% | 79.7% | 79.3% |
| answer_from = table-text | 164 | 81.1% | 78.0% | 80.5% |
| answer_from = text | 114 | 39.5% | 36.0% | 43.0% |
| answer_type = arithmetic | 210 | 83.3% | 83.8% | 83.3% |
| answer_type = count | 12 | 66.7% | 83.3% | 83.3% |
| answer_type = multi-span | 63 | 82.5% | 74.6% | 77.8% |
| answer_type = span | 215 | 57.2% | 52.6% | 57.2% |
