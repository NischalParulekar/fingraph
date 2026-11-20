# Development runs (150-question dev subset)

Dev is used freely for building, so these numbers are optimistic: the prompts were
changed after looking at these very questions. Only held-out runs go in the main README.

| Run | Change | EM | F1 | Confident-wrong | Cost / 1k q |
|---|---|---|---|---|---|
| `single_agent` (v1) | first version | 103/150 = 68.7% | 0.773 | 31.3% | $0.45 |
| `single_agent-v2` | answer rules v2 (scale, span copying, sign), always search the text | 116/150 = 77.3% | 0.851 | 22.7% | $0.55 |

v1 → v2, paired on EM: 17 wins, 4 losses, 129 ties, sign test p = 0.007. Caveat: Haiku
5.5 takes no temperature, so some of those 21 flips are run-to-run noise. One of the
four losses was answered wrongly in the 10-question smoke run, right in v1, and wrong
again in v2, with the same prompt as v1 for the smoke run. A repeat run is needed to
size that noise.

The v2 rules are shared by every arm (`fingraph/arms/base.py`), so they do not favour
Arm A. The conventions behind them were measured on the whole official dev split, never
on held-out.
