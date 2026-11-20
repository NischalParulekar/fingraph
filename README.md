# FinGraph

A LangGraph multi-agent analyst that answers questions over financial reports that mix
tables and text (TAT-QA), and an honest experiment on whether graph orchestration beats a
single tool-loop agent that has the same tools and the same model.

**Status:** week 3 of 4. Held-out results are in. Every number in this README comes from a
committed run in `results/` with its config and raw outputs.

## Research question

On hybrid table+text financial QA, does a LangGraph pipeline of specialised nodes beat a
single tool-using agent with the same tools and the same model, in accuracy, cost and
confidently-wrong answers?

## Hypotheses (fixed before any held-out run)

These were committed before any arm was run on the held-out set; `git log -- README.md`
shows that. They are not edited after results exist. A failed hypothesis is reported as
failed.

- **H1 (orchestration).** Arm B (graph, no verifier) has a higher exact match than
  Arm A (single agent) on the held-out set.
- **H2 (verifier).** Arm C (graph + verifier) has a lower confident-wrong rate than
  Arm B, at the cost of a higher abstention rate and more tokens per question.
- **H3 (cost).** Arm B costs more per question than Arm A. The write-up states how much
  more, and whether the accuracy difference (if any) is worth it.

### How they will be tested

- **Data.** 500 held-out questions from the official TAT-QA test split (whose gold
  answers are public), stratified by `answer_from` × `answer_type`, seed 42, IDs
  committed in `splits/heldout.json`. All development uses a separate 150-question subset
  of the official dev split (`splits/dev.json`). Nothing is tuned on the held-out set.
- **Scoring.** The official TAT-QA scorer (vendored unchanged in
  `fingraph/eval/tatqa_official/`): exact match (EM) and numeracy-focused F1, both
  scale-aware. Other metrics are defined in [METRICS.md](METRICS.md).
- **Tests.** Paired, per question, two-sided exact sign test: A vs B on EM (H1); B vs C
  on confident-wrong (H2). Report n, wins, losses, ties and p. A hypothesis is
  supported only if the difference is in the predicted direction with p < 0.05.
  Proportions carry Wilson 95% intervals. H3 is descriptive (cost per 1k questions,
  measured from token usage).
- **Fairness.** All arms use the same model, the same effort setting, the same tool
  implementations (DuckDB table sandbox, retriever, calculator) and the same answer
  format. Only the orchestration and the verifier differ.

## Arms

| Arm | Description |
|---|---|
| A: single agent | One tool-use loop with `query_table` (DuckDB), `retrieve_text`, `calculate`. |
| B: graph | LangGraph pipeline: planner → table / retrieval / calc nodes → synthesizer. |
| C: graph + verifier | Arm B plus an evidence verifier with retry and abstain. |

## Results (held-out, run once)

**Headline: the graph does not beat the single agent; the verifier is what helps.** On
500 held-out questions the LangGraph pipeline (B) is no more accurate than one tool-loop
agent with the same tools and model (71.6% vs 69.2% EM, p = 0.18), and costs 1.48x as
much as billed. Adding the evidence verifier (C) cuts confident-wrong answers from 30.8%
to 26.2% against B (p = 0.0014) for 2.4% abstentions, which brings the graph level with
the single agent on accuracy (71.4% vs 71.6%) but not below it on confident-wrong (26.2%
vs 28.4%, p = 0.24).

All three arms were frozen (tag `arms-frozen-heldout`) before this run, which is the only
one on the held-out set. Produced by `make report` from the committed run files in
`results/heldout/`.

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

Unsupported-number rate (answers with a number found neither in the arm's evidence nor in
a logged calculation; same verifier code for every arm, run offline,
`results/heldout/unsupported_numbers.json`): A 4/500 (0.8%), B 21/500 (4.2%), C 0/488
answered (0.0%; C only answers once the verifier passes, so this is by construction).

Notes on the numbers:
- **Cost depends on caching.** Every arm sends prompts with automatic prompt caching on.
  Arm A's loop re-sends its conversation, so caching saves it money; each graph call is a
  one-off prompt, so B and C pay the cache-write premium and almost never read it back.
  With caching switched off, B would cost 0.94x A, not 1.48x. H3 is judged on the billed
  cost, as METRICS.md defines it.
- **C's billed cost is too low.** B and C ran at the same time and send identical
  planner prompts for the same question, so C read B's cached prompts (3,444 cache-read
  tokens per question). Run alone, C would have been billed $0.93 per 1k questions,
  1.67x A. That is the figure to use for C.
- **Dev numbers were optimistic.** Arm A was 77.3% EM on dev and is 71.6% here; the
  prompts were tuned on dev, as allowed, and the drop is the price of that.
- Run-to-run noise: about 3% of questions change correctness between two identical runs
  (measured on dev; `results/dev/README.md`).

## Deviations from the project spec

Recorded here so nobody has to find them in the diff.

- **No temperature setting.** The spec asks for temperature 0. Claude Haiku 5.5 rejects
  any non-default `temperature` with a 400, so runs are not deterministic. Every arm uses
  the same fixed `effort` instead, and run-to-run variation will be measured with a
  repeat run of Arm A on the dev subset.
- **No Wilson CI code to reuse.** The spec says tableqa has Wilson intervals; it has only
  the sign test. `fingraph/eval/stats.py` adds Wilson intervals and copies the sign
  test.
- **Anthropic SDK instead of langchain-anthropic for model calls.** Every arm calls the
  model through one wrapper (`fingraph/llm.py`) on the Anthropic SDK, so the request
  settings and the cost accounting are identical across arms. LangGraph still
  orchestrates arms B and C; it does not need a LangChain chat model for that.
- **tableqa code is vendored, not imported.** tableqa pins Python 3.11 and this project
  uses 3.12, so the DuckDB sandbox and table code are copied with a note of their source.

## Retrieval: context mode vs corpus mode

The arms retrieve in **context mode**: only the paragraphs of the question's own report.
All headline numbers use it. **Corpus mode** searches all 2,635 paragraphs of the dev and
test reports in pgvector, to show retrieval at a more realistic scale. Measured on the
901 official-dev questions that have gold relevant paragraphs, with the question as the
query (`make retrieval`, `results/retrieval/dev-official.json`; no model calls):

| Mode | hit@1 | hit@3 | recall@3 |
|---|---|---|---|
| Context (in memory, or pgvector filtered to the report) | 64.0% | 91.0% | 90.1% |
| Corpus (pgvector, all reports) | 33.7% | 45.1% | 44.4% |

pgvector in context mode returns the same top-3 ranking as the in-memory retriever on
all 901 questions. Context mode is easy partly because many reports have only 2-3
paragraphs.

## Running it

```
make install    # Python 3.12 + locked dependencies (uv)
make data       # download TAT-QA (pinned commit) into data/ - not redistributed here
make db         # Postgres 16 + pgvector in Docker, and index the paragraphs
make test       # the Postgres tests are skipped if the database is not running
```

Paid runs need `ANTHROPIC_API_KEY` in a git-ignored `.env` (see `.env.example`), and each
one stops at its `--max-cost`.

## Dataset and licence

TAT-QA: Zhu et al., *TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and
Textual Content in Finance*, ACL 2021 ([repo](https://github.com/NExTplusplus/TAT-QA)).
The paper restricts the data to non-commercial research use, so this repo treats it that
way and does not include it: `make data` downloads it. The official evaluation scripts
are MIT-licensed and vendored with their licence.
