# FinGraph

A LangGraph multi-agent analyst that answers questions over financial reports that mix
tables and text (TAT-QA), and an honest experiment on whether graph orchestration beats a
single tool-loop agent that has the same tools and the same model.

**Status:** week 1 of 4. No results yet. Every number in this README will come from a
committed run in `results/` with its config and raw outputs; until then it says `TBD`.

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

## Results

| Arm | EM | F1 | Confident-wrong | Abstain | Cost / 1k q |
|---|---|---|---|---|---|
| A | TBD | TBD | TBD | TBD | TBD |
| B | TBD | TBD | TBD | TBD | TBD |
| C | TBD | TBD | TBD | TBD | TBD |

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
