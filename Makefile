# make install   Python 3.12 and the locked dependencies, both embedders (torch, onnx)
# make data      download TAT-QA at a pinned commit into data/ (not redistributed)
# make splits    rebuild the frozen question-ID splits (they are committed; this checks them)
# make db        start Postgres 16 + pgvector (docker compose), then index the paragraphs
# make stack     the whole local stack: API, Postgres, Redis, Prometheus, Grafana
# make retrieval  retrieval hit/recall, context vs corpus mode (no model calls)
# make test, make lint

.PHONY: install data splits db stack aws-up aws-down retrieval test lint eval-dev-a eval-heldout report plots labels demo ui

install:
	uv sync --locked --extra torch --extra onnx

data:
	uv run python -m fingraph.data.tatqa download

splits:
	uv run python -m fingraph.data.splits

db:
	docker compose up -d --wait postgres
	uv run python -m fingraph.tools.pgvector_store index

stack:
	docker compose up -d --build --wait

retrieval:
	uv run python -m fingraph.eval.retrieval --split dev-official

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

# Paid: Arm A on the 150-question dev subset. Needs a key in .env; the cost is not
# measured yet, so the run is capped and stops at --max-cost.
eval-dev-a:
	uv run python -m fingraph.eval.runner --arm single_agent --split dev --max-cost 0.50

# Paid, and done once (2026-11-20): every arm on the 500 held-out questions. Do not rerun
# without a note in the README. Run the arms one after another: run side by side, B and
# C share prompt-cache hits, which lowers C's billed cost.
eval-heldout:
	uv run python -m fingraph.eval.runner --arm single_agent --split heldout --max-cost 0.45
	uv run python -m fingraph.eval.runner --arm graph --split heldout --max-cost 0.50
	uv run python -m fingraph.eval.runner --arm graph_verifier --split heldout --max-cost 0.60

# Free: the README's held-out table and hypothesis tests, from the committed run files.
report:
	uv run python -m fingraph.eval.report results/heldout

# Free: the held-out plots and Arm C's coverage replay, and the error-label tally.
plots:
	uv run python -m fingraph.eval.plots results/heldout

labels:
	uv run python -m fingraph.eval.error_labels

# Paid, on your AWS account: see infra/aws/README.md (about $2 a day while it runs).
# Needs terraform.tfvars and TF_VAR_anthropic_api_key; the budget alarm goes first.
aws-up:
	bash infra/aws/deploy.sh

aws-down:
	cd infra/aws && terraform destroy

# Free: the recorded demo's data, from the committed dev runs (frontend/public/demo.json).
demo:
	uv run python scripts/build_demo.py

# The UI on http://localhost:5173 against the API on :8000 (make stack first).
ui:
	cd frontend && npm install && npm run dev
