# make install   Python 3.12 and the locked dependencies
# make data      download TAT-QA at a pinned commit into data/ (not redistributed)
# make splits    rebuild the frozen question-ID splits (they are committed; this checks them)
# make test, make lint

.PHONY: install data splits test lint eval-dev-a

install:
	uv sync --locked

data:
	uv run python -m fingraph.data.tatqa download

splits:
	uv run python -m fingraph.data.splits

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

# Paid: Arm A on the 150-question dev subset. Needs a key in .env; the cost is not
# measured yet, so the run is capped and stops at --max-cost.
eval-dev-a:
	uv run python -m fingraph.eval.runner --arm single_agent --split dev --max-cost 0.50
