# make install   Python 3.12 and the locked dependencies
# make data      download TAT-QA at a pinned commit into data/ (not redistributed)
# make splits    rebuild the frozen question-ID splits (they are committed; this checks them)
# make test, make lint

.PHONY: install data splits test lint

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
