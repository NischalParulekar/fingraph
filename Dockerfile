# The FinGraph API. CPU-only PyTorch comes in through sentence-transformers.
FROM python:3.12.8-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv

# Build as the unprivileged user from the start: a `chown -R` afterwards would copy
# every file into a second layer, and uv's download cache is not kept (--no-cache).
RUN useradd --create-home app && mkdir /app && chown app /app
USER app
# PYTHONWARNINGS: the vendored official scorer has invalid escapes in plain strings.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/hf PYTHONWARNINGS=ignore::SyntaxWarning
WORKDIR /app

COPY --chown=app pyproject.toml uv.lock ./
COPY --chown=app fingraph ./fingraph
COPY --chown=app splits ./splits
RUN uv sync --locked --no-dev

# TAT-QA (not redistributed in the repo) and the embedding model are fetched once, at
# build time, both pinned: the dataset by commit and SHA-256, the model by revision.
RUN uv run --no-sync python -m fingraph.data.tatqa download \
 && uv run --no-sync python -c "from fingraph.tools.retriever import bge_embedder; bge_embedder()"

# Hosts such as Render say which port to listen on through PORT.
EXPOSE 8000
CMD ["sh", "-c", "exec uv run --no-sync uvicorn fingraph.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
