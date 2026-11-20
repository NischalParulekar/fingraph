# The FinGraph API, slim: the embedder is an ONNX port of bge-small run with
# onnxruntime (the onnx extra), with no PyTorch, so the service fits a 512 MB instance.
# The evaluated runs used the PyTorch embedder; fingraph/tools/retriever.py says how
# closely the two agree.
FROM python:3.12.8-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv

# Build as the unprivileged user from the start: a `chown -R` afterwards would copy
# every file into a second layer, and uv's download cache is not kept (UV_NO_CACHE).
RUN useradd --create-home app && mkdir /app && chown app /app
USER app
# PYTHONWARNINGS: the vendored official scorer has invalid escapes in plain strings.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::SyntaxWarning \
    FINGRAPH_EMBEDDER=onnx HF_HOME=/app/.cache/hf
WORKDIR /app

COPY --chown=app pyproject.toml uv.lock ./
COPY --chown=app fingraph ./fingraph
COPY --chown=app splits ./splits
RUN uv sync --locked --no-dev --extra onnx

# TAT-QA (not redistributed in the repo) and the embedding model are fetched once, at
# build time: the dataset pinned by commit and SHA-256. After that the image never
# contacts the Hugging Face Hub.
RUN uv run --no-sync python -m fingraph.data.tatqa download \
 && uv run --no-sync python -c "from fingraph.tools.retriever import bge_embedder; bge_embedder()"
ENV HF_HUB_OFFLINE=1

# Hosts such as Render say which port to listen on through PORT.
EXPOSE 8000
CMD ["sh", "-c", "exec uv run --no-sync uvicorn fingraph.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
