"""The `retrieve_text` tool: dense retrieval over a report's paragraphs.

Context mode (the default, and what headline numbers use) searches only the paragraphs
of the report the question is about: a TAT-QA context has 2-7 of them. Corpus mode,
which searches every paragraph to show retrieval at a realistic scale, comes later with
pgvector; an exact in-memory search over one context gives the same ranking pgvector
would, so the two arms see identical results either way.

Embeddings: BAAI/bge-small-en-v1.5, normalised, cosine similarity. Queries get bge's
retrieval instruction prefix; passages do not. Two backends, chosen by FINGRAPH_EMBEDDER:

- torch (default; `uv sync --extra torch`): the model through sentence-transformers at a
  pinned revision. Every reported result used this.
- onnx (`--extra onnx`): the quantized ONNX port fastembed uses
  (Qdrant/bge-small-en-v1.5-onnx-Q, pinned revision), run with onnxruntime directly and
  no PyTorch, for the 512 MB demo server. One text at a time, with onnxruntime's memory
  arena off, so memory is bounded by the longest paragraph (fastembed's batched arena
  peaked over 100 MB higher). Its vectors equal fastembed's (cosine 1.0 on 220 texts). On
  the 901 official-dev questions with gold paragraphs it gives the same top-3 order as
  torch on 897 and the same top hit on 900; scores differ by at most 0.0009.
"""

import json
import os
from collections.abc import Callable, Sequence
from functools import cache

import numpy as np

from fingraph.data.tatqa import Paragraph

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"  # pinned HF commit
ONNX_SOURCE = "Qdrant/bge-small-en-v1.5-onnx-Q"  # the ONNX port fastembed uses
ONNX_REVISION = "aa8f8b060edb00e03bfdd08813a2949946c8ba55"  # pinned HF commit
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_K = 3

Embedder = Callable[[Sequence[str]], np.ndarray]  # texts -> (n, d) unit vectors


def backend() -> str:
    name = os.environ.get("FINGRAPH_EMBEDDER", "torch")
    if name not in ("torch", "onnx"):
        raise ValueError(f"FINGRAPH_EMBEDDER must be torch or onnx, not {name!r}")
    return name


@cache
def bge_embedder() -> Embedder:
    if backend() == "onnx":
        return _onnx_embedder()
    from sentence_transformers import SentenceTransformer  # slow import; only when used

    model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device="cpu")

    def embed(texts: Sequence[str]) -> np.ndarray:
        return model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)

    return embed


def _onnx_embedder() -> Embedder:
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    root = snapshot_download(
        ONNX_SOURCE,
        revision=ONNX_REVISION,
        allow_patterns=["model_optimized.onnx", "tokenizer.json"],
    )
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False  # memory stays bounded by one text
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        os.path.join(root, "model_optimized.onnx"), options, providers=["CPUExecutionProvider"]
    )
    tokenizer = Tokenizer.from_file(os.path.join(root, "tokenizer.json"))
    tokenizer.enable_truncation(512)
    inputs = {i.name for i in session.get_inputs()}

    def one(text: str) -> np.ndarray:
        encoding = tokenizer.encode(text)
        ids = np.array([encoding.ids], dtype=np.int64)
        mask = np.array([encoding.attention_mask], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        vector = session.run(None, feeds)[0][0, 0]  # the [CLS] token, as bge pools
        return vector / np.linalg.norm(vector)

    def embed(texts: Sequence[str]) -> np.ndarray:
        return np.array([one(t) for t in texts])

    return embed


class ContextRetriever:
    def __init__(self, paragraphs: Sequence[Paragraph], embed: Embedder | None = None):
        self._paragraphs = list(paragraphs)
        self._embed = embed or bge_embedder()
        self._vectors = self._embed([p.text for p in self._paragraphs])

    def search(self, query: str, k: int = DEFAULT_K) -> list[tuple[Paragraph, float]]:
        """The k best paragraphs for a query, best first, with their cosine scores."""
        if not self._paragraphs:
            return []
        q = self._embed([QUERY_PREFIX + query])[0]
        scores = self._vectors @ q
        best = np.argsort(-scores, kind="stable")[: max(1, k)]
        return [(self._paragraphs[i], float(scores[i])) for i in best]

    def search_text(self, query: str, k: int = DEFAULT_K) -> str:
        """The search results as the model sees them."""
        hits = self.search(query, k)
        return json.dumps(
            [{"paragraph": p.order, "score": round(s, 3), "text": p.text} for p, s in hits],
            ensure_ascii=False,
        )
