"""The `retrieve_text` tool: dense retrieval over a report's paragraphs.

Context mode (the default, and what headline numbers use) searches only the paragraphs
of the report the question is about: a TAT-QA context has 2-7 of them. Corpus mode,
which searches every paragraph to show retrieval at a realistic scale, comes later with
pgvector; an exact in-memory search over one context gives the same ranking pgvector
would, so the two arms see identical results either way.

Embeddings: BAAI/bge-small-en-v1.5 via sentence-transformers, normalised, cosine
similarity. Queries get bge's retrieval instruction prefix; passages do not.
"""

import json
from collections.abc import Callable, Sequence
from functools import cache

import numpy as np

from fingraph.data.tatqa import Paragraph

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"  # pinned HF commit
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_K = 3

Embedder = Callable[[Sequence[str]], np.ndarray]  # texts -> (n, d) unit vectors


@cache
def bge_embedder() -> Embedder:
    from sentence_transformers import SentenceTransformer  # slow import; only when used

    model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device="cpu")

    def embed(texts: Sequence[str]) -> np.ndarray:
        return model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)

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
