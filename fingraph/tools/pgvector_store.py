"""Paragraph retrieval from pgvector, through LangChain's PGVector store.

    python -m fingraph.tools.pgvector_store index     # embed and store every paragraph

One collection holds the paragraphs of the official dev and test splits (each with its
report's context uid), embedded with the same bge-small model as the in-memory
retriever. Two modes:

- context: search only the question's own report (metadata filter on context_uid).
  An exact search over a handful of paragraphs, so it ranks like the in-memory
  ContextRetriever; it is what the headline numbers use.
- corpus: search every report's paragraphs, to show retrieval at a more realistic
  scale. A hit may come from another company's report, which the result says.

PGVector uses cosine distance; scores here are 1 - distance, i.e. cosine similarity,
as in the in-memory retriever.
"""

import argparse
import json
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from fingraph import db
from fingraph.data.tatqa import Example, Paragraph, load
from fingraph.tools.retriever import (
    DEFAULT_K,
    EMBEDDING_MODEL,
    QUERY_PREFIX,
    Embedder,
    bge_embedder,
)

COLLECTION = "tatqa_paragraphs_bge_small_v1_5"


class BgeEmbeddings(Embeddings):
    """LangChain wrapper around the project's embedder, with bge's query prefix."""

    def __init__(self, embed: Embedder | None = None):
        self._embed = embed or bge_embedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embed([QUERY_PREFIX + text])[0].tolist()


def _sqlalchemy_url(url: str) -> str:
    return url.replace("postgresql://", "postgresql+psycopg://", 1)


def store(embed: Embedder | None = None, *, url: str | None = None, collection: str = COLLECTION):
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=BgeEmbeddings(embed),
        connection=_sqlalchemy_url(url or db.database_url()),
        collection_name=collection,
        collection_metadata={"embedding_model": EMBEDDING_MODEL},
        use_jsonb=True,
    )


def index(examples: Sequence[Example], vector_store) -> int:
    """Store each report's paragraphs once. Idempotent: ids are context_uid:order."""
    docs, ids = [], []
    seen = set()
    for e in examples:
        if e.context_uid in seen:
            continue
        seen.add(e.context_uid)
        for p in e.paragraphs:
            docs.append(
                Document(
                    p.text, metadata={"context_uid": e.context_uid, "order": p.order, "uid": p.uid}
                )
            )
            ids.append(f"{e.context_uid}:{p.order}")
    for start in range(0, len(docs), 256):
        vector_store.add_documents(docs[start : start + 256], ids=ids[start : start + 256])
    return len(docs)


class PgVectorRetriever:
    """Same interface as ContextRetriever, backed by pgvector."""

    def __init__(self, vector_store, context_uid: str, mode: str = "context"):
        if mode not in ("context", "corpus"):
            raise ValueError(mode)
        self._store = vector_store
        self._context_uid = context_uid
        self.mode = mode

    def search(self, query: str, k: int = DEFAULT_K) -> list[tuple[Paragraph, float, str]]:
        """The k best paragraphs, best first: (paragraph, cosine similarity, its report)."""
        where = {"context_uid": self._context_uid} if self.mode == "context" else None
        hits = self._store.similarity_search_with_score(query, k=max(1, k), filter=where)
        return [
            (
                Paragraph(doc.metadata["uid"], int(doc.metadata["order"]), doc.page_content),
                1.0 - distance,
                doc.metadata["context_uid"],
            )
            for doc, distance in hits
        ]

    def search_text(self, query: str, k: int = DEFAULT_K) -> str:
        results = []
        for p, score, context_uid in self.search(query, k):
            hit = {"paragraph": p.order, "score": round(score, 3), "text": p.text}
            if context_uid != self._context_uid:
                hit["from_another_report"] = True
            results.append(hit)
        return json.dumps(results, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["index"])
    parser.parse_args()
    vector_store = store()
    examples = [*load("dev").values(), *load("test").values()]
    print(f"stored {index(examples, vector_store)} paragraphs in {COLLECTION}")


if __name__ == "__main__":
    main()
