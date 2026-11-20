"""Integration tests against the docker-compose Postgres. Skipped when it is not running."""

import uuid

import pytest
from langgraph.types import Command

from fingraph import db
from fingraph.arms.graph import GraphArm
from fingraph.llm import LLM, Spend
from fingraph.tools.registry import ContextTools
from tests.test_arm_c import GOOD_PLAN, answer
from tests.test_single_agent import EXAMPLE, FakeClient
from tests.test_tools import keyword_embedder

pytestmark = pytest.mark.skipif(not db.is_up(), reason="Postgres not running (docker compose up)")


def review_arm(saver, responses=()):
    return GraphArm(
        LLM(client=FakeClient(responses)),
        verifier=True,
        max_retries=0,
        mode="api",
        checkpointer=saver,
        embed=keyword_embedder,
    )


def test_a_paused_review_survives_a_restart_and_resumes_from_postgres():
    thread = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}
    tools = ContextTools(EXAMPLE, keyword_embedder)
    try:
        # First process: runs until human_review and stops there.
        with db.checkpointer() as saver:
            arm = review_arm(saver, [GOOD_PLAN, answer(["99"])])
            context = arm.context(EXAMPLE, tools, Spend())
            first = arm._graph.invoke(
                {"question": EXAMPLE.question, "evidence": []}, thread, context=context
            )
            assert first["__interrupt__"][0].value["answer"] == ["99"]

        # Second process: a new connection and a new graph, with no model responses
        # left, picks the run up from the checkpoint and applies the reviewer's edit.
        with db.checkpointer() as saver:
            arm = review_arm(saver)
            pending = arm._graph.get_state(thread)
            assert pending.next == ("human_review",)
            context = arm.context(EXAMPLE, tools, Spend())
            final = arm._graph.invoke(
                Command(resume={"action": "edit", "answer": ["21.55"], "scale": "percent"}),
                thread,
                context=context,
            )
            assert final["answer"] == ["21.55"] and final["review"] == {"action": "edit"}
            assert arm._graph.get_state(thread).next == ()
    finally:
        tools.close()


def test_pgvector_context_mode_stays_in_the_report_and_corpus_mode_flags_others():
    import json

    from fingraph.data.tatqa import Paragraph
    from fingraph.tools.pgvector_store import PgVectorRetriever, index, store
    from tests.test_graph import TEXT_EXAMPLE

    other = TEXT_EXAMPLE.__class__(
        **{
            **TEXT_EXAMPLE.__dict__,
            "uid": "q3",
            "context_uid": "c2",
            "paragraphs": (Paragraph("z", 1, "Debt debt debt."),),
        }
    )
    vector_store = store(keyword_embedder, collection=f"test_{uuid.uuid4().hex}")
    try:
        assert index([TEXT_EXAMPLE, other, TEXT_EXAMPLE], vector_store) == 3  # once per report
        context = PgVectorRetriever(vector_store, "c1")
        assert [p.order for p, _, c in context.search("debt", k=3)] == [2, 1]
        corpus = json.loads(PgVectorRetriever(vector_store, "c1", "corpus").search_text("debt", 3))
        flagged = [h["text"] for h in corpus if h.get("from_another_report")]
        assert flagged == ["Debt debt debt."] and len(corpus) == 3
    finally:
        vector_store.delete_collection()
