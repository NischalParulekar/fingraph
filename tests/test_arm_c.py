from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from fingraph.arms.graph import GraphArm
from fingraph.eval.runner import record
from fingraph.llm import LLM, Spend
from fingraph.tools.registry import ContextTools
from tests.test_graph import plan
from tests.test_single_agent import EXAMPLE, FakeClient, response, tool_use
from tests.test_tools import keyword_embedder

CELLS = [{"name": "w2019", "row": 2, "column": 1}, {"name": "w2018", "row": 2, "column": 2}]
GOOD_PLAN = response(plan(CELLS, expression="(w2019 - w2018) / w2018 * 100", scale="percent"))


def answer(values, scale="percent"):
    return response(tool_use("f", "final_answer", {"answer": values, "scale": scale}))


def arm_c(responses, **options):
    client = FakeClient(responses)
    arm = GraphArm(LLM(client=client), verifier=True, embed=keyword_embedder, **options)
    return arm, client


def test_supported_answer_passes_without_a_retry():
    arm, _ = arm_c([GOOD_PLAN, answer(["21.55"])])
    prediction = arm.answer(EXAMPLE)
    assert prediction.answer == ["21.55"] and prediction.error is None
    state = prediction.trace[-1]
    assert [r["passed"] for r in state["verifier_reports"]] == [True]
    assert state["attempt"] is None  # never retried
    assert record(EXAMPLE, prediction)["em"] == 1.0


def test_unsupported_answer_is_retried_with_the_verifiers_feedback():
    arm, client = arm_c([GOOD_PLAN, answer(["22.5"]), GOOD_PLAN, answer(["21.55"])])
    prediction = arm.answer(EXAMPLE)
    assert prediction.answer == ["21.55"] and prediction.llm_calls == 4
    reports = prediction.trace[-1]["verifier_reports"]
    assert [r["passed"] for r in reports] == [False, True]
    retry_input = client.requests[2]["messages"][0]["content"]
    assert "failed the evidence check" in retry_input and "'22.5'" in retry_input
    # The retry's evidence is its own: each attempt is tagged.
    assert {e["attempt"] for e in prediction.trace[-1]["evidence"]} == {0, 1}


def test_eval_mode_abstains_once_retries_are_used_up():
    arm, _ = arm_c([GOOD_PLAN, answer(["99"])] * 3)
    prediction = arm.answer(EXAMPLE)
    assert prediction.answer == [] and prediction.error.startswith("abstained")
    assert len(prediction.trace[-1]["verifier_reports"]) == 3  # first try + 2 retries
    scored = record(EXAMPLE, prediction)
    assert not scored["answered"] and not scored["confident_wrong"]


def run_until_review(decision):
    """API mode: the graph stops at human_review, then a reviewer's decision resumes it."""
    client = FakeClient([GOOD_PLAN, answer(["99"])])
    arm = GraphArm(
        LLM(client=client),
        verifier=True,
        max_retries=0,
        mode="api",
        checkpointer=InMemorySaver(),
        embed=keyword_embedder,
    )
    tools = ContextTools(EXAMPLE, keyword_embedder)
    context = arm.context(EXAMPLE, tools, Spend())
    config = {"configurable": {"thread_id": "t1"}}
    first = arm._graph.invoke(
        {"question": EXAMPLE.question, "evidence": []}, config, context=context
    )
    pending = first["__interrupt__"][0].value
    assert pending["answer"] == ["99"] and not pending["verifier_report"]["passed"]
    assert pending["actions"] == ["approve", "edit", "reject"]
    final = arm._graph.invoke(Command(resume=decision), config, context=context)
    tools.close()
    return final


def test_reviewer_can_approve_edit_or_reject():
    assert run_until_review({"action": "approve"})["answer"] == ["99"]
    edited = run_until_review({"action": "edit", "answer": ["21.55"], "scale": "percent"})
    assert edited["answer"] == ["21.55"] and edited["review"] == {"action": "edit"}
    assert run_until_review({"action": "reject"})["answer"] == []
