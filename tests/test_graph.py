from types import SimpleNamespace

from fingraph.arms.graph import GraphArm, route_plan
from fingraph.data.tatqa import Example, Paragraph
from fingraph.eval.runner import record
from fingraph.llm import LLM
from tests.test_single_agent import EXAMPLE, FakeClient, response, tool_use
from tests.test_tools import GRID, keyword_embedder


def arm(responses):
    client = FakeClient(responses)
    return GraphArm(LLM(client=client), embed=keyword_embedder), client


def plan(cells=(), lookups=(), expression="", scale=""):
    return tool_use(
        "p",
        "submit_plan",
        {
            "table_cells": list(cells),
            "text_lookups": list(lookups),
            "expression": expression,
            "scale": scale,
        },
    )


def test_table_question_runs_planner_table_calc_and_synthesizer():
    graph, client = arm(
        [
            response(
                plan(
                    cells=[
                        {"name": "w2019", "row": 2, "column": 1},
                        {"name": "w2018", "row": 2, "column": 2},
                    ],
                    expression="(w2019 - w2018) / w2018 * 100",
                    scale="percent",
                )
            ),
            response(tool_use("f", "final_answer", {"answer": ["21.55"], "scale": "percent"})),
        ]
    )
    prediction = graph.answer(EXAMPLE)
    assert (prediction.answer, prediction.scale, prediction.error) == (["21.55"], "percent", None)
    assert prediction.llm_calls == 2  # no text lookups, so no reader call

    state = prediction.trace[-1]
    assert [(e["row"], e["column"], e["text"]) for e in state["evidence"]] == [
        (2, 1, "1,320"),
        (2, 2, "1,086"),
    ]
    assert state["calculation"]["result"] == "21.546961326"

    # The planner saw the whole table; the synthesizer saw the evidence and the result.
    assert '"Wireless"' in client.requests[0]["messages"][0]["content"]
    synth_input = client.requests[1]["messages"][0]["content"]
    assert "table row 2, column c1" in synth_input and "= 21.546961326" in synth_input
    assert 'Planned scale: "percent"' in synth_input
    assert record(EXAMPLE, prediction)["em"] == 1.0


TEXT_EXAMPLE = Example(
    uid="q2",
    question="How much debt was repaid?",
    answer_type="span",
    answer_from="text",
    scale="",
    context_uid="c1",
    table=GRID,
    paragraphs=(Paragraph("a", 1, "Revenue grew."), Paragraph("b", 2, "Debt fell by $9,028.")),
    gold={
        "uid": "q2",
        "answer": ["$9,028"],
        "answer_type": "span",
        "answer_from": "text",
        "scale": "",
    },
)


def test_text_question_quotes_are_checked_against_their_paragraph():
    graph, _ = arm(
        [
            response(plan(lookups=[{"name": "repaid", "query": "debt", "what": "the amount"}])),
            response(
                tool_use(
                    "r",
                    "submit_findings",
                    {
                        "findings": [
                            {"name": "repaid", "paragraph": 2, "quote": "$9,028"},
                            {"name": "made_up", "paragraph": 2, "quote": "$1,000"},
                        ]
                    },
                )
            ),
            response(tool_use("f", "final_answer", {"answer": ["$9,028"], "scale": ""})),
        ]
    )
    prediction = graph.answer(TEXT_EXAMPLE)
    evidence = {e["name"]: e for e in prediction.trace[-1]["evidence"]}
    assert evidence["repaid"]["error"] is None and evidence["repaid"]["number"] == 9028.0
    assert evidence["made_up"]["error"] == "quote not found in that paragraph"
    assert prediction.answer == ["$9,028"] and prediction.llm_calls == 3


def test_missing_value_makes_the_calculation_fail_not_guess():
    graph, client = arm(
        [
            response(plan(cells=[{"name": "a", "row": 2, "column": 1}], expression="a - not_read")),
            response(tool_use("f", "final_answer", {"answer": ["1320"], "scale": ""})),
        ]
    )
    prediction = graph.answer(EXAMPLE)
    assert "no value for ['not_read']" in prediction.trace[-1]["calculation"]["error"]
    assert "failed" in client.requests[1]["messages"][0]["content"]


def test_no_plan_is_an_abstention():
    text = SimpleNamespace(type="text", text="I think the answer is 22")
    graph, _ = arm([response(text, stop_reason="end_turn")] * 2)
    prediction = graph.answer(EXAMPLE)
    assert prediction.answer == [] and prediction.error == "planner gave no plan"
    assert prediction.llm_calls == 2  # one reminder, then give up


def test_routing_sends_work_to_the_nodes_the_plan_needs():
    def state(cells, lookups):
        return {"plan": {"table_cells": cells, "text_lookups": lookups, "expression": ""}}

    assert route_plan(state([1], [1])) == ["table_node", "retrieval_node"]
    assert route_plan(state([], [1])) == ["retrieval_node"]
    assert route_plan(state([], [])) == ["calc_node"]
