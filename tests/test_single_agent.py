from types import SimpleNamespace

from fingraph.arms.single_agent import SingleAgent
from fingraph.data.tatqa import Example, Paragraph
from fingraph.eval.runner import record, summarize
from fingraph.llm import LLM
from tests.test_tools import GRID, keyword_embedder

EXAMPLE = Example(
    uid="q1",
    question="What was the percentage change in Wireless between 2018 and 2019?",
    answer_type="arithmetic",
    answer_from="table",
    scale="percent",
    context_uid="c1",
    table=GRID,
    paragraphs=(Paragraph("p1", 1, "Revenue grew."),),
    gold={
        "uid": "q1",
        "answer": 21.55,
        "answer_type": "arithmetic",
        "answer_from": "table",
        "scale": "percent",
    },
)


def tool_use(id, name, input):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def response(*blocks, stop_reason="tool_use"):
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=100,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, usage=usage)


class FakeClient:
    """Replays scripted responses and records every request."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **request):
        # Snapshot: the agent keeps appending to the same messages list after the call.
        self.requests.append({**request, "messages": list(request["messages"])})
        return self._responses.pop(0)


def agent(responses, **options):
    client = FakeClient(responses)
    return SingleAgent(LLM(client=client), embed=keyword_embedder, **options), client


def test_agent_queries_calculates_and_answers():
    arm, client = agent(
        [
            response(tool_use("t1", "query_table", {"sql": "SELECT c1, c2 FROM t WHERE row = 2"})),
            response(tool_use("t2", "calculate", {"expression": "(1320 - 1086) / 1086 * 100"})),
            response(tool_use("t3", "final_answer", {"answer": ["21.55"], "scale": "percent"})),
        ]
    )
    prediction = arm.answer(EXAMPLE)
    assert (prediction.answer, prediction.scale, prediction.error) == (["21.55"], "percent", None)
    assert prediction.llm_calls == 3
    assert prediction.cost_usd == 3 * (1000 * 0.10 + 100 * 0.50) / 1_000_000

    # Tool results went back to the model, and the trace keeps them for the verifier.
    second_request = client.requests[1]["messages"]
    assert '"1,320"' in second_request[-1]["content"][0]["content"]
    assert prediction.trace[1]["tool_calls"][0]["result"] == '{"result": "21.546961326"}'

    # Every request carries the same fixed settings and no temperature.
    for request in client.requests:
        assert request["model"] == "claude-haiku-5-5"
        assert request["output_config"] == {"effort": "medium"}
        assert "temperature" not in request

    scored = record(EXAMPLE, prediction)
    assert scored["em"] == 1.0 and not scored["confident_wrong"]


def test_tool_errors_go_back_to_the_model():
    arm, client = agent(
        [
            response(tool_use("t1", "query_table", {"sql": "DELETE FROM t"})),
            response(tool_use("t2", "final_answer", {"answer": ["22"], "scale": "percent"})),
        ]
    )
    arm.answer(EXAMPLE)
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True


def test_running_out_of_steps_is_an_abstention():
    text = SimpleNamespace(type="text", text="thinking out loud")
    arm, _ = agent([response(text, stop_reason="end_turn")] * 2, max_steps=2)
    prediction = arm.answer(EXAMPLE)
    assert prediction.answer == [] and prediction.error == "step budget exhausted"
    scored = record(EXAMPLE, prediction)
    assert not scored["answered"] and not scored["confident_wrong"] and scored["em"] == 0


def test_summary_metrics():
    records = [
        {"em": 1.0, "f1": 1.0, "answered": True, "confident_wrong": False},
        {"em": 0.0, "f1": 0.5, "answered": True, "confident_wrong": True},
        {"em": 0.0, "f1": 0.0, "answered": False, "confident_wrong": False},
        {"em": 1.0, "f1": 1.0, "answered": True, "confident_wrong": False},
    ]
    for r in records:
        r.update(latency_ms=100.0, cost_usd=0.001, llm_calls=3, error=None)
    s = summarize(records)
    assert s["em"]["value"] == 0.5
    assert s["confident_wrong"]["value"] == 0.25
    assert s["abstention"]["value"] == 0.25
    assert s["selective_accuracy"]["value"] == 2 / 3
    assert s["cost_per_1k_usd"] == 1.0
