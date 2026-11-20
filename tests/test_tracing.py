"""LangSmith tracing: off by default, and when on, each model call is an LLM run with
its token usage (a mock LangSmith client stands in for the service)."""

from unittest.mock import MagicMock

from langsmith import Client
from langsmith.run_helpers import tracing_context

from fingraph.llm import LLM, Spend
from tests.test_single_agent import FakeClient, response, tool_use


def call(llm):
    llm.create(
        system="s",
        messages=[{"role": "user", "content": "q"}],
        tools=[],
        spend=Spend(),
        label="planner",
    )


def fake_llm():
    return LLM(client=FakeClient([response(tool_use("t", "final_answer", {"answer": ["1"]}))]))


def test_an_llm_call_is_traced_with_its_usage():
    langsmith = MagicMock(spec=Client)
    with tracing_context(enabled=True, client=langsmith, project_name="test"):
        call(fake_llm())

    created = next(c.kwargs for c in langsmith.method_calls if c[0] == "create_run")
    assert created["name"] == "planner" and created["run_type"] == "llm"
    assert "client" not in created["inputs"]
    assert created["extra"]["metadata"]["ls_provider"] == "anthropic"
    updated = next(c.kwargs for c in langsmith.method_calls if c[0] == "update_run")
    assert updated["outputs"]["usage_metadata"]["input_tokens"] == 1000
    assert updated["outputs"]["usage_metadata"]["output_tokens"] == 100


def test_nothing_is_sent_when_tracing_is_off():
    langsmith = MagicMock(spec=Client)
    with tracing_context(enabled=False, client=langsmith):
        call(fake_llm())
    assert langsmith.method_calls == []
