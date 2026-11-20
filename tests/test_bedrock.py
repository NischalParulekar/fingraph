"""The Bedrock provider against a stubbed boto3 client: no AWS account, no network.

botocore's Stubber checks every request against the real Converse API schema, so a
mapping that Bedrock would reject fails here too.
"""

import boto3
import pytest
from botocore.stub import Stubber

from fingraph.arms.single_agent import SingleAgent
from fingraph.bedrock import BedrockClient, BedrockError
from fingraph.llm import LLM, Spend, prices
from tests.test_single_agent import EXAMPLE
from tests.test_tools import keyword_embedder

PROFILE = "global.anthropic.claude-haiku-5-5"
TOOL = {
    "name": "final_answer",
    "description": "Give the answer.",
    "strict": True,
    "input_schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
}


def boto_client():
    return boto3.client(
        "bedrock-runtime",
        region_name="ap-south-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )


def converse_response(content, stop_reason="tool_use", read=0, write=0):
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop_reason,
        "usage": {
            "inputTokens": 1000,
            "outputTokens": 100,
            "totalTokens": 1100 + read + write,
            "cacheReadInputTokens": read,
            "cacheWriteInputTokens": write,
        },
        "metrics": {"latencyMs": 900},
    }


def bedrock_llm(client):
    return LLM(
        client=BedrockClient(PROFILE, client=client, sleep=lambda s: None), provider="bedrock"
    )


def test_request_mapping_and_usage():
    client = boto_client()
    expected = {
        "modelId": PROFILE,
        "system": [{"text": "Be exact."}],
        "messages": [{"role": "user", "content": [{"text": "What was revenue?"}]}],
        "inferenceConfig": {"maxTokens": 4096},
        "additionalModelRequestFields": {
            "output_config": {"effort": "medium"},
            "cache_control": {"type": "ephemeral"},
        },
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": "final_answer",
                        "description": "Give the answer.",
                        "inputSchema": {"json": TOOL["input_schema"]},
                        "strict": True,
                    }
                }
            ]
        },
    }
    content = [
        {"reasoningContent": {"reasoningText": {"text": "", "signature": "sig"}}},
        {"text": "Here it is."},
        {"toolUse": {"toolUseId": "t1", "name": "final_answer", "input": {"answer": "5"}}},
    ]
    with Stubber(client) as stub:
        stub.add_response("converse", converse_response(content, read=2000, write=500), expected)
        spend = Spend()
        llm = bedrock_llm(client)
        response = llm.create(
            system="Be exact.",
            messages=[{"role": "user", "content": "What was revenue?"}],
            tools=[TOOL],
            spend=spend,
            label="agent",
        )
        stub.assert_no_pending_responses()

    assert [b.type for b in response.content] == ["thinking", "text", "tool_use"]
    call = response.content[2]
    assert (call.id, call.name, call.input) == ("t1", "final_answer", {"answer": "5"})
    assert response.stop_reason == "tool_use"
    step = spend.steps[0]
    assert (step["input_tokens"], step["output_tokens"]) == (1000, 100)
    assert (step["cache_read_tokens"], step["cache_write_tokens"]) == (2000, 500)
    assert step["retries"] == 0 and step["text"] == ["Here it is."]
    # Same arithmetic as the direct API: the global profile has the same prices.
    expected_cost = (1000 + 2000 * 0.1 + 500 * 1.25) * 0.10 / 1e6 + 100 * 0.50 / 1e6
    assert spend.cost_usd == pytest.approx(expected_cost)
    assert llm.config["provider"] == "bedrock" and llm.config["bedrock_model_id"] == PROFILE


def test_assistant_turn_and_tool_results_go_back_unchanged():
    """A whole tool loop: thinking blocks return with their signature, results map to
    toolResult with an error status where the tool failed."""
    client = boto_client()
    first = [
        {"reasoningContent": {"reasoningText": {"text": "", "signature": "sig-1"}}},
        {"toolUse": {"toolUseId": "t1", "name": "calculate", "input": {"expression": "1/0"}}},
    ]
    final = [
        {
            "toolUse": {
                "toolUseId": "t2",
                "name": "final_answer",
                "input": {"answer": ["21.55"], "scale": "percent"},
            }
        }
    ]
    with Stubber(client) as stub:
        stub.add_response("converse", converse_response(first))
        stub.add_response("converse", converse_response(final))
        captured = []
        client.meta.events.register(
            "provide-client-params.bedrock-runtime.Converse",
            lambda params, **kw: captured.append(params),
        )
        agent = SingleAgent(bedrock_llm(client), embed=keyword_embedder)
        prediction = agent.answer(EXAMPLE)

    assert prediction.answer == ["21.55"] and prediction.scale == "percent"
    second = captured[1]["messages"]
    assert second[1] == {"role": "assistant", "content": first}
    result = second[2]["content"][0]["toolResult"]
    assert result["toolUseId"] == "t1" and result["status"] == "error"


def test_throttling_is_retried_and_counted():
    client = boto_client()
    content = [{"text": "ok"}]
    with Stubber(client) as stub:
        stub.add_client_error("converse", "ThrottlingException", "Too many requests", 429)
        stub.add_client_error("converse", "ThrottlingException", "Too many requests", 429)
        stub.add_response("converse", converse_response(content, stop_reason="end_turn"))
        spend = Spend()
        bedrock_llm(client).create(
            system="s",
            messages=[{"role": "user", "content": "q"}],
            tools=[TOOL],
            spend=spend,
            label="x",
        )
    assert spend.steps[0]["retries"] == 2
    assert spend.steps[0]["stop_reason"] == "end_turn"


def test_throttling_gives_up_after_the_last_attempt():
    client = boto_client()
    with Stubber(client) as stub:
        for _ in range(6):
            stub.add_client_error("converse", "ThrottlingException", "Too many requests", 429)
        with pytest.raises(BedrockError, match="still failing after 6 attempts"):
            bedrock_llm(client).create(
                system="s",
                messages=[{"role": "user", "content": "q"}],
                tools=[],
                spend=Spend(),
                label="x",
            )


@pytest.mark.parametrize(
    "code, message, hint",
    [
        ("AccessDeniedException", "You don't have access to the model", "bedrock:InvokeModel"),
        ("ValidationException", "The provided model identifier is invalid.", "AWS_REGION"),
    ],
)
def test_access_and_model_errors_are_not_retried(code, message, hint):
    client = boto_client()
    with Stubber(client) as stub:
        stub.add_client_error("converse", code, message, 400)
        with pytest.raises(BedrockError, match=hint):
            bedrock_llm(client).create(
                system="s",
                messages=[{"role": "user", "content": "q"}],
                tools=[],
                spend=Spend(),
                label="x",
            )
        stub.assert_no_pending_responses()


def test_refusal_maps_to_the_anthropic_stop_reason():
    client = boto_client()
    with Stubber(client) as stub:
        stub.add_response(
            "converse", converse_response([{"text": ""}], stop_reason="content_filtered")
        )
        response = bedrock_llm(client).create(
            system="s",
            messages=[{"role": "user", "content": "q"}],
            tools=[],
            spend=Spend(),
            label="x",
        )
    assert response.stop_reason == "refusal"


def test_default_provider_is_unchanged(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    llm = LLM(client=object())
    assert llm.provider == "anthropic"
    assert prices(llm.config) == (0.10, 0.50)


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="anthropic or bedrock"):
        LLM(client=object(), provider="vertex")
