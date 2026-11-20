"""The one way every arm calls the model, so all arms send identical request settings.

Built on the Anthropic SDK directly (as in tableqa) rather than langchain-anthropic:
it keeps the request parameters (effort, caching) and the cost accounting exactly the
same for the single agent and the graph nodes. LangGraph orchestrates the graph arms;
it does not need a LangChain chat model to do so.

Claude Haiku 5.5 rejects any non-default temperature, so there is none here; `effort`
is fixed per run and recorded in the run config instead.

`LLM_PROVIDER` picks where the request goes: `anthropic` (the default, and what every
frozen result used) or `bedrock` (Amazon Bedrock's Converse API, `fingraph/bedrock.py`).
Both return the same response shape, so nothing above this file changes.

With `LANGSMITH_TRACING=true` and a `LANGSMITH_API_KEY`, every call is traced to
LangSmith as an LLM run with its token usage, nested under the graph node (LangGraph
traces itself) or the question that made it. Off by default; it only observes.
"""

import os
from dataclasses import dataclass, field

from langsmith import traceable

MODEL = "claude-haiku-5-5"
EFFORT = "medium"

# USD per million tokens (input, output), for prompts up to 100K tokens.
# Cache reads cost 0.1x the input price and 5-minute cache writes 1.25x.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}
CACHE_READ, CACHE_WRITE = 0.1, 1.25

# Amazon Bedrock on-demand prices, by model or inference-profile ID, USD per million
# tokens (input, output) for prompts up to 100K tokens. Read from
# https://aws.amazon.com/bedrock/pricing/ on 21 November 2026, for US East (Ohio) and
# Asia Pacific (Mumbai), which list the same figures. Cache reads and 5-minute cache
# writes are 0.1x and 1.25x the input price there too ($0.01 and $0.125 for the global
# profile), so CACHE_READ and CACHE_WRITE apply unchanged. Geo and in-region profiles
# cost 10% more than the global one; Mumbai has only the global one for Haiku 5.5.
BEDROCK_PRICES_CHECKED = "2026-11-21"
BEDROCK_PRICES = {
    "global.anthropic.claude-haiku-5-5": (0.10, 0.50),
    "us.anthropic.claude-haiku-5-5": (0.11, 0.55),
    "eu.anthropic.claude-haiku-5-5": (0.11, 0.55),
    "jp.anthropic.claude-haiku-5-5": (0.11, 0.55),
    "au.anthropic.claude-haiku-5-5": (0.11, 0.55),
    "ca.anthropic.claude-haiku-5-5": (0.11, 0.55),
}


def prices(config: dict) -> tuple[float, float]:
    """(input, output) USD per million tokens for a run's LLM config."""
    if config.get("provider") == "bedrock":
        return BEDROCK_PRICES[config["bedrock_model_id"]]
    return PRICES[config["model"]]


@dataclass
class Spend:
    """Tokens, cost and model calls for one question, summed over every call."""

    calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    steps: list[dict] = field(default_factory=list)


def _trace_inputs(inputs: dict) -> dict:
    return {k: v for k, v in inputs.items() if k != "client"}


def _trace_outputs(response) -> dict:
    usage = response.usage
    cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", None) or 0
    total_in = usage.input_tokens + cache_read + cache_write
    return {
        "content": response.content,
        "stop_reason": response.stop_reason,
        "usage_metadata": {
            "input_tokens": total_in,
            "output_tokens": usage.output_tokens,
            "total_tokens": total_in + usage.output_tokens,
            "input_token_details": {"cache_read": cache_read, "cache_creation": cache_write},
        },
    }


@traceable(run_type="llm", process_inputs=_trace_inputs, process_outputs=_trace_outputs)
def _request(client, **request):
    return client.messages.create(**request)


class LLM:
    def __init__(
        self,
        model: str = MODEL,
        effort: str = EFFORT,
        max_output_tokens: int = 4096,
        client=None,
        provider: str | None = None,
    ):
        self.provider = provider or os.environ.get("LLM_PROVIDER") or "anthropic"
        self.bedrock_model_id = None
        if self.provider == "bedrock":
            from fingraph.bedrock import MODEL_IDS, BedrockClient

            self.bedrock_model_id = os.environ.get("BEDROCK_MODEL_ID") or MODEL_IDS[model]
            if client is None:
                client = BedrockClient(self.bedrock_model_id)
        elif self.provider != "anthropic":
            raise ValueError(f"LLM_PROVIDER must be anthropic or bedrock, not {self.provider!r}")
        if client is None:
            import anthropic  # imported here so tests with a fake client need no key

            client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_output_tokens = max_output_tokens
        self._client = client

    @property
    def config(self) -> dict:
        config = {
            "model": self.model,
            "effort": self.effort,
            "max_output_tokens": self.max_output_tokens,
            "provider": self.provider,
        }
        if self.provider == "bedrock":
            config["bedrock_model_id"] = self.bedrock_model_id
            config["aws_region"] = os.environ.get("AWS_REGION") or os.environ.get(
                "AWS_DEFAULT_REGION"
            )
        return config

    def create(self, *, system: str, messages: list, tools: list, spend: Spend, label: str):
        """One model request. Its usage is added to `spend` under `label` (e.g. a node)."""
        response = _request(
            self._client,
            model=self.model,
            max_tokens=self.max_output_tokens,
            system=system,
            tools=tools,
            messages=messages,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            langsmith_extra={
                "name": label,
                "metadata": {"ls_provider": self.provider, "ls_model_name": self.model},
            },
        )
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", None) or 0
        price_in, price_out = prices(self.config)
        cost = (
            (usage.input_tokens + cache_read * CACHE_READ + cache_write * CACHE_WRITE) * price_in
            + usage.output_tokens * price_out
        ) / 1_000_000
        spend.calls += 1
        spend.tokens += usage.input_tokens + usage.output_tokens + cache_read + cache_write
        spend.cost_usd += cost
        step = {
            "label": label,
            "stop_reason": response.stop_reason,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
            "text": [b.text for b in response.content if b.type == "text"],
            "tool_calls": [],
        }
        if self.provider == "bedrock":  # throttling and other retried errors, per call
            step["retries"] = response.retries
        spend.steps.append(step)
        return response
