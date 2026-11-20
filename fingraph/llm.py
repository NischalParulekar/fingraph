"""The one way every arm calls the model, so all arms send identical request settings.

Built on the Anthropic SDK directly (as in tableqa) rather than langchain-anthropic:
it keeps the request parameters (effort, caching) and the cost accounting exactly the
same for the single agent and the graph nodes. LangGraph orchestrates the graph arms;
it does not need a LangChain chat model to do so.

Claude Haiku 5.5 rejects any non-default temperature, so there is none here; `effort`
is fixed per run and recorded in the run config instead.
"""

from dataclasses import dataclass, field

MODEL = "claude-haiku-5-5"
EFFORT = "medium"

# USD per million tokens (input, output), for prompts up to 100K tokens.
# Cache reads cost 0.1x the input price and 5-minute cache writes 1.25x.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}
CACHE_READ, CACHE_WRITE = 0.1, 1.25


@dataclass
class Spend:
    """Tokens, cost and model calls for one question, summed over every call."""

    calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    steps: list[dict] = field(default_factory=list)


class LLM:
    def __init__(
        self,
        model: str = MODEL,
        effort: str = EFFORT,
        max_output_tokens: int = 4096,
        client=None,
    ):
        if client is None:
            import anthropic  # imported here so tests with a fake client need no key

            client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_output_tokens = max_output_tokens
        self._client = client

    @property
    def config(self) -> dict:
        return {
            "model": self.model,
            "effort": self.effort,
            "max_output_tokens": self.max_output_tokens,
        }

    def create(self, *, system: str, messages: list, tools: list, spend: Spend, label: str):
        """One model request. Its usage is added to `spend` under `label` (e.g. a node)."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_output_tokens,
            system=system,
            tools=tools,
            messages=messages,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
        )
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", None) or 0
        price_in, price_out = PRICES[self.model]
        cost = (
            (usage.input_tokens + cache_read * CACHE_READ + cache_write * CACHE_WRITE) * price_in
            + usage.output_tokens * price_out
        ) / 1_000_000
        spend.calls += 1
        spend.tokens += usage.input_tokens + usage.output_tokens + cache_read + cache_write
        spend.cost_usd += cost
        spend.steps.append(
            {
                "label": label,
                "stop_reason": response.stop_reason,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": cache_read,
                "cache_write_tokens": cache_write,
                "text": [b.text for b in response.content if b.type == "text"],
                "tool_calls": [],
            }
        )
        return response
