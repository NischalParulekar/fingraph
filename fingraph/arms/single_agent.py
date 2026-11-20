"""Arm A: one tool-use loop with the table, retrieval and calculator tools.

The loop follows tableqa's agent: each step is one model call; the model's tool calls
run against this question's context and all their results go back in one message; the
loop ends when the model calls `final_answer` or a step or token budget runs out. Running
out of budget leaves the question unanswered, which counts as an abstention.
"""

import time

from fingraph.arms.base import ANSWER_RULES, FINAL_ANSWER_TOOL, Prediction
from fingraph.data.tatqa import Example
from fingraph.llm import LLM, Spend
from fingraph.tools.registry import TOOL_DEFINITIONS, ContextTools
from fingraph.tools.retriever import DEFAULT_K, EMBEDDING_MODEL, EMBEDDING_REVISION, Embedder

SYSTEM_PROMPT = f"""\
You answer questions about one company's financial report. The report has a table and \
a few paragraphs of text, and the tools are your only access to them:
- query_table runs SQL against the table,
- retrieve_text searches the paragraphs,
- calculate does arithmetic.

How to work
- Start by looking at the table with query_table (it is small, so SELECT * is fine) \
and by searching the paragraphs with retrieve_text. You can call both in the same turn. \
Many answers are only in the text, or need a number from the text, so search it even \
when the table looks relevant.
- Read the header rows: they say which column is which period and what unit the \
amounts are in.
- Do every calculation with calculate.
- Call final_answer when you have the answer. Text you write outside a tool call is \
not read.
- You have at most {{max_steps}} turns, so answer once you are confident.

{ANSWER_RULES}"""

REMINDER = (
    "You have not called final_answer yet. Call it now with your answer; "
    "text outside a tool call is not read."
)


class SingleAgent:
    name = "single_agent"

    def __init__(
        self,
        llm: LLM | None = None,
        *,
        max_steps: int = 10,
        max_question_tokens: int = 80_000,
        embed: Embedder | None = None,
    ):
        self.llm = llm or LLM()
        self.max_steps = max_steps
        self.max_question_tokens = max_question_tokens
        self._embed = embed
        self._system = SYSTEM_PROMPT.format(max_steps=max_steps)
        self._tools = [*TOOL_DEFINITIONS, FINAL_ANSWER_TOOL]

    @property
    def config(self) -> dict:
        return {
            "arm": self.name,
            **self.llm.config,
            "max_steps": self.max_steps,
            "max_question_tokens": self.max_question_tokens,
            "retrieval": {
                "mode": "context",
                "k": DEFAULT_K,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_revision": EMBEDDING_REVISION,
            },
            "system_prompt": self._system,
            "tools": self._tools,
        }

    def answer(self, example: Example) -> Prediction:
        start = time.perf_counter()
        spend = Spend()
        tools = ContextTools(example, self._embed)
        try:
            answer, scale, error = self._loop(example, tools, spend)
        finally:
            tools.close()
        return Prediction(
            answer=answer,
            scale=scale,
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=spend.cost_usd,
            tokens=spend.tokens,
            llm_calls=spend.calls,
            error=error,
            trace=spend.steps,
        )

    def _loop(self, example: Example, tools: ContextTools, spend: Spend):
        content = f"{tools.table.describe()}\n\nQuestion: {example.question}"
        messages = [{"role": "user", "content": content}]
        for _ in range(self.max_steps):
            response = self.llm.create(
                system=self._system,
                messages=messages,
                tools=self._tools,
                spend=spend,
                label="agent",
            )
            record = spend.steps[-1]
            if response.stop_reason in ("refusal", "max_tokens"):
                return [], "", f"model stopped: {response.stop_reason}"

            # The whole content goes back unchanged: it can hold thinking blocks, which
            # the API requires to be returned as they came.
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for call in (b for b in response.content if b.type == "tool_use"):
                if call.name == "final_answer":
                    record["tool_calls"].append({"name": call.name, "input": call.input})
                    return list(call.input["answer"]), call.input["scale"], None
                result = tools.execute(call.name, call.input)
                record["tool_calls"].append(
                    {
                        "name": call.name,
                        "input": call.input,
                        "result": result.text,
                        "error": result.is_error,
                    }
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": result.text,
                        "is_error": result.is_error,
                    }
                )
            messages.append({"role": "user", "content": results or REMINDER})
            if spend.tokens > self.max_question_tokens:
                return [], "", "token budget exhausted"
        return [], "", "step budget exhausted"
