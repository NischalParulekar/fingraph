"""Arm B: a LangGraph pipeline of specialised nodes (Arm C adds the verifier later).

    START -> planner -> table_node | retrieval_node (in parallel) -> calc_node
          -> synthesizer -> END

- planner (LLM): sees the question and the whole table (read with the same query_table
  tool Arm A has) and writes a structured plan: which table cells to read, what to look
  up in the text, and the arithmetic expression over those named values.
- table_node: reads the planned cells with query_table. No LLM; every value keeps its
  (row, column) coordinates.
- retrieval_node: searches the paragraphs with retrieve_text for each text lookup, then
  one LLM call (the reader) quotes the requested value or passage. A quote must appear
  verbatim in its paragraph, or it is marked unverified.
- calc_node: substitutes the named values into the expression and evaluates it with the
  calculate tool. No LLM; the model never does arithmetic.
- synthesizer (LLM): writes the final answer and scale from the evidence and the
  calculation, under the same answer rules every arm gets.

The tools, the LLM wrapper and the per-question spend are passed as LangGraph run
context, not graph state, so the state stays plain data a checkpointer can store.
"""

import json
import re
import time
from dataclasses import dataclass
from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, ValidationError

from fingraph.arms.base import ANSWER_RULES, FINAL_ANSWER_TOOL, Prediction
from fingraph.data.tatqa import Example
from fingraph.eval.scorer import SCALES
from fingraph.llm import LLM, Spend
from fingraph.tools.numbers import parse_financial
from fingraph.tools.registry import TOOL_DEFINITIONS, ContextTools
from fingraph.tools.retriever import DEFAULT_K, EMBEDDING_MODEL, EMBEDDING_REVISION, Embedder

# --- structured outputs --------------------------------------------------------------


class TableCell(BaseModel):
    name: str
    row: int
    column: int


class TextLookup(BaseModel):
    name: str
    query: str
    what: str


class Plan(BaseModel):
    table_cells: list[TableCell]
    text_lookups: list[TextLookup]
    expression: str
    scale: str


class Finding(BaseModel):
    name: str
    paragraph: int
    quote: str


class Findings(BaseModel):
    findings: list[Finding]


def _strict_tool(name: str, description: str, schema: dict) -> dict:
    return {"name": name, "description": description, "strict": True, "input_schema": schema}


def _object(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_NAME_FIELD = {"type": "string", "description": "a short snake_case name"}

SUBMIT_PLAN_TOOL = _strict_tool(
    "submit_plan",
    "Submit the plan. This is the only way to answer.",
    _object(
        {
            "table_cells": {
                "type": "array",
                "description": "Table cells to read, each under a name.",
                "items": _object(
                    {
                        "name": _NAME_FIELD,
                        "row": {"type": "integer", "description": "the `row` value"},
                        "column": {"type": "integer", "description": "J for column cJ"},
                    }
                ),
            },
            "text_lookups": {
                "type": "array",
                "description": "Things to find in the report's paragraphs.",
                "items": _object(
                    {
                        "name": _NAME_FIELD,
                        "query": {"type": "string", "description": "search query"},
                        "what": {
                            "type": "string",
                            "description": "exactly what to quote from the text",
                        },
                    }
                ),
            },
            "expression": {
                "type": "string",
                "description": 'Arithmetic over the names, e.g. "(a - b) / b * 100"; '
                '"" if no calculation is needed.',
            },
            "scale": {
                "type": "string",
                "enum": list(SCALES),
                "description": "The unit of the final answer, by the answer rules.",
            },
        }
    ),
)

SUBMIT_FINDINGS_TOOL = _strict_tool(
    "submit_findings",
    "Submit what you found. This is the only way to answer.",
    _object(
        {
            "findings": {
                "type": "array",
                "items": _object(
                    {
                        "name": _NAME_FIELD,
                        "paragraph": {"type": "integer", "description": "paragraph number"},
                        "quote": {
                            "type": "string",
                            "description": "the exact words from that paragraph",
                        },
                    }
                ),
            }
        }
    ),
)

# --- prompts -------------------------------------------------------------------------

PLANNER_PROMPT = """\
You plan how to answer a question about one company's financial report. The report has \
a table, shown to you in full, and a few paragraphs of text, which you cannot see. \
Other steps carry out your plan: they read the table cells you name, search the text \
for what you ask, and do the arithmetic you write. You do not answer the question \
yourself.

Write the plan with submit_plan:
- table_cells: every table cell the answer needs, each with a short snake_case name, \
its `row` value and its column number J (for column cJ). Read the header rows to pick \
the right period and item.
- text_lookups: anything the answer needs from the text (a reason, a description, a \
figure the table does not have, or the answer itself when it is a phrase). Give each a \
name, a search query, and what exactly to quote. Many answers are only in the text, so \
add a lookup unless the table alone clearly answers the question.
- expression: if the answer has to be calculated, an arithmetic expression over the \
names using only + - * / and parentheses, such as "(rev_2019 - rev_2018) / rev_2018 * \
100" for a percentage change; "" if nothing has to be calculated. Every name in it \
must be a table cell or a text lookup whose quote is a number.
- scale: the unit the final answer will be in, by the rules below. You are the only \
step that sees the table's headers, so the later steps rely on you for it.

{answer_rules}"""

READER_PROMPT = """\
You read paragraphs from a company's financial report and quote what you are asked \
for. For each request, give its name, the number of the paragraph you quote from, and \
the exact words from that paragraph: copy them character for character, and quote \
only what was asked (just the number, as written, if a number was asked for, for \
example "$9,028" or "12.5%"). Leave out a request if no paragraph contains it. Submit \
with submit_findings."""

SYNTHESIZER_PROMPT = f"""\
You write the final answer to a question about a company's financial report, using \
only the evidence and the calculation result you are given. Do not calculate anything \
yourself: if the answer is a calculated number, it is the calculation result. Give the \
answer with final_answer.

The planner, which saw the table's headers and units, gives a planned scale. Use it \
unless the evidence plainly says otherwise (for example, the answer is copied from a \
paragraph, which takes "").

{ANSWER_RULES}"""

REMINDER = "Call the tool now; text outside a tool call is not read."

# --- state and context ---------------------------------------------------------------


class State(TypedDict, total=False):
    question: str
    plan: dict
    evidence: Annotated[list[dict], add]  # appended to by the parallel nodes
    calculation: dict
    answer: list[str]
    scale: str
    error: str


@dataclass
class Context:
    llm: LLM
    tools: ContextTools
    spend: Spend


# --- nodes ---------------------------------------------------------------------------


def _ask(runtime: Runtime[Context], label: str, system: str, content: str, tool: dict):
    """One LLM call that must end in `tool`; one reminder if it does not. Input or None."""
    ctx = runtime.context
    messages = [{"role": "user", "content": content}]
    for _ in range(2):
        response = ctx.llm.create(
            system=system, messages=messages, tools=[tool], spend=ctx.spend, label=label
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            return None
        for block in response.content:
            if block.type == "tool_use" and block.name == tool["name"]:
                ctx.spend.steps[-1]["tool_calls"].append({"name": block.name, "input": block.input})
                return block.input
        messages += [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": REMINDER},
        ]
    return None


def planner(state: State, runtime: Runtime[Context]) -> dict:
    tools = runtime.context.tools
    table = tools.execute("query_table", {"sql": "SELECT * FROM t ORDER BY row"})
    content = (
        f"{tools.table.describe()}\n\nThe whole table:\n{table.text}\n\n"
        f"Question: {state['question']}"
    )
    system = PLANNER_PROMPT.format(answer_rules=ANSWER_RULES)
    raw = _ask(runtime, "planner", system, content, SUBMIT_PLAN_TOOL)
    if raw is None:
        return {"error": "planner gave no plan"}
    try:
        plan = Plan.model_validate(raw)
    except ValidationError as e:
        return {"error": f"invalid plan: {e.errors()[0]['msg']}"}
    return {"plan": plan.model_dump()}


def table_node(state: State, runtime: Runtime[Context]) -> dict:
    tools = runtime.context.tools
    evidence = []
    for cell in state["plan"]["table_cells"]:
        row, col = int(cell["row"]), int(cell["column"])
        item = {"name": cell["name"], "source": "table", "row": row, "column": col}
        if not 0 <= col < tools.table.n_cols:
            evidence.append({**item, "text": None, "number": None, "error": "no such column"})
            continue
        result = tools.execute(
            "query_table", {"sql": f"SELECT c{col}, n{col} FROM t WHERE row = {row}"}
        )
        rows = json.loads(result.text)["rows"] if not result.is_error else []
        if not rows:
            evidence.append({**item, "text": None, "number": None, "error": "no such cell"})
            continue
        text, number = rows[0]
        evidence.append({**item, "text": text, "number": number, "error": None})
    return {"evidence": evidence}


def retrieval_node(state: State, runtime: Runtime[Context]) -> dict:
    tools = runtime.context.tools
    lookups = state["plan"]["text_lookups"]
    paragraphs: dict[int, str] = {}
    for lookup in lookups:
        result = tools.execute("retrieve_text", {"query": lookup["query"]})
        for hit in json.loads(result.text):
            paragraphs[hit["paragraph"]] = hit["text"]
    shown = "\n\n".join(f"Paragraph {n}: {text}" for n, text in sorted(paragraphs.items()))
    requests = "\n".join(f"- {lookup['name']}: {lookup['what']}" for lookup in lookups)
    content = f"{shown}\n\nQuote these:\n{requests}"
    raw = _ask(runtime, "reader", READER_PROMPT, content, SUBMIT_FINDINGS_TOOL)
    if raw is None:
        return {"evidence": []}
    try:
        findings = Findings.model_validate(raw).findings
    except ValidationError:
        return {"evidence": []}
    evidence = []
    for f in findings:
        source = paragraphs.get(f.paragraph, "")
        verified = _squash(f.quote) in _squash(source) if f.quote.strip() else False
        evidence.append(
            {
                "name": f.name,
                "source": "text",
                "paragraph": f.paragraph,
                "text": f.quote,
                "number": parse_financial(f.quote),
                "error": None if verified else "quote not found in that paragraph",
            }
        )
    return {"evidence": evidence}


def calc_node(state: State, runtime: Runtime[Context]) -> dict:
    expression = (state.get("plan") or {}).get("expression", "").strip()
    if not expression:
        return {}
    values = {e["name"]: e["number"] for e in state.get("evidence", []) if e["number"] is not None}
    missing = []

    def substitute(match: re.Match) -> str:
        name = match.group(0)
        if name not in values:
            missing.append(name)
            return name
        return f"({values[name]!r})"

    substituted = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", substitute, expression)
    if missing:
        return {"calculation": {"expression": expression, "error": f"no value for {missing}"}}
    result = runtime.context.tools.execute("calculate", {"expression": substituted})
    if result.is_error:
        return {"calculation": {"expression": expression, "error": result.text}}
    return {
        "calculation": {
            "expression": expression,
            "substituted": substituted,
            "result": json.loads(result.text)["result"],
            "error": None,
        }
    }


def synthesizer(state: State, runtime: Runtime[Context]) -> dict:
    lines = []
    for e in state.get("evidence", []):
        where = (
            f"table row {e['row']}, column c{e['column']}"
            if e["source"] == "table"
            else f"paragraph {e['paragraph']}"
        )
        note = f" ({e['error']})" if e["error"] else ""
        lines.append(f"- {e['name']} [{where}]: {json.dumps(e['text'], ensure_ascii=False)}{note}")
    calc = state.get("calculation")
    if not calc:
        calc_text = "No calculation was needed."
    elif calc.get("error"):
        calc_text = f"The calculation {calc['expression']} failed: {calc['error']}"
    else:
        calc_text = f"{calc['expression']} = {calc['result']}"
    content = (
        f"Question: {state['question']}\n\nEvidence:\n"
        + ("\n".join(lines) or "(none)")
        + f"\n\nCalculation: {calc_text}"
        + f"\n\nPlanned scale: {json.dumps(state['plan']['scale'])}"
    )
    raw = _ask(runtime, "synthesizer", SYNTHESIZER_PROMPT, content, FINAL_ANSWER_TOOL)
    if raw is None:
        return {"error": "synthesizer gave no answer"}
    return {"answer": list(raw["answer"]), "scale": raw["scale"]}


def _squash(text: str) -> str:
    return " ".join(text.split()).lower()


def route_plan(state: State) -> list[str]:
    if state.get("error"):
        return [END]
    plan = state["plan"]
    branches = []
    if plan["table_cells"]:
        branches.append("table_node")
    if plan["text_lookups"]:
        branches.append("retrieval_node")
    return branches or ["calc_node"]


def build_graph():
    graph = StateGraph(State, context_schema=Context)
    graph.add_node("planner", planner)
    graph.add_node("table_node", table_node)
    graph.add_node("retrieval_node", retrieval_node)
    graph.add_node("calc_node", calc_node)
    graph.add_node("synthesizer", synthesizer)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner", route_plan, ["table_node", "retrieval_node", "calc_node", END]
    )
    graph.add_edge("table_node", "calc_node")
    graph.add_edge("retrieval_node", "calc_node")
    graph.add_edge("calc_node", "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile()


# --- the arm -------------------------------------------------------------------------


class GraphArm:
    name = "graph"

    def __init__(self, llm: LLM | None = None, *, embed: Embedder | None = None):
        self.llm = llm or LLM()
        self._embed = embed
        self._graph = build_graph()

    @property
    def config(self) -> dict:
        return {
            "arm": self.name,
            **self.llm.config,
            "verifier": False,
            "retrieval": {
                "mode": "context",
                "k": DEFAULT_K,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_revision": EMBEDDING_REVISION,
            },
            "prompts": {
                "planner": PLANNER_PROMPT.format(answer_rules=ANSWER_RULES),
                "reader": READER_PROMPT,
                "synthesizer": SYNTHESIZER_PROMPT,
            },
            "tools": {
                "planner": SUBMIT_PLAN_TOOL,
                "reader": SUBMIT_FINDINGS_TOOL,
                "synthesizer": FINAL_ANSWER_TOOL,
                "shared": TOOL_DEFINITIONS,
            },
        }

    def answer(self, example: Example) -> Prediction:
        start = time.perf_counter()
        spend = Spend()
        tools = ContextTools(example, self._embed)
        try:
            final = self._graph.invoke(
                {"question": example.question, "evidence": []},
                context=Context(llm=self.llm, tools=tools, spend=spend),
            )
        finally:
            tools.close()
        answer = final.get("answer") or []
        state = {k: final.get(k) for k in ("plan", "evidence", "calculation")}
        return Prediction(
            answer=answer,
            scale=final.get("scale", ""),
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=spend.cost_usd,
            tokens=spend.tokens,
            llm_calls=spend.calls,
            error=final.get("error") or (None if answer else "empty answer"),
            trace=[*spend.steps, {"label": "graph_state", **state}],
        )
