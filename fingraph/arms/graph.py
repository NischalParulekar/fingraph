"""Arms B and C: a LangGraph pipeline of specialised nodes, with or without a verifier.

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

Arm C adds, after the synthesizer:

    -> verifier -> END            if the answer passes the evidence check
                -> retry -> planner   (with the verifier's feedback), at most max_retries
                -> human_review -> END    once retries are used up

- verifier: the standalone fingraph.verifier, on this attempt's evidence and
  calculation (unverified quotes support nothing).
- human_review: in eval mode it abstains, so runs stay automatic; in API mode it calls
  LangGraph interrupt() and waits for a reviewer to approve, edit or reject, resumed with
  Command(resume=...). That needs a checkpointer.

Arm B is the same graph with the synthesizer going straight to END; with no retries,
every evidence item is attempt 0 and the Arm C state keys stay unset, so Arm B behaves
exactly as it did when frozen.

The tools, the LLM wrapper and the per-question spend are passed as LangGraph run
context, not graph state, so the state stays plain data a checkpointer can store.
"""

import json
import re
import time
from dataclasses import asdict, dataclass, field
from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import BaseModel, ValidationError

from fingraph.arms.base import ANSWER_RULES, FINAL_ANSWER_TOOL, Prediction
from fingraph.data.tatqa import Example
from fingraph.eval.scorer import SCALES
from fingraph.llm import LLM, Spend
from fingraph.tools.numbers import parse_financial
from fingraph.tools.registry import TOOL_DEFINITIONS, ContextTools
from fingraph.tools.retriever import DEFAULT_K, EMBEDDING_MODEL, EMBEDDING_REVISION, Embedder
from fingraph.verifier import Calculation, Evidence, unit_hints, verify

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
a table, shown to you in full, one line per row, and a few paragraphs of text, of which \
you are shown the ones that best match the question. Other steps carry out your plan: \
they read the table cells you name, quote the text you ask for, and do the arithmetic \
you write. You do not answer the question yourself.

Write the plan with submit_plan:
- table_cells: every table cell the answer needs, each with a short snake_case name, \
its row number (the number after "row") and its column number J (for "cJ"). Read the \
header rows to pick the right period and item, and check that the cell you name is \
not empty.
- text_lookups: anything the answer needs from the paragraphs: a reason, a \
description, a figure the table does not have, or the answer itself when it is a \
phrase. Give each a name, a search query, and what exactly to quote. If a paragraph \
answers the question, take the answer from it rather than from the table, even when \
the table has a similar figure: the question asks about what the paragraph says.
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
    calculation: dict | None
    answer: list[str]
    scale: str
    error: str
    # Arm C only
    attempt: int  # 0 for the first plan, +1 per retry; evidence is tagged with it
    feedback: str  # why the verifier rejected the last answer, for the planner
    verifier_reports: Annotated[list[dict], add]  # one per verified answer
    review: dict  # the human_review outcome


@dataclass
class Context:
    llm: LLM
    tools: ContextTools
    spend: Spend
    unit_hints: list[str] = field(default_factory=list)
    mode: str = "eval"  # Arm C's human_review: "eval" abstains, "api" interrupts


def _current(state: State) -> list[dict]:
    """The evidence gathered for the current attempt (all of it, for Arm B)."""
    attempt = state.get("attempt", 0)
    return [e for e in state.get("evidence", []) if e.get("attempt", 0) == attempt]


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
    # The planner gets what Arm A looks at on its first turn: the whole table and the
    # paragraphs that best match the question, both read with the shared tools.
    text_columns = ", ".join(f"c{j}" for j in range(tools.table.n_cols))
    table = tools.execute("query_table", {"sql": f"SELECT row, {text_columns} FROM t ORDER BY row"})
    paragraphs = tools.execute("retrieve_text", {"query": state["question"]})
    content = (
        f"The table:\n{_render_rows(table.text)}\n\n"
        f"The paragraphs that best match the question:\n{_render_paragraphs(paragraphs.text)}"
        f"\n\nQuestion: {state['question']}"
    )
    if state.get("feedback"):  # Arm C retry
        content += (
            f"\n\nA previous plan led to the answer {json.dumps(state.get('answer'))} with "
            f"scale {json.dumps(state.get('scale'))}, which failed the evidence check:\n"
            f"{state['feedback']}\nPlan again so that every part of the answer can be found "
            "in the table, the text or a calculation, with the right scale."
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
        item = {
            "name": cell["name"],
            "source": "table",
            "row": row,
            "column": col,
            "attempt": state.get("attempt", 0),
        }
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
                "attempt": state.get("attempt", 0),
            }
        )
    return {"evidence": evidence}


def calc_node(state: State, runtime: Runtime[Context]) -> dict:
    expression = (state.get("plan") or {}).get("expression", "").strip()
    if not expression:
        return {"calculation": None}  # also clears a previous attempt's calculation
    values = {e["name"]: e["number"] for e in _current(state) if e["number"] is not None}
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
    for e in _current(state):
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
        return {"error": "synthesizer gave no answer", "answer": [], "scale": ""}
    return {"answer": list(raw["answer"]), "scale": raw["scale"]}


# --- Arm C: verifier, retry, human review --------------------------------------------


def verifier_node(state: State, runtime: Runtime[Context]) -> dict:
    """Check the draft against this attempt's evidence with the standalone verifier."""
    evidence = []
    for e in _current(state):
        if e.get("text") and not e.get("error"):  # unverified quotes support nothing
            where = (
                f"table row {e['row']}, column c{e['column']}"
                if e["source"] == "table"
                else f"paragraph {e['paragraph']}"
            )
            evidence.append(Evidence(str(e["text"]), where))
    calc = state.get("calculation") or {}
    calculations = (
        [Calculation(calc["expression"], float(calc["result"]))] if calc.get("result") else []
    )
    report = verify(
        state.get("answer") or [],
        state.get("scale", ""),
        evidence,
        calculations,
        runtime.context.unit_hints,
    )
    record = {
        "attempt": state.get("attempt", 0),
        "answer": state.get("answer") or [],
        "scale": state.get("scale", ""),
        "passed": report.passed,
        "supported": [asdict(c) for c in report.supported],
        "unsupported": [asdict(c) for c in report.unsupported],
        "scale_problems": report.scale_problems,
    }
    update: dict = {"verifier_reports": [record]}
    if not report.passed:
        update["feedback"] = report.feedback() or "- no answer was given"
    return update


def retry(state: State) -> dict:
    return {"attempt": state.get("attempt", 0) + 1, "error": ""}


def human_review(state: State, runtime: Runtime[Context]) -> dict:
    """Retries are used up. Eval mode abstains; API mode asks a reviewer."""
    if runtime.context.mode == "eval":
        return {"answer": [], "scale": "", "review": {"action": "abstain", "by": "eval mode"}}
    decision = interrupt(
        {
            "question": state["question"],
            "answer": state.get("answer") or [],
            "scale": state.get("scale", ""),
            "verifier_report": state["verifier_reports"][-1],
            "evidence": _current(state),
            "calculation": state.get("calculation"),
            "actions": ["approve", "edit", "reject"],
        }
    )
    action = decision.get("action")
    if action == "approve":
        return {"review": {"action": "approve"}}
    if action == "edit":
        return {
            "answer": list(decision["answer"]),
            "scale": decision.get("scale", state.get("scale", "")),
            "review": {"action": "edit"},
        }
    return {"answer": [], "scale": "", "review": {"action": "reject"}}


def _render_rows(query_result: str) -> str:
    """A query_table result as one labelled line per row: row 3 | c0: "Cable" | c1: ..."""
    result = json.loads(query_result)
    lines = []
    for row in result["rows"]:
        cells = " | ".join(
            f"{column}: {json.dumps(value or '', ensure_ascii=False)}"
            for column, value in zip(result["columns"][1:], row[1:], strict=True)
        )
        lines.append(f"row {row[0]} | {cells}")
    return "\n".join(lines)


def _render_paragraphs(retrieve_result: str) -> str:
    return "\n\n".join(
        f"Paragraph {h['paragraph']}: {h['text']}" for h in json.loads(retrieve_result)
    )


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


def build_graph(verifier: bool = False, max_retries: int = 2, checkpointer=None):
    """Arm B (verifier=False) or Arm C: synthesizer -> verifier -> END, retry or review."""
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
    if not verifier:
        graph.add_edge("synthesizer", END)
        return graph.compile(checkpointer=checkpointer)

    def route_verdict(state: State) -> str:
        if state["verifier_reports"][-1]["passed"]:
            return END
        return "retry" if state.get("attempt", 0) < max_retries else "human_review"

    graph.add_node("verifier", verifier_node)
    graph.add_node("retry", retry)
    graph.add_node("human_review", human_review)
    graph.add_edge("synthesizer", "verifier")
    graph.add_conditional_edges("verifier", route_verdict, [END, "retry", "human_review"])
    graph.add_edge("retry", "planner")
    graph.add_edge("human_review", END)
    return graph.compile(checkpointer=checkpointer)


# --- the arm -------------------------------------------------------------------------


class GraphArm:
    """Arm B (verifier=False) or Arm C (verifier=True)."""

    def __init__(
        self,
        llm: LLM | None = None,
        *,
        verifier: bool = False,
        max_retries: int = 2,
        mode: str = "eval",
        checkpointer=None,
        embed: Embedder | None = None,
    ):
        self.name = "graph_verifier" if verifier else "graph"
        self.llm = llm or LLM()
        self.verifier = verifier
        self.max_retries = max_retries
        self.mode = mode
        self._embed = embed
        self._graph = build_graph(verifier, max_retries, checkpointer)

    @property
    def config(self) -> dict:
        config = {
            "arm": self.name,
            **self.llm.config,
            "verifier": self.verifier,
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
        if self.verifier:
            config.update(max_retries=self.max_retries, mode=self.mode, strict_verifier=False)
        return config

    def context(self, example: Example, tools: ContextTools, spend: Spend) -> Context:
        cells = [cell for row in example.table for cell in row]
        hints = unit_hints(cells, [p.text for p in example.paragraphs])
        return Context(self.llm, tools, spend, unit_hints=hints, mode=self.mode)

    def answer(self, example: Example) -> Prediction:
        start = time.perf_counter()
        spend = Spend()
        tools = ContextTools(example, self._embed)
        try:
            final = self._graph.invoke(
                {"question": example.question, "evidence": []},
                context=self.context(example, tools, spend),
            )
        finally:
            tools.close()
        return self.prediction(final, spend, time.perf_counter() - start)

    def prediction(self, final: dict, spend: Spend, seconds: float) -> Prediction:
        # A planner or synthesizer failure, or an abstention after review, is no answer.
        answer = [] if final.get("error") else (final.get("answer") or [])
        review = final.get("review") or {}
        if final.get("error"):
            error = final["error"]
        elif review.get("action") in ("abstain", "reject"):
            error = f"abstained: failed the evidence check ({review['action']})"
        else:
            error = None if answer else "empty answer"
        keys = ["plan", "evidence", "calculation"]
        if self.verifier:
            keys += ["attempt", "verifier_reports", "review"]
        return Prediction(
            answer=answer,
            scale=final.get("scale", "") if answer else "",
            latency_ms=seconds * 1000,
            cost_usd=spend.cost_usd,
            tokens=spend.tokens,
            llm_calls=spend.calls,
            error=error,
            trace=[*spend.steps, {"label": "graph_state", **{k: final.get(k) for k in keys}}],
        )
