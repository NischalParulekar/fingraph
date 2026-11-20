"""The three tools every arm gets, defined once and bound to one question's context.

Arm A exposes them to the model as tool calls; the graph nodes call the same methods.
Both go through ContextTools, so neither arm has a better tool than the other.
"""

import json

from fingraph.data.tatqa import Example
from fingraph.tools.calculator import CalculatorError, calculate, format_number
from fingraph.tools.retriever import DEFAULT_K, ContextRetriever, Embedder
from fingraph.tools.table import TableTool, ToolResult


def _tool(name: str, description: str, properties: dict) -> dict:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


TOOL_DEFINITIONS = [
    _tool(
        "query_table",
        "Run one read-only DuckDB SELECT statement against the report's table `t` and "
        "return the result. If the query fails, the error comes back so you can fix it.",
        {"sql": {"type": "string", "description": "A single SELECT statement."}},
    ),
    _tool(
        "retrieve_text",
        f"Search the report's paragraphs and return the {DEFAULT_K} most relevant ones, "
        "best first, each with its paragraph number.",
        {"query": {"type": "string", "description": "What to look for, in plain words."}},
    ),
    _tool(
        "calculate",
        "Evaluate an arithmetic expression exactly: numbers, + - * / and parentheses only. "
        'Example: "(1320 - 1086) / 1086 * 100".',
        {"expression": {"type": "string", "description": "The expression to evaluate."}},
    ),
]


class ContextTools:
    """The tools for one question: its table, its paragraphs and a calculator."""

    def __init__(self, example: Example, embed: Embedder | None = None):
        self.table = TableTool(example.table)
        self.retriever = ContextRetriever(example.paragraphs, embed)
        self.calculations: list[dict] = []  # every successful calculate call, in order

    def execute(self, name: str, arguments: dict) -> ToolResult:
        if name == "query_table":
            return self.table.query(arguments["sql"])
        if name == "retrieve_text":
            return ToolResult(self.retriever.search_text(arguments["query"]), False)
        if name == "calculate":
            return self._calculate(arguments["expression"])
        return ToolResult(f"Unknown tool: {name}", True)

    def _calculate(self, expression: str) -> ToolResult:
        try:
            calc = calculate(expression)
        except CalculatorError as e:
            return ToolResult(f"CalculatorError: {e}", True)
        record = {
            "expression": calc.expression,
            "operands": list(calc.operands),
            "result": calc.result,
        }
        self.calculations.append(record)
        return ToolResult(json.dumps({"result": format_number(calc.result)}), False)

    def close(self) -> None:
        self.table.close()
