"""The `calculate` tool: deterministic arithmetic, so the model never does maths itself.

An expression is parsed with Python's `ast` and only numbers, + - * /, unary minus and
parentheses are evaluated; anything else (names, calls, attributes, **, ...) is rejected.
Thousands separators are removed first, so "1,320 - 1,086" works.
Every call is returned as a Calculation (expression, operands, result) so the verifier
can check that a number in an answer came from a logged operation on evidence values.
"""

import ast
import math
import operator
from dataclasses import dataclass

MAX_EXPRESSION_CHARS = 500

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}


class CalculatorError(ValueError):
    """The expression is not allowed or cannot be evaluated."""


@dataclass(frozen=True)
class Calculation:
    expression: str
    operands: tuple[float, ...]  # the numbers written in the expression, in order
    result: float


def calculate(expression: str) -> Calculation:
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise CalculatorError("expression too long")
    try:
        tree = ast.parse(expression.replace(",", ""), mode="eval")
    except SyntaxError as e:
        raise CalculatorError(f"not a valid expression: {e.msg}") from None
    operands: list[float] = []
    result = _eval(tree.body, operands)
    if not math.isfinite(result):
        raise CalculatorError("result is not a finite number")
    return Calculation(expression, tuple(operands), result)


def _eval(node: ast.AST, operands: list[float]) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        operands.append(float(node.value))
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left, operands), _eval(node.right, operands)
        if isinstance(node.op, ast.Div) and right == 0:
            raise CalculatorError("division by zero")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand, operands))
    raise CalculatorError(f"not allowed: {ast.unparse(node)!r} (only numbers, + - * / and ())")


def format_number(value: float) -> str:
    """A result as the model sees it: up to 10 significant decimals, no float noise."""
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text
