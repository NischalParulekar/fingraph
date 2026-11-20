import json
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from fingraph.data.tatqa import Paragraph
from fingraph.tools.calculator import CalculatorError, calculate, format_number
from fingraph.tools.numbers import parse_financial
from fingraph.tools.retriever import ContextRetriever
from fingraph.tools.table import TableTool

# --- numbers -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, value",
    [
        ("1,320", 1320.0),
        ("(19)", -19.0),
        ("(1,234.5)", -1234.5),
        ("18.6%", 18.6),
        ("$ 2.5", 2.5),
        ("$1,496.5", 1496.5),
        ("-3", -3.0),
        ("2019", 2019.0),
        (".5", 0.5),
    ],
)
def test_parse_financial_reads_report_numbers(text, value):
    assert parse_financial(text) == value


@pytest.mark.parametrize("text", ["", "-", "—", "N/A", "0.1 pts", "Wireless", "1,32", "$"])
def test_parse_financial_rejects_non_numbers(text):
    assert parse_financial(text) is None


# --- calculator ----------------------------------------------------------------------

small_ints = st.integers(min_value=-(10**6), max_value=10**6)


@given(small_ints, small_ints)
def test_calculator_adds_subtracts_and_multiplies_exactly(a, b):
    assert calculate(f"{a} + {b}").result == a + b
    assert calculate(f"{a} - ({b})").result == a - b
    assert calculate(f"{a} * {b}").result == a * b


@given(small_ints, small_ints.filter(lambda b: b != 0))
def test_calculator_divides(a, b):
    assert calculate(f"{a} / {b}").result == pytest.approx(a / b)


def expressions():
    """Random expression trees over small integers, as (text, exact Fraction value)."""
    leaf = st.integers(min_value=-1000, max_value=1000).map(lambda n: (f"({n})", Fraction(n)))

    def extend(children):
        def combine(op, left, right):
            (lt, lv), (rt, rv) = left, right
            if op == "/" and rv == 0:
                return lt, lv
            value = {"+": lv + rv, "-": lv - rv, "*": lv * rv, "/": lv / rv if rv else lv}[op]
            return f"({lt} {op} {rt})", value

        return st.builds(combine, st.sampled_from("+-*/"), children, children)

    return st.recursive(leaf, extend, max_leaves=8)


@given(expressions())
def test_calculator_matches_exact_arithmetic_on_random_expressions(case):
    text, exact = case
    assert calculate(text).result == pytest.approx(float(exact), rel=1e-9, abs=1e-9)


def test_calculator_logs_operands_in_order():
    calc = calculate("(1,320 - 1,086) / 1,086 * 100")
    assert calc.operands == (1320.0, 1086.0, 1086.0, 100.0)
    assert format_number(calc.result) == "21.546961326"


@pytest.mark.parametrize(
    "expression",
    ["__import__('os')", "x + 1", "2 ** 10", "abs(-1)", "1 / 0", "[1, 2]", "1 if 1 else 2", ""],
)
def test_calculator_rejects_anything_but_arithmetic(expression):
    with pytest.raises(CalculatorError):
        calculate(expression)


def test_format_number():
    assert format_number(0.1 + 0.2) == "0.3"
    assert format_number(-0.0) == "0"
    assert format_number(1234.0) == "1234"


# --- table ---------------------------------------------------------------------------

GRID = (
    ("(In millions)", "Years ended December 31", "", ""),
    ("", "2019", "2018", "%Chg"),
    ("Wireless", "1,320", "1,086", "22"),
    ("Cable", "1,153", "1,429", "(19)"),
)


def test_table_keeps_text_numbers_and_cell_coordinates():
    table = TableTool(GRID)
    try:
        result = json.loads(table.query("SELECT row, c0, c1, n1, n3 FROM t WHERE row >= 2").text)
        assert result["rows"] == [
            [2, "Wireless", "1,320", 1320.0, 22.0],
            [3, "Cable", "1,153", 1153.0, -19.0],
        ]
        assert "4 rows and 4 columns" in table.describe()
    finally:
        table.close()


def test_table_is_read_only_and_returns_errors():
    table = TableTool(GRID)
    try:
        assert table.query("DROP TABLE t").is_error
        assert table.query("SELECT nope FROM t").is_error
        assert table.query("SELECT * FROM read_csv('x.csv')").is_error
    finally:
        table.close()


# --- retriever -----------------------------------------------------------------------


def keyword_embedder(texts):
    """A fake embedder: one dimension per keyword, so rankings are predictable."""
    words = ["revenue", "debt", "employees"]
    vectors = np.array([[t.lower().count(w) for w in words] for t in texts], dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def test_retriever_ranks_paragraphs_and_reports_their_numbers():
    paragraphs = [
        Paragraph("a", 1, "Revenue grew."),
        Paragraph("b", 2, "Debt fell."),
        Paragraph("c", 3, "We have 500 employees."),
    ]
    retriever = ContextRetriever(paragraphs, keyword_embedder)
    hits = retriever.search("how much debt", k=2)
    assert hits[0][0].order == 2
    assert len(hits) == 2
    assert json.loads(retriever.search_text("employees", k=1))[0]["paragraph"] == 3
