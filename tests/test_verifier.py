import ast
import pathlib

from hypothesis import given
from hypothesis import strategies as st

from fingraph.verifier import Calculation, Evidence, verify
from fingraph.verifier.core import numbers_in

CELL = Evidence("1,320", "table row 2, column c1")
OLD = Evidence("1,086", "table row 2, column c2")
PARA = Evidence(
    "Depreciation expense was $12,548 and $12,643, of which $9,028 related to software.",
    "paragraph 4",
)
CHANGE = Calculation("(1320 - 1086) / 1086 * 100", 21.546961326)


def test_the_verifier_does_not_import_langgraph():
    package = pathlib.Path(__file__).resolve().parents[1] / "fingraph" / "verifier"
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        names += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
        assert not [n for n in names if n.startswith(("langgraph", "langchain"))], path


def test_calculated_number_is_supported_by_its_logged_calculation():
    report = verify(["21.55"], "percent", [CELL, OLD], [CHANGE])
    assert report.passed
    assert report.supported[0].support.startswith("calculation")


def test_number_not_in_evidence_or_calculations_is_unsupported():
    report = verify(["21.6"], "percent", [CELL, OLD], [CHANGE])
    assert not report.passed and report.has_unsupported_number
    assert "21.6" in report.feedback()


def test_copied_spans_are_supported_where_they_appear():
    report = verify(["$9,028"], "", [PARA])
    assert report.passed and report.supported[0].support == "paragraph 4"
    phrase = verify(["related to software"], "", [PARA])
    assert phrase.passed and phrase.supported[0].kind == "span"


def test_invented_text_is_unsupported():
    report = verify(["related to hardware"], "", [PARA])
    assert not report.passed and not report.has_unsupported_number


def test_sign_is_ignored_for_decreases():
    assert verify(["234"], "", [], [Calculation("1086 - 1320", -234.0)]).passed


def test_table_amount_needs_the_unit_the_table_states():
    hints = ["(In thousands)"]
    assert verify(["1,320"], "thousand", [CELL], unit_hints=hints).passed
    wrong = verify(["1,320"], "million", [CELL], unit_hints=hints)
    assert wrong.scale_problems and "million" in wrong.scale_problems[0]
    assert verify(["1,320"], "", [CELL], unit_hints=hints).passed  # lenient by default
    missing = verify(["1,320"], "", [CELL], unit_hints=hints, strict=True)
    assert missing.scale_problems and "thousand" in missing.scale_problems[0]


def test_a_unit_the_report_never_states_cannot_be_checked():
    assert verify(["1,320"], "million", [CELL], unit_hints=[]).passed


def test_paragraph_quotes_keep_scale_empty_even_in_a_thousands_report():
    assert verify(["$9,028"], "", [PARA], unit_hints=["In thousands"]).passed
    doubled = verify(["$3.0 million"], "million", [Evidence("was $3.0 million", "paragraph 1")])
    assert any("already names its unit" in p for p in doubled.scale_problems)


def test_percent_scale_needs_a_percentage_somewhere():
    assert not verify(["1,320"], "percent", [CELL]).passed
    assert verify(["18.6%"], "", [Evidence("18.6%", "table row 8, column c1")]).passed


def test_empty_answer_never_passes():
    assert not verify([], "", [CELL]).passed


@given(st.integers(min_value=-(10**9), max_value=10**9))
def test_numbers_in_reads_report_formats(n):
    written = f"{abs(n):,}"
    text = f"({written})" if n < 0 else written
    assert numbers_in(f"total was {text} in 2019")[0][1] == n
