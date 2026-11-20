import hashlib
from pathlib import Path

import pytest

from fingraph.eval.scorer import score

OFFICIAL = Path(__file__).resolve().parents[1] / "fingraph" / "eval" / "tatqa_official"
UPSTREAM_SHA256 = {
    "tatqa_metric.py": "2aeeac479f89f8c76300af1cc0e8d098eb86af84bc386b38b6ab4af484a6dea8",
    "tatqa_utils.py": "a84bb2f960737cf0a53733637a674cc4b20ef030a2be6a4b21dc2c4356f415ec",
    "tatqa_eval.py": "a4110626c477e7ed88f42b5edeb8449eaa9a4370740a627631b4a462bcf43762",
}


@pytest.mark.parametrize("name", list(UPSTREAM_SHA256))
def test_vendored_scorer_is_unmodified(name):
    assert hashlib.sha256((OFFICIAL / name).read_bytes()).hexdigest() == UPSTREAM_SHA256[name]


def gold(answer, answer_type, scale="", answer_from="table"):
    return {
        "uid": "q",
        "answer": answer,
        "answer_type": answer_type,
        "answer_from": answer_from,
        "scale": scale,
    }


def test_arithmetic_answer_is_compared_as_a_number_with_its_scale():
    g = gold(1496.5, "arithmetic", "million")
    assert score(g, ["1496.5"], "million") == (1.0, 1.0)
    assert score(g, ["1,496.50"], "million") == (1.0, 1.0)
    assert score(g, ["1496.5"], "thousand") == (0.0, 0.0)  # wrong scale is wrong
    assert score(g, ["1496.6"], "million") == (0.0, 0.0)


def test_percent_answers():
    g = gold(23.42, "arithmetic", "percent")
    assert score(g, ["23.42"], "percent")[0] == 1.0
    assert score(g, ["23.42%"], "percent")[0] == 1.0


def test_spans_get_partial_f1_but_no_em():
    g = gold(["fixed or variable fee arrangement"], "span", answer_from="text")
    em, f1 = score(g, ["variable fee arrangement"], "")
    assert em == 0.0 and 0.0 < f1 < 1.0


def test_multi_span_order_does_not_matter():
    g = gold(["Wireless", "Cable"], "multi-span")
    assert score(g, ["Cable", "Wireless"], "")[0] == 1.0


def test_no_answer_scores_zero():
    g = gold(5, "count")
    assert score(g, [], "") == (0.0, 0.0)
    assert score(g, None, "") == (0.0, 0.0)
