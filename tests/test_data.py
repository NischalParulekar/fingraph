import json
from collections import Counter

import pytest

from fingraph.data import tatqa
from fingraph.data.splits import SPECS, SPLITS_DIR, allocate, load_split, sample, stratum

HAVE_DATA = all((tatqa.DATA_DIR / name).exists() for name, _ in tatqa.FILES.values())
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="TAT-QA not downloaded (make data)")


def test_allocate_is_proportional_and_exact():
    counts = allocate({"a": 500, "b": 300, "c": 200}, 10)
    assert counts == {"a": 5, "b": 3, "c": 2}
    counts = allocate({"a": 7, "b": 7, "c": 7}, 10)
    assert sum(counts.values()) == 10
    assert max(counts.values()) - min(counts.values()) <= 1


def test_allocate_rejects_impossible_sizes():
    with pytest.raises(ValueError):
        allocate({"a": 3}, 4)


@pytest.mark.parametrize("name", list(SPECS))
def test_committed_splits_are_well_formed(name):
    record = json.loads((SPLITS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    _, n = SPECS[name]
    assert record["n"] == n == len(record["ids"]) == len(set(record["ids"]))
    assert sum(record["strata"].values()) == n


def test_dev_and_heldout_do_not_overlap():
    ids = {
        name: set(json.loads((SPLITS_DIR / f"{name}.json").read_text(encoding="utf-8"))["ids"])
        for name in SPECS
    }
    assert not ids["dev"] & ids["heldout"]


@needs_data
@pytest.mark.parametrize("name", list(SPECS))
def test_committed_splits_match_what_the_code_builds(name):
    official, n = SPECS[name]
    record = json.loads((SPLITS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    assert sample(tatqa.load(official), n) == record["ids"]


@needs_data
def test_load_split_returns_examples_with_their_context():
    examples = load_split("dev")
    assert len(examples) == 150
    strata = Counter(stratum(e) for e in examples)
    assert strata["table/arithmetic"] > strata["text/arithmetic"]
    for e in examples:
        assert e.table and e.paragraphs
        assert e.gold["uid"] == e.uid
        assert [p.order for p in e.paragraphs] == sorted(p.order for p in e.paragraphs)
