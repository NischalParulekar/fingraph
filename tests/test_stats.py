import pytest

from fingraph.eval.stats import paired, sign_test, wilson


def test_wilson_matches_a_known_value():
    # 50/100: the textbook Wilson 95% interval is about (0.4038, 0.5962).
    low, high = wilson(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-4)
    assert high == pytest.approx(0.5962, abs=1e-4)


def test_wilson_stays_inside_zero_and_one():
    assert wilson(0, 10)[0] == 0.0
    assert wilson(10, 10)[1] == pytest.approx(1.0)
    assert wilson(0, 0) == (0.0, 1.0)


def test_sign_test():
    assert sign_test(0, 0) == 1.0
    assert sign_test(5, 5) == 1.0
    assert sign_test(10, 0) == pytest.approx(2 / 2**10)
    assert sign_test(3, 7) == sign_test(7, 3)


def test_paired_counts_wins_losses_and_ties():
    baseline = {"a": True, "b": False, "c": False, "d": True}
    variant = {"a": True, "b": True, "c": False, "d": False}
    result = paired(baseline, variant)
    assert (result["wins"], result["losses"], result["ties"]) == (1, 1, 2)
    with pytest.raises(ValueError):
        paired(baseline, {"a": True})
