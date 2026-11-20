"""Confidence intervals and paired tests.

The sign test is copied from tableqa (tableqa/eval/compare.py). tableqa has no Wilson
interval, so that is new here.
"""

from math import comb, sqrt

Z_95 = 1.959963984540054


def wilson(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion (95% by default)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test: p-value for a wins/losses split under no difference.

    Ties (questions where both systems agree) carry no information and are left out.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(baseline: dict[str, bool], variant: dict[str, bool]) -> dict:
    """Compare two runs question by question on a boolean outcome where True is good.

    Wins are questions where only the variant is good, losses where only the baseline
    is. For a bad outcome such as confident-wrong, pass its negation.
    """
    if baseline.keys() != variant.keys():
        raise ValueError("the two runs did not answer the same questions")
    wins = sum(variant[q] and not baseline[q] for q in baseline)
    losses = sum(baseline[q] and not variant[q] for q in baseline)
    return {
        "n": len(baseline),
        "baseline": sum(baseline.values()),
        "variant": sum(variant.values()),
        "wins": wins,
        "losses": losses,
        "ties": len(baseline) - wins - losses,
        "p_value": sign_test(wins, losses),
    }
