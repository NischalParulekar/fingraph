"""Score one prediction with the official TAT-QA scorer.

The scorer itself is vendored unmodified in tatqa_official/. Its modules import each
other by bare name, so that directory goes on sys.path rather than being edited.
"""

import sys
from pathlib import Path

_OFFICIAL = Path(__file__).resolve().parent / "tatqa_official"
if str(_OFFICIAL) not in sys.path:
    sys.path.insert(0, str(_OFFICIAL))

from tatqa_metric import TaTQAEmAndF1  # noqa: E402

SCALES = ("", "thousand", "million", "billion", "percent")


def score(gold: dict, answer: list[str] | None, scale: str) -> tuple[float, float]:
    """(EM, F1) for one question, exactly as the official evaluate_json computes them.

    `gold` is the question's record from the dataset file. An empty or missing answer
    scores 0, as an unanswered question does in the official script.
    """
    metric = TaTQAEmAndF1()
    metric(ground_truth=gold, prediction=answer or None, pred_scale=scale)
    em, f1, _, _ = metric.get_overall_metric()
    return float(em), float(f1)
