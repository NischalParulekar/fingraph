"""What every arm takes and returns, and the parts all arms share."""

from dataclasses import dataclass, field
from typing import Protocol

from fingraph.data.tatqa import Example
from fingraph.eval.scorer import SCALES


@dataclass
class Prediction:
    answer: list[str]  # empty means the arm gave no answer (an abstention)
    scale: str
    latency_ms: float
    cost_usd: float = 0.0
    tokens: int = 0
    llm_calls: int = 0
    error: str | None = None  # why there is no answer, if the arm failed
    trace: list = field(default_factory=list)  # every model call and tool call, in order


class Arm(Protocol):
    name: str

    @property
    def config(self) -> dict: ...

    def answer(self, example: Example) -> Prediction: ...


# How TAT-QA answers are scored, told to every arm in the same words. The official
# scorer compares numbers after rounding to 2 decimals and multiplying by the scale, and
# compares text after normalisation, so the form of an answer decides whether it counts.
ANSWER_RULES = """\
How the answer is scored
- It is compared with a gold answer automatically, so give only the answer itself, \
never a sentence or an explanation.
- Text answers: copy the words exactly as they appear in the table or the paragraph.
- If the question asks for several items, give each one as its own list entry.
- Numbers you calculate (differences, changes, totals, averages, ratios, percentages): \
compute them with the calculate tool, never in your head, and give the result rounded \
to 2 decimal places as a bare number, without currency signs or %.
- Percentage change is (new - old) / old * 100.
- Counts: the count as a bare number.
- `scale` is the unit the number is in. Use "thousand", "million" or "billion" for an \
amount that the report states in those units (for example a table headed "In \
thousands"), also when you copy the amount as it is. Use "percent" for any percentage \
(give 12.5 for 12.5%). Use "" for counts, years, plain ratios and text answers.
"""

FINAL_ANSWER_TOOL = {
    "name": "final_answer",
    "description": "Give the final answer and finish. This is the only way to answer.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The answer: one entry, or one entry per item for a list.",
            },
            "scale": {
                "type": "string",
                "enum": list(SCALES),
                "description": "The unit the answer is expressed in.",
            },
        },
        "required": ["answer", "scale"],
        "additionalProperties": False,
    },
}
