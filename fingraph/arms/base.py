"""What every arm takes and returns, and the parts all arms share."""

from dataclasses import dataclass, field
from typing import Protocol

from fingraph.data.tatqa import Example
from fingraph.scales import SCALES


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
#
# v2 (2026-11-20), after the first dev run. The conventions were measured on the whole
# official dev split (never on held-out): text spans have scale "" in 343 of 349 cases;
# copied cells with a % sign have scale "" in 46 of 46; a "change" keeps its sign (121 of
# 314 are negative); a question that asks only for a "decrease" has a positive answer.
ANSWER_RULES = """\
How the answer is scored
- It is compared with a gold answer automatically, so give only the answer itself, \
never a sentence or an explanation.

Answers copied from the report
- Copy the shortest exact span that answers the question, word for word. Do not add \
words that are not part of the answer ("Calculated based on ..." when the text says \
"(2) Calculated based on ...": answer "based on ..."; not "They result from ...").
- From a paragraph: copy the words as written, including any units or currency in \
them (for example "$3.0 million" or "over 50 countries", not a bare number). Scale \
is "".
- From a table cell: copy the cell text as printed (for example "1,320", "(2,088)", \
"36%"). Scale is the unit the table states for that amount ("thousand", "million" or \
"billion", for example from a header "In thousands"), or "" if it states none or the \
cell is a year, a label or a percentage.
- If the question asks for several items, give each one as its own list entry. A \
single phrase that happens to contain "and" is one entry.

Answers you calculate (differences, changes, totals, averages, ratios, percentages)
- Compute them with the calculate tool, never in your head, and give the result \
rounded to 2 decimal places as a bare number, without currency signs, commas or %.
- A change is new - old and keeps its sign, so a fall is negative. A percentage change \
is (new - old) / old * 100. If the question asks only how much something decreased, \
give the decrease as a positive number.
- Scale: "percent" if the result is a percentage (give 12.5 for 12.5%); otherwise the \
unit the inputs are stated in ("thousand", "million", "billion"), or "" for plain \
numbers and ratios.
- Counts: the count as a bare number, scale "".
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
