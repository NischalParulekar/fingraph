"""Check a draft answer against the evidence that was gathered for it.

Standalone on purpose: plain dataclasses in, a report out, no LangGraph and no model
calls, so the false-completion benchmark can import it and every arm can be checked
offline with the same code.

For each answer entry:
- If the entry appears word for word in an evidence text, it is supported as a span.
- Otherwise every number in it must be supported: equal (to 2 decimals, ignoring sign)
  to a number in an evidence text or to the result of a logged calculation. An entry
  with no numbers that is not a span of the evidence is unsupported.

Scale is checked separately:
- "percent" needs a percentage in the supporting evidence, a calculation that
  multiplies or divides by 100, or a "%"/"percent" in the unit hints;
- "thousand" / "million" / "billion" is flagged when the report states a different
  unit and never this one. A report that states no unit at all cannot be checked (about
  1 in 10 scaled TAT-QA dev questions), so that passes;
- a non-empty scale on an entry whose own text already names a unit ("$3.0 million")
  is flagged, since the scale would count the unit twice;
- strict=True only: an empty scale on a number read from a table cell, when the unit
  hints name a unit, is flagged as a possibly missing scale. Off by default: on the
  dev runs it rejected 9 correct answers for every 3 wrong ones it caught, because the
  gold scale is often "" for table numbers in reports that state a unit elsewhere.

Sign is ignored when matching numbers because questions about a decrease are answered
with a positive number. That makes the check slightly lenient, never stricter.
"""

import re
from dataclasses import dataclass, field

_NUMBER = re.compile(r"\(?-?\$?\s?\d[\d,]*(?:\.\d+)?\)?%?|\(?-?\$?\s?\.\d+\)?%?")
_UNITS = {
    "thousand": re.compile(r"thousand|\$\s?'?000|in 000s|'000", re.IGNORECASE),
    "million": re.compile(r"million|\$\s?m\b|\bmn\b", re.IGNORECASE),
    "billion": re.compile(r"billion|\$\s?bn?\b|\bbn\b", re.IGNORECASE),
    "percent": re.compile(r"%|percent", re.IGNORECASE),
}
_UNIT_IN_ANSWER = re.compile(r"\b(thousand|million|billion)s?\b|\d\s?(m|bn)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Evidence:
    text: str
    location: str  # e.g. "table row 3, column c1" or "paragraph 4"


@dataclass(frozen=True)
class Calculation:
    expression: str
    result: float


@dataclass(frozen=True)
class Claim:
    entry: str  # the answer entry this claim comes from
    kind: str  # "span" or "number"
    value: str  # the span text, or the number as written
    support: str | None  # where it was found, None if unsupported


@dataclass
class Report:
    supported: list[Claim] = field(default_factory=list)
    unsupported: list[Claim] = field(default_factory=list)
    scale_problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.unsupported and not self.scale_problems

    @property
    def has_unsupported_number(self) -> bool:
        return any(c.kind == "number" for c in self.unsupported)

    def feedback(self) -> str:
        """The report in words, for a retry."""
        lines = [f"- not found in the evidence: {c.value!r}" for c in self.unsupported]
        lines += [f"- scale: {p}" for p in self.scale_problems]
        return "\n".join(lines)


_UNIT_HINT = re.compile(r"thousand|million|billion|'000|\$000|%|percent", re.IGNORECASE)


def unit_hints(cells: list[str], paragraphs: list[str] = ()) -> list[str]:
    """The table cells and paragraph sentences that name a unit ("In thousands", ...)."""
    hints = [cell for cell in cells if cell and _UNIT_HINT.search(cell)]
    for text in paragraphs:
        hints += [s for s in re.split(r"(?<=[.;])\s+", text) if _UNIT_HINT.search(s)]
    return hints


def parse_number(text: str) -> float | None:
    s = text.strip().replace("$", "").replace(",", "").replace(" ", "").rstrip("%")
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negative else value


def numbers_in(text: str) -> list[tuple[str, float]]:
    found = []
    for match in _NUMBER.finditer(text):
        value = parse_number(match.group(0))
        if value is not None:
            found.append((match.group(0).strip(), value))
    return found


def _same(a: float, b: float) -> bool:
    return round(abs(a), 2) == round(abs(b), 2)


def _squash(text: str) -> str:
    return " ".join(text.split()).lower()


def verify(
    answer: list[str],
    scale: str,
    evidence: list[Evidence],
    calculations: list[Calculation] = (),
    unit_hints: list[str] = (),
    strict: bool = False,
) -> Report:
    report = Report()
    if not answer:
        report.unsupported.append(Claim("", "span", "", None))  # nothing to check
        return report

    evidence_numbers = [(e, n) for e in evidence for _, n in numbers_in(e.text)]
    hints = " ".join(unit_hints)
    support_texts: list[str] = []  # evidence texts that supported something
    from_calculation = False
    from_table_number = False  # an answer number read straight from a table cell

    for entry in answer:
        span_source = next(
            (e for e in evidence if entry.strip() and _squash(entry) in _squash(e.text)), None
        )
        if span_source is not None:
            report.supported.append(Claim(entry, "span", entry, span_source.location))
            support_texts.append(span_source.text)
            from_table_number |= span_source.location.startswith("table") and bool(
                numbers_in(entry)
            )
            continue
        numbers = numbers_in(entry)
        if not numbers:
            report.unsupported.append(Claim(entry, "span", entry, None))
            continue
        for written, value in numbers:
            calc = next((c for c in calculations if _same(c.result, value)), None)
            if calc is not None:
                support = f"calculation {calc.expression}"
                from_calculation = True
                support_texts.append(calc.expression)
            else:
                hit = next((e for e, n in evidence_numbers if _same(n, value)), None)
                support = hit.location if hit else None
                if hit:
                    support_texts.append(hit.text)
                    from_table_number |= hit.location.startswith("table")
            target = report.supported if support else report.unsupported
            target.append(Claim(entry, "number", written, support))

    report.scale_problems = _scale_problems(
        answer, scale, support_texts, hints, from_calculation, from_table_number, strict
    )
    return report


def _scale_problems(answer, scale, support_texts, hints, from_calculation, from_table, strict):
    problems = []
    support = " ".join(support_texts)
    if scale == "percent":
        times_100 = from_calculation and "100" in support
        if not (times_100 or _UNITS["percent"].search(support) or _UNITS["percent"].search(hints)):
            problems.append('"percent", but nothing in the evidence is a percentage')
    elif scale in ("thousand", "million", "billion"):
        stated_here = _UNITS[scale].search(support) or _UNITS[scale].search(hints)
        others = [u for u in ("thousand", "million", "billion") if u != scale]
        stated_other = [u for u in others if _UNITS[u].search(hints)]
        if not stated_here and stated_other:
            problems.append(f'"{scale}", but the report states amounts in {stated_other[0]}s')
        if any(_UNIT_IN_ANSWER.search(entry) for entry in answer):
            problems.append(f'"{scale}", but the answer text already names its unit')
    elif (
        strict and scale == "" and from_table and not any(_UNIT_IN_ANSWER.search(a) for a in answer)
    ):
        stated = [u for u in ("thousand", "million", "billion") if _UNITS[u].search(hints)]
        if stated and not any(_UNITS["percent"].search(a) for a in answer):
            problems.append(f'"" (none), but the table states amounts in {stated[0]}s')
    return problems
