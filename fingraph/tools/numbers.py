"""Read a number the way a financial report writes it."""

import re

# Optional sign, optional currency, digits with or without thousands separators.
_NUMBER = re.compile(r"[-+]?[$€£¥]?\s*(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?")


def parse_financial(text: str) -> float | None:
    """'1,320' -> 1320, '(19)' -> -19, '$ 2.5' -> 2.5, '18.6%' -> 18.6; else None.

    Parentheses mean a negative amount, as in accounts. A trailing % is dropped (the
    value stays in percent units). Dashes, blanks and anything with words are None.
    """
    s = text.strip()
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1].strip()
    s = s.removesuffix("%").strip()
    if not s or not any(c.isdigit() for c in s) or not _NUMBER.fullmatch(s):
        return None
    if s[0] in "+-":
        negative, s = negative != (s[0] == "-"), s[1:]
    value = float(s.lstrip("$€£¥").strip().replace(",", ""))
    return -value if negative else value
