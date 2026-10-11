"""Keyword safety net that catches risky wording even when the LLM misses it.

The lists live here and nowhere else. Tune them against the dev set only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Wording that suggests fraud, a scam, account takeover or a person at risk.
CRITICAL_PATTERNS = (
    r"(didn't|did not) make",
    r"(don't|do not|didn't|did not) recogni[sz]e",
    r"unauthori[sz]ed",
    r"scam(s|med|mer|mers|ming)?",
    r"stolen",
    r"fraud(s|ulent|ster|sters)?",
    r"taken over",
    r"gave (them|him|her) the code",
    r"(don't|do not) feel safe",
    r"(don't|do not) see the point",
    r"controls (all )?(our|my) money",
    r"in my name",
    r"skimm(ed|ing)",
    r"remote access",
)

# Wording that suggests the customer is struggling to pay.
HARDSHIP_PATTERNS = (
    r"lost my job",
    r"made redundant",
    r"redundan(t|cy)",
    r"(can'?t|cannot) (keep up|afford|pay|make|work)",
    r"hardship",
    r"behind on",
    r"reduced (hours|pay)",
    r"hours (cut|reduced)",
    r"on centrelink",
    r"between jobs",
)


# Apostrophe look-alikes that phones and email clients produce instead of a plain '.
_APOSTROPHES = re.compile("[\u2019\u2018\u02bc\u00b4`]")


def _compile(patterns: tuple[str, ...]) -> list[re.Pattern[str]]:
    """Whole-word, case-insensitive matching so 'scam' does not fire on 'scampi'.

    A space in a pattern matches any run of whitespace, so a line break or a double space
    in the middle of a phrase does not hide it.
    """
    flexible = [pattern.replace(" ", r"\s+") for pattern in patterns]
    return [re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE) for pattern in flexible]


_CRITICAL = _compile(CRITICAL_PATTERNS)
_HARDSHIP = _compile(HARDSHIP_PATTERNS)


@dataclass(frozen=True)
class KeywordHits:
    """The words that matched, kept so a reviewer can see why a rule fired."""

    critical: list[str] = field(default_factory=list)
    hardship: list[str] = field(default_factory=list)

    def all(self) -> list[str]:
        """Every hit in one list, critical first, for logging and display."""
        return [*self.critical, *self.hardship]


def scan(text: str) -> KeywordHits:
    """Find risky wording in customer text, whatever apostrophe character was typed."""
    normalised = _APOSTROPHES.sub("'", text)
    return KeywordHits(
        critical=_matches(_CRITICAL, normalised),
        hardship=_matches(_HARDSHIP, normalised),
    )


def _matches(patterns: list[re.Pattern[str]], text: str) -> list[str]:
    """Matched words in order of first appearance, lower-cased, single-spaced, each once."""
    found = [(m.start(), _tidy(m.group(0))) for p in patterns for m in p.finditer(text)]
    unique: list[str] = []
    for _, word in sorted(found):
        if word not in unique:
            unique.append(word)
    return unique


def _tidy(matched: str) -> str:
    """Lower-case a match and collapse its whitespace so the same phrase always reads the same."""
    return " ".join(matched.lower().split())
