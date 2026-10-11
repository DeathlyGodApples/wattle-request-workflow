"""FAQ selection: the LLM only picks an entry; the answer text is always the FAQ's own."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from wattle.data import load_faq
from wattle.llm import LLMClient, LLMError
from wattle.models import FaqEntry, FaqSelection, Request
from wattle.prompts import faq_system_prompt, faq_user_prompt

logger = logging.getLogger(__name__)

# Below this the pick is treated as a guess and the enquiry goes to a person instead.
FAQ_MIN_CONFIDENCE = 0.6


@dataclass(frozen=True)
class FaqMatch:
    """The outcome of FAQ selection, with the reason kept for the decision log."""

    entry: FaqEntry | None  # the matched entry, or None when the enquiry must be handed off
    faq_id: str | None  # what the LLM picked, kept even when the pick was rejected
    confidence: float
    reason: str
    llm_error: bool = False

    @property
    def matched(self) -> bool:
        """True when there is an FAQ answer that may be sent."""
        return self.entry is not None


def select_faq(req: Request, llm: LLMClient, entries: Sequence[FaqEntry] | None = None) -> FaqMatch:
    """Ask the LLM which FAQ entry answers the request; any failure means no match."""
    faq = list(entries) if entries is not None else load_faq()
    try:
        output = llm.generate_json(faq_system_prompt(faq), faq_user_prompt(req), FaqSelection)
        selection = FaqSelection.model_validate(output.model_dump())
    except (LLMError, ValidationError) as exc:
        logger.warning("FAQ selection failed for %s: %s", req.id, exc)
        return FaqMatch(None, None, 0.0, f"LLM error: {exc}", llm_error=True)
    return _to_match(selection, faq)


def _to_match(selection: FaqSelection, faq: list[FaqEntry]) -> FaqMatch:
    """Accept the pick only if it names a real entry with enough confidence."""
    by_id = {entry.id: entry for entry in faq}
    picked, confidence = selection.faq_id, selection.confidence
    if picked is None:
        return FaqMatch(None, None, confidence, "no FAQ entry covers this question")
    if picked not in by_id:
        return FaqMatch(None, picked, confidence, f"unknown FAQ id {picked!r}")
    if confidence < FAQ_MIN_CONFIDENCE:
        reason = f"confidence {confidence:.2f} is below {FAQ_MIN_CONFIDENCE}"
        return FaqMatch(None, picked, confidence, reason)
    return FaqMatch(by_id[picked], picked, confidence, "matched")
