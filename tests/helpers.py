"""Shared builders for tests, so each test states only what it changes."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from wattle.llm import FakeLLM, LLMError
from wattle.models import Classification, FaqSelection, LLMClassification, Request

# A fixed "now" in Melbourne time (daylight saving, UTC+11) so due dates are predictable.
NOW = datetime(2026, 10, 12, 9, 30, tzinfo=ZoneInfo("Australia/Melbourne"))


def make_request(body: str = "Hello", *, subject: str = "Subject", id: str = "T-1") -> Request:
    """A request written for the tests; never a row from the held-out test set."""
    return Request(id=id, channel="email", subject=subject, body=body)


def make_classification(request_type: str = "complaint", **overrides: object) -> Classification:
    """A valid Classification of the given type; pass only the fields a test cares about."""
    fields: dict[str, object] = {
        "request_type": request_type,
        "product": "unknown",
        "risk_flags": [],
        "extracted": {"summary": "What the customer wants, in one sentence."},
        "missing_info": [],
        "confidence": 0.9,
        "rationale": "Why this type was chosen.",
    }
    fields.update(overrides)
    return Classification.model_validate(fields)


def scripted_llm(classification: Classification, faq: FaqSelection | None = None) -> FakeLLM:
    """A fake that answers the classifier call and, if given, the FAQ-selection call.

    The two calls ask for different schemas, so the fake answers by schema. With no `faq`
    answer the FAQ call fails, which is how a test simulates an outage at that step.
    """

    def respond(system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        if schema is FaqSelection:
            if faq is None:
                raise LLMError("no FAQ answer scripted")
            return faq
        return LLMClassification(**classification.model_dump(exclude={"llm_error"}))

    return FakeLLM(respond)
