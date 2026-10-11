"""Single live smoke test: proves the real Gemini call works. Skipped unless RUN_LIVE=1."""

from __future__ import annotations

import pytest

from wattle.classifier import classify
from wattle.llm import GeminiClient
from wattle.models import Request, RequestType


@pytest.mark.live
def test_gemini_classifies_a_made_up_complaint() -> None:
    req = Request(
        id="LIVE-1",
        channel="email",
        subject="Late fee",
        body="You charged me a $15 late fee even though I paid on time. I want it refunded.",
    )
    result = classify(req, GeminiClient())
    assert result.llm_error is False, result.rationale
    assert result.request_type is RequestType.COMPLAINT
    assert 0.0 <= result.confidence <= 1.0
