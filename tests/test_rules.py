"""Tests for rules. M2 covers the AI-only decision; the R0-R4 rules are added in M3."""

from __future__ import annotations

import pytest

from wattle.models import (
    Classification,
    Extracted,
    FinalType,
    Product,
    RequestType,
    Urgency,
)
from wattle.rules import ai_only_decision


def make_classification(request_type: str, *, llm_error: bool = False) -> Classification:
    """Build a Classification where only the fields the rules read are varied."""
    return Classification(
        request_type=RequestType(request_type),
        product=Product.UNKNOWN,
        risk_flags=[],
        extracted=Extracted(summary="summary"),
        missing_info=[],
        confidence=0.9,
        rationale="rationale",
        llm_error=llm_error,
    )


@pytest.mark.parametrize(
    ("request_type", "final_type", "urgency", "requires_human"),
    [
        ("critical", FinalType.CRITICAL, Urgency.CRITICAL, True),
        ("hardship", FinalType.HARDSHIP, Urgency.HIGH, True),
        ("complaint", FinalType.COMPLAINT, Urgency.HIGH, False),
        ("service_request", FinalType.SERVICE_REQUEST, Urgency.MEDIUM, False),
        ("general_enquiry", FinalType.GENERAL_ENQUIRY, Urgency.LOW, False),
        ("unclear", FinalType.HUMAN_REVIEW, Urgency.MEDIUM, True),
    ],
)
def test_ai_only_decision_follows_the_llm_type(
    request_type: str, final_type: FinalType, urgency: Urgency, requires_human: bool
) -> None:
    decision = ai_only_decision(make_classification(request_type))
    assert decision.final_type is final_type
    assert decision.urgency is urgency
    assert decision.requires_human is requires_human
    assert decision.rules_fired == ["AI_ONLY"]


def test_ai_only_decision_sends_llm_errors_to_human_review() -> None:
    decision = ai_only_decision(make_classification("complaint", llm_error=True))
    assert decision.final_type is FinalType.HUMAN_REVIEW
    assert decision.requires_human is True
