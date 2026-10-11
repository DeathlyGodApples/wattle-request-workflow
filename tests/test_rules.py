"""Tests for rules: one table row per rule, plus the precedence between them."""

from __future__ import annotations

import pytest

from wattle.keywords import KeywordHits
from wattle.models import (
    Classification,
    Extracted,
    FinalType,
    Product,
    RequestType,
    RiskFlag,
    Urgency,
)
from wattle.rules import ai_only_decision, apply_rules

NO_HITS = KeywordHits()
CRITICAL_HIT = KeywordHits(critical=["stolen"])
HARDSHIP_HIT = KeywordHits(hardship=["lost my job"])


def make_classification(
    request_type: str,
    *,
    flags: list[str] | None = None,
    confidence: float = 0.9,
    llm_error: bool = False,
) -> Classification:
    """Build a Classification where only the fields the rules read are varied."""
    return Classification(
        request_type=RequestType(request_type),
        product=Product.UNKNOWN,
        risk_flags=[RiskFlag(flag) for flag in flags or []],
        extracted=Extracted(summary="summary"),
        missing_info=[],
        confidence=confidence,
        rationale="rationale",
        llm_error=llm_error,
    )


# (name, classification, keyword hits, expected final type, rule that decides or None)
RULE_CASES = [
    ("R0 llm error", make_classification("unclear", confidence=0.0, llm_error=True), NO_HITS,
     FinalType.HUMAN_REVIEW, "R0"),
    ("R1 llm says critical", make_classification("critical"), NO_HITS,
     FinalType.CRITICAL, "R1"),
    ("R1 fraud flag", make_classification("complaint", flags=["fraud"]), NO_HITS,
     FinalType.CRITICAL, "R1"),
    ("R1 scam flag", make_classification("service_request", flags=["scam"]), NO_HITS,
     FinalType.CRITICAL, "R1"),
    ("R1 unauthorised flag", make_classification("complaint", flags=["unauthorised_transaction"]),
     NO_HITS, FinalType.CRITICAL, "R1"),
    ("R1 self-harm flag", make_classification("hardship", flags=["self_harm"]), NO_HITS,
     FinalType.CRITICAL, "R1"),
    ("R1 financial abuse flag", make_classification("general_enquiry", flags=["financial_abuse"]),
     NO_HITS, FinalType.CRITICAL, "R1"),
    ("R1 critical keyword", make_classification("general_enquiry"), CRITICAL_HIT,
     FinalType.CRITICAL, "R1"),
    ("R2 prompt injection", make_classification("general_enquiry", flags=["prompt_injection"]),
     NO_HITS, FinalType.HUMAN_REVIEW, "R2"),
    ("R3 complaint + difficulty flag",
     make_classification("complaint", flags=["financial_difficulty"]), NO_HITS,
     FinalType.HARDSHIP, "R3"),
    ("R3 service request + hardship keyword", make_classification("service_request"),
     HARDSHIP_HIT, FinalType.HARDSHIP, "R3"),
    ("R3 enquiry + hardship keyword", make_classification("general_enquiry"), HARDSHIP_HIT,
     FinalType.HARDSHIP, "R3"),
    ("R4 low confidence", make_classification("complaint", confidence=0.69), NO_HITS,
     FinalType.HUMAN_REVIEW, "R4"),
    ("R4 unclear", make_classification("unclear"), NO_HITS, FinalType.HUMAN_REVIEW, "R4"),
    ("no rule: complaint", make_classification("complaint"), NO_HITS, FinalType.COMPLAINT, None),
    ("no rule: service request", make_classification("service_request"), NO_HITS,
     FinalType.SERVICE_REQUEST, None),
    ("no rule: enquiry", make_classification("general_enquiry"), NO_HITS,
     FinalType.GENERAL_ENQUIRY, None),
    ("no rule: hardship", make_classification("hardship"), NO_HITS, FinalType.HARDSHIP, None),
    ("no rule: confidence exactly at threshold", make_classification("complaint", confidence=0.7),
     NO_HITS, FinalType.COMPLAINT, None),
    ("no rule: hardship keyword on a hardship request", make_classification("hardship"),
     HARDSHIP_HIT, FinalType.HARDSHIP, None),
]  # fmt: skip


@pytest.mark.parametrize(
    ("classification", "hits", "final_type", "deciding_rule"),
    [case[1:] for case in RULE_CASES],
    ids=[case[0] for case in RULE_CASES],
)
def test_each_rule(
    classification: Classification,
    hits: KeywordHits,
    final_type: FinalType,
    deciding_rule: str | None,
) -> None:
    decision = apply_rules(classification, hits, threshold=0.7)
    assert decision.final_type is final_type
    if deciding_rule is None:
        assert decision.rules_fired == []
    else:
        assert decision.rules_fired[0] == deciding_rule


def test_r1_beats_r4_low_confidence_critical_stays_critical() -> None:
    c = make_classification("critical", confidence=0.2)
    decision = apply_rules(c, NO_HITS, threshold=0.7)
    assert decision.final_type is FinalType.CRITICAL
    assert decision.rules_fired == ["R1", "R4"]


def test_r1_keyword_beats_r4_unclear() -> None:
    decision = apply_rules(make_classification("unclear", confidence=0.1), CRITICAL_HIT)
    assert decision.final_type is FinalType.CRITICAL
    assert decision.rules_fired[0] == "R1"


def test_r1_beats_r2_and_r3() -> None:
    c = make_classification("complaint", flags=["scam", "prompt_injection", "financial_difficulty"])
    decision = apply_rules(c, HARDSHIP_HIT, threshold=0.7)
    assert decision.final_type is FinalType.CRITICAL
    assert decision.rules_fired == ["R1", "R2", "R3"]


def test_r2_beats_r3_injection_is_not_auto_routed_to_hardship() -> None:
    c = make_classification("complaint", flags=["prompt_injection", "financial_difficulty"])
    decision = apply_rules(c, NO_HITS, threshold=0.7)
    assert decision.final_type is FinalType.HUMAN_REVIEW
    assert decision.rules_fired == ["R2", "R3"]


def test_r3_beats_r4_low_confidence_hardship_signal_goes_to_hardship() -> None:
    c = make_classification("complaint", confidence=0.4)
    decision = apply_rules(c, HARDSHIP_HIT, threshold=0.7)
    assert decision.final_type is FinalType.HARDSHIP
    assert decision.rules_fired == ["R3", "R4"]


def test_r0_beats_r1_an_llm_failure_goes_to_a_person_even_with_keywords() -> None:
    c = make_classification("unclear", confidence=0.0, llm_error=True)
    decision = apply_rules(c, CRITICAL_HIT, threshold=0.7)
    assert decision.final_type is FinalType.HUMAN_REVIEW
    assert decision.rules_fired[:2] == ["R0", "R1"]
    assert decision.requires_human is True


def test_r3_does_not_apply_to_unclear_requests() -> None:
    decision = apply_rules(make_classification("unclear"), HARDSHIP_HIT, threshold=0.7)
    assert decision.final_type is FinalType.HUMAN_REVIEW
    assert decision.rules_fired == ["R4"]


def test_threshold_is_configurable() -> None:
    c = make_classification("complaint", confidence=0.8)
    assert apply_rules(c, NO_HITS, threshold=0.7).final_type is FinalType.COMPLAINT
    assert apply_rules(c, NO_HITS, threshold=0.9).final_type is FinalType.HUMAN_REVIEW


@pytest.mark.parametrize(
    ("final_type", "urgency", "requires_human"),
    [
        (FinalType.CRITICAL, Urgency.CRITICAL, True),
        (FinalType.HARDSHIP, Urgency.HIGH, True),
        (FinalType.COMPLAINT, Urgency.HIGH, False),
        (FinalType.SERVICE_REQUEST, Urgency.MEDIUM, False),
        (FinalType.GENERAL_ENQUIRY, Urgency.LOW, False),
        (FinalType.HUMAN_REVIEW, Urgency.MEDIUM, True),
    ],
)
def test_urgency_and_human_gate_are_derived_from_the_final_type(
    final_type: FinalType, urgency: Urgency, requires_human: bool
) -> None:
    llm_type = "unclear" if final_type is FinalType.HUMAN_REVIEW else final_type.value
    for use_rules in (True, False):
        decision = apply_rules(make_classification(llm_type), NO_HITS, use_rules=use_rules)
        assert decision.final_type is final_type
        assert decision.urgency is urgency
        assert decision.requires_human is requires_human


def test_ai_only_mode_ignores_flags_keywords_and_confidence() -> None:
    c = make_classification("general_enquiry", flags=["fraud"], confidence=0.1)
    decision = apply_rules(c, CRITICAL_HIT, use_rules=False)
    assert decision.final_type is FinalType.GENERAL_ENQUIRY
    assert decision.rules_fired == ["AI_ONLY"]
    assert decision.requires_human is False


def test_ai_only_mode_still_sends_unclear_and_errors_to_human_review() -> None:
    assert ai_only_decision(make_classification("unclear")).final_type is FinalType.HUMAN_REVIEW
    errored = make_classification("complaint", llm_error=True)
    assert apply_rules(errored, NO_HITS, use_rules=False).final_type is FinalType.HUMAN_REVIEW
