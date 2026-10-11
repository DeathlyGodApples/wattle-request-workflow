"""Integration tests: crafted requests go through the whole pipeline with a fake LLM."""

from __future__ import annotations

from datetime import timedelta

import pytest

from helpers import NOW, make_classification, make_request, scripted_llm
from wattle import data
from wattle.llm import FakeLLM
from wattle.models import CaseStatus, Classification, FaqSelection, FinalType, SendMode
from wattle.pipeline import process

BSB_ANSWER = FaqSelection(faq_id="FAQ-02", confidence=0.9)

# One crafted request per final type: (name, body, LLM classification, final type, status,
# customer send modes). These are written for the tests, not taken from the data files.
CASES: list[tuple[str, str, Classification, FinalType, CaseStatus, list[SendMode]]] = [
    (
        "critical",
        "There is a $900 payment on my card that I never authorised.",
        make_classification("critical", product="credit_card", risk_flags=["fraud"]),
        FinalType.CRITICAL,
        CaseStatus.AWAITING_HUMAN,
        [SendMode.HELD_FOR_APPROVAL],
    ),
    (
        "hardship",
        "My shifts dried up and I am struggling with my personal loan.",
        make_classification("hardship", product="personal_loan"),
        FinalType.HARDSHIP,
        CaseStatus.AWAITING_HUMAN,
        [SendMode.HELD_FOR_APPROVAL],
    ),
    (
        "complaint",
        "Your branch staff were rude to me on Friday.",
        make_classification("complaint"),
        FinalType.COMPLAINT,
        CaseStatus.ROUTED,
        [SendMode.AUTO],
    ),
    (
        "service_request",
        "Please update my mailing address.",
        make_classification("service_request", missing_info=["new address"]),
        FinalType.SERVICE_REQUEST,
        CaseStatus.AWAITING_CUSTOMER,
        [SendMode.AUTO],
    ),
    (
        "general_enquiry",
        "What is the BSB for the Docklands branch?",
        make_classification("general_enquiry"),
        FinalType.GENERAL_ENQUIRY,
        CaseStatus.AUTO_RESOLVED,
        [SendMode.AUTO],
    ),
    (
        "human_review",
        "asdf",
        make_classification("unclear", confidence=0.2),
        FinalType.HUMAN_REVIEW,
        CaseStatus.AWAITING_HUMAN,
        [],
    ),
]


@pytest.mark.parametrize(
    ("body", "classification", "final_type", "status", "send_modes"),
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_one_request_per_final_type_runs_end_to_end(
    body: str,
    classification: Classification,
    final_type: FinalType,
    status: CaseStatus,
    send_modes: list[SendMode],
) -> None:
    request = make_request(body)
    result = process(request, scripted_llm(classification, BSB_ANSWER), now=NOW)

    assert result.request == request
    assert result.received_at == NOW
    assert result.classification.request_type is classification.request_type
    assert result.decision.final_type is final_type
    assert result.branch.status is status
    customer = [d for d in result.branch.drafts if d.audience == "customer"]
    assert [d.send_mode for d in customer] == send_modes
    assert result.branch.steps


def test_the_keyword_check_turns_a_missed_fraud_into_a_critical_case() -> None:
    request = make_request("My wallet was stolen at the weekend. How do I get a new card?")
    llm = scripted_llm(make_classification("service_request"), BSB_ANSWER)

    with_rules = process(request, llm, now=NOW)
    ai_only = process(request, llm, use_rules=False, now=NOW)

    assert with_rules.decision.final_type is FinalType.CRITICAL
    assert with_rules.decision.rules_fired == ["R1"]
    assert with_rules.keyword_hits == ["stolen"]
    assert [d.send_mode for d in with_rules.branch.drafts] == [SendMode.HELD_FOR_APPROVAL]
    assert with_rules.branch.due_at == NOW + timedelta(minutes=15)

    assert ai_only.decision.final_type is FinalType.SERVICE_REQUEST
    assert [d.send_mode for d in ai_only.branch.drafts] == [SendMode.AUTO]


def test_a_hardship_keyword_upgrades_a_service_request() -> None:
    request = make_request("I lost my job. Can you move my credit card due date?")
    llm = scripted_llm(make_classification("service_request", product="credit_card"))
    result = process(request, llm, now=NOW)
    assert result.decision.final_type is FinalType.HARDSHIP
    assert result.decision.rules_fired == ["R3"]
    assert result.branch.route_to == "hardship_team"
    assert result.branch.due_at == NOW + timedelta(days=21)


def test_an_llm_outage_never_raises_and_goes_to_human_review() -> None:
    result = process(make_request("Where is my new card?"), FakeLLM(), now=NOW)
    assert result.classification.llm_error is True
    assert result.decision.final_type is FinalType.HUMAN_REVIEW
    assert result.decision.rules_fired == ["R0", "R4"]
    assert result.branch.drafts == []
    assert result.branch.status is CaseStatus.AWAITING_HUMAN


def test_an_llm_outage_with_a_critical_keyword_is_still_handled_as_critical() -> None:
    result = process(make_request("Someone has taken over my account."), FakeLLM(), now=NOW)
    assert result.decision.final_type is FinalType.CRITICAL
    assert result.decision.rules_fired[:2] == ["R0", "R1"]
    assert result.branch.route_to == "fraud_and_safety_team"
    assert [d.send_mode for d in result.branch.drafts] == [SendMode.SUPPRESSED]
    assert "suppress_outbound_contact" in [step.name for step in result.branch.steps]


def test_an_faq_outage_hands_the_enquiry_to_a_person_instead_of_raising() -> None:
    llm = scripted_llm(make_classification("general_enquiry"), faq=None)
    result = process(make_request("What are your opening hours?"), llm, now=NOW)
    assert result.decision.final_type is FinalType.GENERAL_ENQUIRY
    assert result.branch.status is CaseStatus.AWAITING_HUMAN
    assert result.branch.drafts == []


def test_only_an_enquiry_makes_a_second_llm_call() -> None:
    complaint_llm = scripted_llm(make_classification("complaint"), BSB_ANSWER)
    enquiry_llm = scripted_llm(make_classification("general_enquiry"), BSB_ANSWER)
    process(make_request("x"), complaint_llm, now=NOW)
    process(make_request("x"), enquiry_llm, now=NOW)
    assert [call.schema_name for call in complaint_llm.calls] == ["LLMClassification"]
    assert [call.schema_name for call in enquiry_llm.calls] == [
        "LLMClassification",
        "FaqSelection",
    ]


def test_the_subject_line_is_scanned_for_keywords_too() -> None:
    request = make_request("Please call me.", subject="Scammed")
    result = process(request, scripted_llm(make_classification("service_request")), now=NOW)
    assert result.keyword_hits == ["scammed"]
    assert result.decision.final_type is FinalType.CRITICAL


def test_received_at_defaults_to_the_current_melbourne_time() -> None:
    result = process(make_request("x"), scripted_llm(make_classification("complaint")))
    assert result.received_at.tzinfo is not None
    assert str(result.received_at.tzinfo) == "Australia/Melbourne"


def test_no_dev_request_raises_or_auto_sends_past_the_human_gate() -> None:
    # No exceptions and no unsafe send, for every dev request crossed with every LLM answer.
    answers = [
        make_classification(request_type, risk_flags=flags)
        for request_type in (
            "critical",
            "hardship",
            "complaint",
            "service_request",
            "general_enquiry",
            "unclear",
        )
        for flags in ([], ["self_harm"], ["prompt_injection"])
    ]
    for row in data.load_dev():
        for answer in answers:
            for use_rules in (True, False):
                llm = scripted_llm(answer, BSB_ANSWER)
                result = process(row.to_request(), llm, use_rules=use_rules, now=NOW)
                if result.decision.requires_human:
                    modes = [d.send_mode for d in result.branch.drafts]
                    assert SendMode.AUTO not in modes
