"""Tests for branches: each path's steps, drafts and send modes, and the human-gate invariant."""

from __future__ import annotations

import re
from datetime import timedelta

import pytest

from helpers import NOW, make_classification, make_request, scripted_llm
from wattle import branches
from wattle.keywords import KeywordHits
from wattle.llm import FakeLLM
from wattle.models import (
    BranchResult,
    CaseStatus,
    Classification,
    Decision,
    FaqSelection,
    FinalType,
    SendMode,
)
from wattle.rules import apply_rules, build_decision

REQUEST = make_request("Customer text", id="T-7")


def decide(final_type: str) -> Decision:
    """The decision the rules would hand to a branch for this final type."""
    return build_decision(FinalType(final_type), [])


def step_names(result: BranchResult) -> list[str]:
    return [step.name for step in result.steps]


def step_detail(result: BranchResult, name: str) -> dict:
    (step,) = [step for step in result.steps if step.name == name]
    return step.detail


def customer_drafts(result: BranchResult) -> list:
    return [draft for draft in result.drafts if draft.audience == "customer"]


# --- critical ---------------------------------------------------------------------------


def test_critical_holds_the_reply_and_sets_a_fifteen_minute_follow_up() -> None:
    c = make_classification("critical", risk_flags=["fraud"])
    result = branches.handle_critical(REQUEST, c, decide("critical"), NOW)

    assert step_names(result) == [
        "pause_automation",
        "alert_supervisor",
        "draft_reply",
        "set_followup",
    ]
    (draft,) = result.drafts
    assert draft.audience == "customer"
    assert draft.send_mode is SendMode.HELD_FOR_APPROVAL
    assert result.route_to == "fraud_and_safety_team"
    assert result.status is CaseStatus.AWAITING_HUMAN
    assert result.due_at == NOW + timedelta(minutes=15)
    assert step_detail(result, "alert_supervisor")["summary"] == c.extracted.summary
    assert step_detail(result, "alert_supervisor")["risk_flags"] == ["fraud"]


def test_self_harm_suppresses_the_draft_and_offers_lifeline() -> None:
    c = make_classification("critical", risk_flags=["self_harm"])
    result = branches.handle_critical(REQUEST, c, decide("critical"), NOW)

    assert step_names(result) == [
        "pause_automation",
        "alert_supervisor",
        "suppress_outbound_contact",
        "draft_reply",
        "set_followup",
    ]
    (draft,) = result.drafts
    assert draft.send_mode is SendMode.SUPPRESSED
    assert "Lifeline" in draft.text
    assert "13 11 14" in draft.text


def test_financial_abuse_suppresses_the_draft() -> None:
    c = make_classification("critical", risk_flags=["financial_abuse"])
    result = branches.handle_critical(REQUEST, c, decide("critical"), NOW)
    (draft,) = result.drafts
    assert draft.send_mode is SendMode.SUPPRESSED
    assert "suppress_outbound_contact" in step_names(result)
    assert "abuse" not in draft.text.lower()


def test_self_harm_wording_wins_when_both_flags_are_set() -> None:
    c = make_classification("critical", risk_flags=["financial_abuse", "self_harm"])
    (draft,) = branches.handle_critical(REQUEST, c, decide("critical"), NOW).drafts
    assert "Lifeline" in draft.text
    assert draft.send_mode is SendMode.SUPPRESSED


# --- hardship ---------------------------------------------------------------------------


def test_hardship_on_a_credit_product_starts_the_statutory_clock() -> None:
    c = make_classification("hardship", product="home_loan")
    result = branches.handle_hardship(REQUEST, c, decide("hardship"), NOW)

    assert step_names(result) == ["draft_reply", "route", "start_clock"]
    (draft,) = result.drafts
    assert draft.send_mode is SendMode.HELD_FOR_APPROVAL
    assert result.route_to == "hardship_team"
    assert result.status is CaseStatus.AWAITING_HUMAN
    assert result.due_at == NOW + timedelta(days=21)
    clock = step_detail(result, "start_clock")
    assert clock["days"] == 21
    assert clock["kind"] == "statutory"
    assert "s72" in clock["reason"]


def test_hardship_on_a_non_credit_product_starts_the_internal_clock() -> None:
    c = make_classification("hardship", product="transaction_account")
    result = branches.handle_hardship(REQUEST, c, decide("hardship"), NOW)
    clock = step_detail(result, "start_clock")
    assert (clock["days"], clock["kind"]) == (10, "internal")
    assert result.due_at == NOW + timedelta(days=10)


def test_hardship_with_an_unknown_product_gets_the_stricter_clock_and_says_why() -> None:
    c = make_classification("hardship", product="unknown")
    result = branches.handle_hardship(REQUEST, c, decide("hardship"), NOW)
    clock = step_detail(result, "start_clock")
    assert (clock["days"], clock["kind"]) == (21, "statutory")
    assert "stricter" in clock["reason"]


# --- complaint --------------------------------------------------------------------------


def test_complaint_acknowledges_automatically_and_writes_an_internal_summary() -> None:
    c = make_classification(
        "complaint",
        extracted={"summary": "Charged twice.", "amount": "$50", "desired_outcome": "refund"},
    )
    result = branches.handle_complaint(REQUEST, c, decide("complaint"), NOW)

    assert step_names(result) == ["draft_acknowledgement", "summarise", "route", "start_clock"]
    customer, internal = result.drafts
    assert (customer.audience, customer.send_mode) == ("customer", SendMode.AUTO)
    assert internal.audience == "internal"
    assert "Charged twice." in internal.text
    assert "Charged twice." not in customer.text
    assert result.route_to == "complaints_team"
    assert result.status is CaseStatus.ROUTED
    assert result.due_at == NOW + timedelta(days=30)
    assert step_detail(result, "summarise") == {
        "what": "Charged twice.",
        "amount": "$50",
        "date": None,
        "desired_outcome": "refund",
    }
    assert step_detail(result, "start_clock")["days"] == 30
    assert (
        step_detail(result, "draft_acknowledgement")["acknowledge_by"]
        == (NOW + timedelta(days=1)).isoformat()
    )


# --- service request --------------------------------------------------------------------


def test_service_request_with_everything_needed_is_routed_with_a_reference() -> None:
    c = make_classification("service_request")
    result = branches.handle_service_request(REQUEST, c, decide("service_request"), NOW)

    assert step_names(result) == ["route", "confirm", "start_clock"]
    reference = step_detail(result, "confirm")["reference"]
    assert re.fullmatch(r"WB-[0-9A-F]{8}", reference)
    (draft,) = result.drafts
    assert draft.send_mode is SendMode.AUTO
    assert reference in draft.text
    assert result.route_to == "service_team"
    assert result.status is CaseStatus.ROUTED
    assert result.due_at == NOW + timedelta(days=5)


def test_service_request_with_missing_info_asks_the_customer_and_starts_no_clock() -> None:
    c = make_classification("service_request", missing_info=["account number"])
    result = branches.handle_service_request(REQUEST, c, decide("service_request"), NOW)

    assert step_names(result) == ["request_missing_info"]
    (draft,) = result.drafts
    assert draft.send_mode is SendMode.AUTO
    assert "- account number" in draft.text
    assert result.status is CaseStatus.AWAITING_CUSTOMER
    assert result.due_at is None
    assert result.route_to == "service_team"


def test_the_same_request_always_gets_the_same_reference() -> None:
    assert branches.service_reference("T-7") == branches.service_reference("T-7")
    assert branches.service_reference("T-7") != branches.service_reference("T-8")


# --- general enquiry --------------------------------------------------------------------


def test_enquiry_with_a_faq_match_answers_verbatim_and_resolves() -> None:
    llm = FakeLLM(default=FaqSelection(faq_id="FAQ-02", confidence=0.9))
    c = make_classification("general_enquiry")
    result = branches.handle_general_enquiry(REQUEST, c, decide("general_enquiry"), NOW, llm)

    assert step_names(result) == ["select_faq", "answer_from_faq"]
    (draft,) = result.drafts
    assert draft.send_mode is SendMode.AUTO
    assert draft.text == (
        "The BSB for our Docklands branch (800 Bourke Street) is 063-522.\n\nSource: FAQ-02"
    )
    assert result.status is CaseStatus.AUTO_RESOLVED
    assert result.route_to is None
    assert step_detail(result, "select_faq")["matched"] is True


@pytest.mark.parametrize(
    "selection",
    [
        FaqSelection(faq_id=None, confidence=0.0),
        FaqSelection(faq_id="FAQ-02", confidence=0.4),
        None,  # the FAQ call itself fails
    ],
    ids=["nothing selected", "low confidence", "llm failure"],
)
def test_enquiry_without_a_faq_match_hands_off_and_sends_nothing(
    selection: FaqSelection | None,
) -> None:
    c = make_classification("general_enquiry")
    llm = scripted_llm(c, selection)
    result = branches.handle_general_enquiry(REQUEST, c, decide("general_enquiry"), NOW, llm)

    assert step_names(result) == ["select_faq", "handoff_no_faq_match"]
    assert result.drafts == []
    assert result.status is CaseStatus.AWAITING_HUMAN
    assert result.route_to == "human_review"
    assert step_detail(result, "select_faq")["llm_error"] is (selection is None)


# --- human review -----------------------------------------------------------------------


def test_human_review_queues_the_case_with_the_reasons_and_drafts_nothing() -> None:
    c = make_classification("complaint", confidence=0.4, rationale="Could be two things.")
    decision = build_decision(FinalType.HUMAN_REVIEW, ["R4"])
    result = branches.handle_human_review(REQUEST, c, decision, NOW)

    assert step_names(result) == ["queue_for_review"]
    assert step_detail(result, "queue_for_review") == {
        "rules_fired": ["R4"],
        "rationale": "Could be two things.",
        "ai_type": "complaint",
        "confidence": 0.4,
    }
    assert result.drafts == []
    assert result.route_to == "human_review"
    assert result.status is CaseStatus.AWAITING_HUMAN
    assert result.due_at is None


# --- dispatch and the invariant ---------------------------------------------------------


def test_every_final_type_has_a_branch() -> None:
    llm = FakeLLM(default=FaqSelection(faq_id="FAQ-01", confidence=0.9))
    for final_type in FinalType:
        c = make_classification()
        result = branches.run_branch(REQUEST, c, decide(final_type.value), NOW, llm)
        assert result.steps, f"{final_type} produced no steps"


def all_classifications() -> list[Classification]:
    """Every LLM type crossed with the inputs that change what a branch drafts."""
    flag_sets = [[], ["self_harm"], ["financial_abuse"], ["financial_difficulty"], ["fraud"]]
    return [
        make_classification(
            request_type,
            risk_flags=flags,
            missing_info=missing,
            confidence=confidence,
        )
        for request_type in (
            "critical",
            "hardship",
            "complaint",
            "service_request",
            "general_enquiry",
            "unclear",
        )
        for flags in flag_sets
        for missing in ([], ["account number"])
        for confidence in (0.9, 0.3)
    ]


@pytest.mark.parametrize("use_rules", [True, False], ids=["with rules", "AI only"])
@pytest.mark.parametrize(
    "hits",
    [KeywordHits(), KeywordHits(critical=["stolen"]), KeywordHits(hardship=["lost my job"])],
    ids=["no keywords", "critical keyword", "hardship keyword"],
)
def test_invariant_requires_human_means_no_auto_draft(use_rules: bool, hits: KeywordHits) -> None:
    llm = FakeLLM(default=FaqSelection(faq_id="FAQ-01", confidence=0.9))
    for c in all_classifications():
        decision = apply_rules(c, hits, use_rules=use_rules)
        result = branches.run_branch(REQUEST, c, decision, NOW, llm)
        if decision.requires_human:
            auto = [d for d in result.drafts if d.send_mode is SendMode.AUTO]
            assert auto == [], f"{decision.final_type} sent an auto draft for {c}"
            assert result.status is CaseStatus.AWAITING_HUMAN


def test_every_step_detail_can_be_stored_as_json() -> None:
    import json

    llm = FakeLLM(default=FaqSelection(faq_id="FAQ-01", confidence=0.9))
    for c in all_classifications():
        for final_type in FinalType:
            result = branches.run_branch(REQUEST, c, decide(final_type.value), NOW, llm)
            for step in result.steps:
                json.dumps(step.detail)
