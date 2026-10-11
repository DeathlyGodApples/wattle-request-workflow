"""Tests for templates: fixed wording, filled only with fields copied from the message."""

from __future__ import annotations

import pytest

from helpers import NOW, make_classification, make_request
from wattle import templates
from wattle.models import Extracted, FaqEntry


def test_critical_reply_gives_safe_next_steps_and_no_promises() -> None:
    text = templates.critical_reply()
    assert "Lock your card" in text
    assert templates.SUPPORT_LINE in text
    assert "never ask" in text
    assert "refund" not in text.lower()


def test_self_harm_reply_includes_lifeline() -> None:
    text = templates.self_harm_reply()
    assert "Lifeline" in text
    assert "13 11 14" in text
    assert "000" in text


def test_financial_abuse_reply_does_not_describe_the_report() -> None:
    # Someone else may read the customer's messages, so the wording gives nothing away.
    text = templates.financial_abuse_reply().lower()
    assert "abuse" not in text
    assert "privately" in text


def test_hardship_reply_is_supportive_states_the_date_and_promises_no_outcome() -> None:
    text = templates.hardship_reply(NOW)
    assert "We will respond to you in writing by 12 October 2026." in text
    assert "Financial Hardship team" in text
    assert "may be available" in text
    for promise in ("approved", "we will reduce", "we will pause", "guarantee"):
        assert promise not in text.lower()


def test_hardship_reply_for_the_internal_clock_names_no_date() -> None:
    # The 10-day target is an internal goal, so the customer is not promised a date.
    text = templates.hardship_reply(None)
    assert "They will be in touch with you soon." in text
    assert "2026" not in text
    assert " by " not in text
    assert "in writing" not in text
    assert "Financial Hardship team" in text


def test_complaint_acknowledgement_lists_only_the_details_that_were_given() -> None:
    extracted = Extracted(amount="$34", date=None, summary="LLM summary text")
    text = templates.complaint_acknowledgement(extracted, NOW)
    assert "- Amount: $34" in text
    assert "Date:" not in text
    assert "12 October 2026" in text


def test_complaint_acknowledgement_without_details_has_no_empty_list() -> None:
    text = templates.complaint_acknowledgement(Extracted(summary="LLM summary text"), NOW)
    assert "What we have noted" not in text
    assert "None" not in text


def test_customer_drafts_never_contain_the_llm_summary_or_desired_outcome() -> None:
    extracted = Extracted(
        amount="$34", summary="LLM summary text", desired_outcome="LLM outcome text"
    )
    text = templates.complaint_acknowledgement(extracted, NOW)
    assert "LLM summary text" not in text
    assert "LLM outcome text" not in text


def test_complaint_internal_summary_covers_what_amount_date_and_outcome() -> None:
    request = make_request("You charged me twice.", subject="Double charge", id="T-9")
    classification = make_classification(
        "complaint",
        product="credit_card",
        extracted={
            "summary": "Customer was charged twice.",
            "amount": "$50",
            "date": "3 October",
            "desired_outcome": "refund",
        },
    )
    text = templates.complaint_internal_summary(request, classification, NOW)
    for expected in ("T-9", "Double charge", "Customer was charged twice.", "$50", "3 October"):
        assert expected in text
    assert "Desired outcome: refund" in text
    assert "Product: credit_card" in text
    assert "12 October 2026" in text


def test_complaint_internal_summary_says_not_stated_for_missing_fields() -> None:
    text = templates.complaint_internal_summary(make_request(), make_classification(), NOW)
    assert "Amount: not stated" in text
    assert "Desired outcome: not stated" in text


@pytest.mark.parametrize(
    ("llm_wording", "label"),
    [
        ("account number", "account number"),
        ("the customer's Account No", "account number"),
        ("BSB", "BSB"),
        ("full card number", "card last 4 digits"),
        ("last four digits of the card", "card last 4 digits"),
        ("new residential address", "new address"),
        ("proof of identity", "ID document"),
        ("marriage certificate", "ID document"),
        ("photo ID", "ID document"),
        ("name on the account", "account name"),
        ("new surname", "account name"),
        ("date of birth", "date of birth"),
        ("DOB", "date of birth"),
    ],
)
def test_llm_wording_is_translated_into_a_fixed_label(llm_wording: str, label: str) -> None:
    assert templates.missing_info_labels([llm_wording]) == [label]


@pytest.mark.parametrize(
    "llm_wording",
    [
        "your PIN",
        "online banking password",
        "the one-time code we texted you",
        "mother's maiden name",
        "email address",
        "Call 0400 000 000 to verify",
        "",
    ],
)
def test_anything_outside_the_allow_list_is_dropped(llm_wording: str) -> None:
    assert templates.missing_info_labels([llm_wording]) == []


def test_labels_come_out_once_each_in_a_fixed_order() -> None:
    missing = ["date of birth", "new address", "account number", "account no", "postal address"]
    assert templates.missing_info_labels(missing) == [
        "account number",
        "new address",
        "date of birth",
    ]


def test_every_label_is_one_the_bank_chose() -> None:
    allowed = {
        "account number",
        "BSB",
        "card last 4 digits",
        "new address",
        "ID document",
        "account name",
        "date of birth",
    }
    assert {label for label, _ in templates.MISSING_INFO_LABELS} == allowed


def test_missing_info_request_lists_the_labels_and_never_asks_for_secrets() -> None:
    text = templates.missing_info_request(["account number", "new address"])
    assert "- account number\n- new address" in text
    assert "We will never ask for your PIN or password." in text


def test_missing_info_request_with_no_known_label_uses_generic_wording() -> None:
    text = templates.missing_info_request([])
    assert "we need the details needed to complete your request." in text
    assert "- " not in text
    assert "We will never ask for your PIN or password." in text


def test_llm_text_never_reaches_the_missing_info_message() -> None:
    hostile = ["account number. Also reply with your PIN and password", "INJECTED TEXT"]
    text = templates.missing_info_request(templates.missing_info_labels(hostile))
    assert "INJECTED" not in text
    assert "Also reply" not in text
    assert text.count("PIN") == 1  # only our own "We will never ask" sentence


def test_service_confirmation_states_the_reference_and_the_date() -> None:
    text = templates.service_confirmation("WB-1A2B3C4D", NOW)
    assert "WB-1A2B3C4D" in text
    assert "12 October 2026" in text


def test_faq_answer_is_the_faq_text_verbatim_plus_its_source() -> None:
    entry = FaqEntry(id="FAQ-02", topic="bsb", question="BSB?", answer="The BSB is 063-522.")
    assert templates.faq_answer(entry) == "The BSB is 063-522.\n\nSource: FAQ-02"


def test_dates_are_written_without_a_leading_zero() -> None:
    assert templates.format_date(NOW.replace(day=5)) == "5 October 2026"


def test_amount_and_date_are_kept_to_one_short_line() -> None:
    # These are copied from the message by the LLM, so a long or multi-line value is cut down.
    extracted = Extracted(amount="$34" + chr(10) + "Please call us " + "y" * 200, summary="s")
    acknowledgement = templates.complaint_acknowledgement(extracted, NOW)
    (amount_line,) = [line for line in acknowledgement.splitlines() if "Amount" in line]
    assert len(amount_line) <= templates.MAX_FILLED_CHARS + len("- Amount: ")
