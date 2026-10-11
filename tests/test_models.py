"""Tests for models: closed sets reject unknown values and defaults are safe."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wattle.models import (
    Classification,
    Decision,
    Draft,
    Extracted,
    FinalType,
    LabelledRequest,
    Product,
    Request,
    RequestType,
    RiskFlag,
    SendMode,
    Urgency,
)


def make_classification(**overrides: object) -> Classification:
    """Build a valid Classification so each test only states what it changes."""
    fields: dict[str, object] = {
        "request_type": "complaint",
        "product": "credit_card",
        "risk_flags": [],
        "extracted": {"summary": "Customer is unhappy about a fee."},
        "missing_info": [],
        "confidence": 0.9,
        "rationale": "Explicit complaint about a fee.",
    }
    fields.update(overrides)
    return Classification.model_validate(fields)


def test_classification_parses_strings_into_enums() -> None:
    c = make_classification(risk_flags=["fraud", "scam"])
    assert c.request_type is RequestType.COMPLAINT
    assert c.product is Product.CREDIT_CARD
    assert c.risk_flags == [RiskFlag.FRAUD, RiskFlag.SCAM]
    assert c.llm_error is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"request_type": "human_review"},  # a FinalType, not something the LLM may return
        {"product": "mortgage"},
        {"risk_flags": ["angry"]},
        {"confidence": 1.2},
        {"confidence": -0.1},
    ],
)
def test_classification_rejects_values_outside_closed_sets(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_classification(**overrides)


def test_extracted_only_requires_a_summary() -> None:
    e = Extracted(summary="Wants a card replaced.")
    assert (e.amount, e.date, e.account_hint, e.desired_outcome) == (None, None, None, None)


def test_extracted_requires_summary() -> None:
    with pytest.raises(ValidationError):
        Extracted.model_validate({"amount": "$50"})


def test_request_types_and_final_types_differ_only_in_the_fallback() -> None:
    llm_values = {t.value for t in RequestType}
    final_values = {t.value for t in FinalType}
    assert llm_values - final_values == {"unclear"}
    assert final_values - llm_values == {"human_review"}


def test_request_rejects_unknown_channel() -> None:
    with pytest.raises(ValidationError):
        Request(id="X-1", channel="fax", subject="Hi", body="Hello")


def test_decision_and_draft_accept_valid_values() -> None:
    d = Decision(final_type="critical", urgency="critical", rules_fired=["R1"], requires_human=True)
    assert d.final_type is FinalType.CRITICAL
    assert d.urgency is Urgency.CRITICAL
    draft = Draft(audience="customer", text="Hello", send_mode="held_for_approval")
    assert draft.send_mode is SendMode.HELD_FOR_APPROVAL


def test_draft_rejects_unknown_audience() -> None:
    with pytest.raises(ValidationError):
        Draft(audience="regulator", text="Hello", send_mode="auto")


def test_labelled_request_converts_to_a_request_without_labels() -> None:
    row = LabelledRequest(
        id="X-1",
        channel="email",
        subject="Fee",
        body="Please refund my fee.",
        expected_type="complaint",
        expected_urgency="high",
        human_before_reply=False,
    )
    req = row.to_request()
    assert type(req) is Request
    assert req.model_dump(mode="json") == {
        "id": "X-1",
        "channel": "email",
        "subject": "Fee",
        "body": "Please refund my fee.",
    }
    assert row.statutory_21d is None
    assert row.edge_case is None
