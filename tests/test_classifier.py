"""Tests for the classifier: it returns a Classification and never lets an LLM failure escape."""

from __future__ import annotations

from pydantic import BaseModel

from wattle.classifier import classify
from wattle.llm import FakeLLM, LLMError
from wattle.models import (
    Classification,
    Extracted,
    LLMClassification,
    Product,
    Request,
    RequestType,
    RiskFlag,
)

REQUEST = Request(id="X-1", channel="email", subject="Fee", body="Please refund my $34 fee.")

LLM_OUTPUT = LLMClassification(
    request_type=RequestType.COMPLAINT,
    product=Product.CREDIT_CARD,
    risk_flags=[],
    extracted=Extracted(amount="$34", summary="Customer wants a $34 fee refunded."),
    missing_info=[],
    confidence=0.92,
    rationale="Explicit request to refund a fee the customer disputes.",
)


def test_classify_returns_the_llm_output_as_a_classification() -> None:
    result = classify(REQUEST, FakeLLM(default=LLM_OUTPUT))
    assert type(result) is Classification
    assert result.request_type is RequestType.COMPLAINT
    assert result.extracted.amount == "$34"
    assert result.confidence == 0.92
    assert result.llm_error is False


def test_classify_sends_the_fenced_message_and_the_llm_schema() -> None:
    llm = FakeLLM(default=LLM_OUTPUT)
    classify(REQUEST, llm)
    (call,) = llm.calls
    assert "<customer_message>" in call.user
    assert "Please refund my $34 fee." in call.user
    assert call.schema_name == "LLMClassification"


def test_the_llm_schema_has_no_llm_error_field() -> None:
    assert "llm_error" not in LLMClassification.model_fields
    assert "llm_error" in Classification.model_fields


def test_classify_retries_once_after_a_failure() -> None:
    attempts: list[int] = []

    def fail_then_succeed(system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        attempts.append(1)
        if len(attempts) == 1:
            raise LLMError("temporary")
        return LLM_OUTPUT

    result = classify(REQUEST, FakeLLM(fail_then_succeed))
    assert len(attempts) == 2
    assert result.llm_error is False
    assert result.request_type is RequestType.COMPLAINT


def test_classify_returns_an_error_classification_after_two_failures() -> None:
    llm = FakeLLM({"refund": LLMError("quota exceeded")})
    result = classify(REQUEST, llm)
    assert len(llm.calls) == 2
    assert result.llm_error is True
    assert result.request_type is RequestType.UNCLEAR
    assert result.product is Product.UNKNOWN
    assert result.confidence == 0.0
    assert result.risk_flags == []
    assert result.rationale.startswith("LLM error:")
    assert "quota exceeded" in result.rationale


def test_classify_treats_a_wrong_shaped_response_as_a_failure() -> None:
    def wrong_shape(system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        return Extracted(summary="not a classification")

    result = classify(REQUEST, FakeLLM(wrong_shape))
    assert result.llm_error is True


def test_classify_keeps_risk_flags() -> None:
    flagged = LLM_OUTPUT.model_copy(update={"risk_flags": [RiskFlag.SCAM]})
    assert classify(REQUEST, FakeLLM(default=flagged)).risk_flags == [RiskFlag.SCAM]
