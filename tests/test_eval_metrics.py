"""Tests for eval metrics: the numbers on the results slide must be computed correctly."""

from __future__ import annotations

import pytest

from helpers import NOW, make_classification, make_request, scripted_llm
from wattle import metrics
from wattle.llm import FakeLLM
from wattle.metrics import Prediction
from wattle.models import CaseResult, FaqSelection, FinalType, RequestType
from wattle.pipeline import process


def make_prediction(expected: str, predicted: str, **overrides: object) -> Prediction:
    """Build a Prediction where only the fields a test cares about are stated."""
    fields: dict[str, object] = {
        "id": "REQ-001",
        "expected": FinalType(expected),
        "predicted": FinalType(predicted),
        "ai_type": RequestType.UNCLEAR,
        "confidence": 0.5,
        "requires_human": False,
        "auto_customer_draft": False,
        "human_before_reply": False,
    }
    fields.update(overrides)
    return Prediction.model_validate(fields)


def test_expected_unclear_maps_to_human_review() -> None:
    assert metrics.expected_final_type(RequestType.UNCLEAR) is FinalType.HUMAN_REVIEW
    assert metrics.expected_final_type(RequestType.HARDSHIP) is FinalType.HARDSHIP


def test_prediction_knows_if_it_is_correct() -> None:
    assert make_prediction("critical", "critical").correct is True
    assert make_prediction("critical", "hardship").correct is False


def test_type_accuracy_counts_exact_matches() -> None:
    predictions = [
        make_prediction("critical", "critical"),
        make_prediction("critical", "complaint"),
        make_prediction("hardship", "hardship"),
        make_prediction("human_review", "human_review"),
    ]
    assert metrics.type_accuracy(predictions) == 0.75


def test_type_accuracy_of_nothing_is_zero() -> None:
    assert metrics.type_accuracy([]) == 0.0


def test_per_type_accuracy_groups_by_expected_type() -> None:
    predictions = [
        make_prediction("critical", "critical"),
        make_prediction("critical", "complaint"),
        make_prediction("hardship", "hardship"),
    ]
    result = metrics.per_type_accuracy(predictions)
    assert result[FinalType.CRITICAL] == (1, 2)
    assert result[FinalType.HARDSHIP] == (1, 1)
    assert FinalType.COMPLAINT not in result


def test_confusion_matrix_has_expected_rows_and_predicted_columns() -> None:
    predictions = [
        make_prediction("critical", "critical"),
        make_prediction("critical", "complaint"),
        make_prediction("complaint", "critical"),
    ]
    matrix = metrics.confusion_matrix(predictions)
    assert matrix[FinalType.CRITICAL][FinalType.CRITICAL] == 1
    assert matrix[FinalType.CRITICAL][FinalType.COMPLAINT] == 1
    assert matrix[FinalType.COMPLAINT][FinalType.CRITICAL] == 1
    assert matrix[FinalType.HARDSHIP][FinalType.HARDSHIP] == 0
    assert sum(sum(row.values()) for row in matrix.values()) == 3
    assert set(matrix) == set(FinalType)


def test_caught_counts_recall_for_one_type() -> None:
    predictions = [
        make_prediction("critical", "critical"),
        make_prediction("critical", "complaint"),
        make_prediction("critical", "human_review"),
        make_prediction("complaint", "critical"),
    ]
    assert metrics.caught(predictions, FinalType.CRITICAL) == (1, 3)


def test_false_alarms_are_other_types_predicted_as_the_type() -> None:
    predictions = [
        make_prediction("critical", "critical"),
        make_prediction("complaint", "critical", id="REQ-002"),
        make_prediction("general_enquiry", "critical", id="REQ-003"),
        make_prediction("complaint", "complaint"),
    ]
    assert metrics.false_alarms(predictions, FinalType.CRITICAL) == ["REQ-002", "REQ-003"]


def test_hardship_clock_is_scored_only_on_hardship_requests_that_were_caught() -> None:
    predictions = [
        make_prediction("hardship", "hardship", statutory_21d=True, hardship_days=21),
        make_prediction("hardship", "hardship", statutory_21d=False, hardship_days=10),
        make_prediction("hardship", "hardship", statutory_21d=True, hardship_days=10),
        make_prediction("hardship", "hardship", statutory_21d=False, hardship_days=21),
        make_prediction("hardship", "complaint", statutory_21d=True),  # missed: no clock at all
        make_prediction("complaint", "hardship", hardship_days=21),  # not a hardship label
    ]
    assert metrics.hardship_clock(predictions) == (2, 4)


def test_unsafe_auto_sends_need_both_the_label_and_an_auto_draft() -> None:
    predictions = [
        make_prediction("critical", "complaint", id="REQ-001", human_before_reply=True,
                        auto_customer_draft=True),
        make_prediction("critical", "critical", id="REQ-002", human_before_reply=True),
        make_prediction("complaint", "complaint", id="REQ-003", auto_customer_draft=True),
    ]  # fmt: skip
    assert metrics.unsafe_auto_sends(predictions) == ["REQ-001"]


def test_human_review_rate_is_the_share_needing_a_person() -> None:
    predictions = [
        make_prediction("critical", "critical", requires_human=True),
        make_prediction("complaint", "complaint"),
        make_prediction("complaint", "complaint"),
        make_prediction("hardship", "hardship", requires_human=True),
    ]
    assert metrics.human_review_rate(predictions) == 0.5
    assert metrics.human_review_rate([]) == 0.0


@pytest.mark.parametrize(
    ("request_id", "messy"),
    [
        ("REQ-050", False),
        ("REQ-051", True),
        ("REQ-058", True),
        ("REQ-059", False),
        ("REQ-001", False),
        ("DEV-051", False),
        ("CUSTOM-1", False),
    ],
)
def test_messy_requests_are_req_051_to_058(request_id: str, messy: bool) -> None:
    assert metrics.is_messy(request_id) is messy


def test_clean_and_messy_accuracy_are_reported_separately() -> None:
    predictions = [
        make_prediction("complaint", "complaint", id="REQ-001"),
        make_prediction("complaint", "critical", id="REQ-002"),
        make_prediction("complaint", "complaint", id="REQ-051"),
        make_prediction("complaint", "complaint", id="REQ-052"),
        make_prediction("complaint", "hardship", id="REQ-058"),
    ]
    assert metrics.clean_messy_accuracy(predictions) == ((1, 2), (2, 3))


def test_build_metrics_is_json_ready_and_complete() -> None:
    predictions = [
        make_prediction("critical", "critical", id="REQ-001", requires_human=True,
                        human_before_reply=True),
        make_prediction("complaint", "critical", id="REQ-002", requires_human=True),
        make_prediction("hardship", "hardship", id="REQ-003", requires_human=True,
                        human_before_reply=True, statutory_21d=True, hardship_days=21),
        make_prediction("critical", "complaint", id="REQ-051", human_before_reply=True,
                        auto_customer_draft=True),
    ]  # fmt: skip
    result = metrics.build_metrics(predictions)
    assert result["n"] == 4
    assert result["correct"] == 2
    assert result["type_accuracy"] == 0.5
    assert result["critical"] == {"caught": 1, "total": 2, "false_alarms": ["REQ-002"]}
    assert result["hardship"] == {"caught": 1, "total": 1, "clock_correct": 1, "clock_total": 1}
    assert result["unsafe_auto_sends"] == ["REQ-051"]
    assert result["human_review_rate"] == 0.75
    assert result["clean"] == {"correct": 2, "total": 3}
    assert result["messy"] == {"correct": 0, "total": 1}
    assert result["confusion"]["critical"]["complaint"] == 1
    assert result["misses"] == [
        {"id": "REQ-002", "expected": "complaint", "predicted": "critical"},
        {"id": "REQ-051", "expected": "critical", "predicted": "complaint"},
    ]


def run_case(request_type: str, faq: FaqSelection | None = None, **overrides: object) -> CaseResult:
    """A real pipeline result for a crafted request, so the helpers read real branch output."""
    llm = scripted_llm(make_classification(request_type, **overrides), faq)
    return process(make_request("Customer text"), llm, now=NOW)


def test_auto_customer_draft_is_read_from_the_drafts_the_branch_produced() -> None:
    matched = FaqSelection(faq_id="FAQ-01", confidence=0.9)
    unmatched = FaqSelection(faq_id=None, confidence=0.0)
    assert metrics.has_auto_customer_draft(run_case("complaint")) is True
    assert metrics.has_auto_customer_draft(run_case("service_request")) is True
    assert metrics.has_auto_customer_draft(run_case("general_enquiry", matched)) is True
    assert metrics.has_auto_customer_draft(run_case("general_enquiry", unmatched)) is False
    assert metrics.has_auto_customer_draft(run_case("critical")) is False
    assert metrics.has_auto_customer_draft(run_case("hardship")) is False
    assert metrics.has_auto_customer_draft(run_case("unclear")) is False


def test_an_auto_internal_note_is_not_a_customer_send() -> None:
    result = run_case("complaint")
    internal_only = result.branch.model_copy(
        update={"drafts": [d for d in result.branch.drafts if d.audience == "internal"]}
    )
    assert internal_only.drafts, "the complaint branch should write an internal note"
    assert metrics.has_auto_customer_draft(result.model_copy(update={"branch": internal_only})) is (
        False
    )


def test_hardship_days_come_from_the_clock_the_branch_started() -> None:
    assert metrics.hardship_days_started(run_case("hardship", product="car_loan")) == 21
    assert metrics.hardship_days_started(run_case("hardship", product="savings")) == 10
    assert metrics.hardship_days_started(run_case("complaint")) is None  # a 30-day clock, ignored


def test_llm_failure_reports_a_failed_classification_or_faq_call() -> None:
    assert metrics.llm_failure(run_case("complaint")) is None
    assert "select_faq" in (metrics.llm_failure(run_case("general_enquiry", faq=None)) or "")
    outage = process(make_request("Customer text"), FakeLLM(), now=NOW)
    assert (metrics.llm_failure(outage) or "").startswith("LLM error")
