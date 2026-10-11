"""Tests for eval metrics: the numbers on the results slide must be computed correctly."""

from __future__ import annotations

import pytest

from wattle import metrics
from wattle.metrics import Prediction
from wattle.models import FinalType, RequestType


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
