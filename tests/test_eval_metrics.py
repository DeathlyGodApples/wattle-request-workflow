"""Tests for eval metrics: the numbers on the results slide must be computed correctly."""

from __future__ import annotations

from wattle.metrics import Prediction, expected_final_type, per_type_accuracy, type_accuracy
from wattle.models import FinalType, RequestType


def make_prediction(expected: str, predicted: str) -> Prediction:
    """Build a Prediction where only the expected and predicted types matter."""
    return Prediction(
        id="X-1",
        expected=FinalType(expected),
        predicted=FinalType(predicted),
        ai_type=RequestType.UNCLEAR,
        confidence=0.5,
    )


def test_expected_unclear_maps_to_human_review() -> None:
    assert expected_final_type(RequestType.UNCLEAR) is FinalType.HUMAN_REVIEW
    assert expected_final_type(RequestType.HARDSHIP) is FinalType.HARDSHIP


def test_type_accuracy_counts_exact_matches() -> None:
    predictions = [
        make_prediction("critical", "critical"),
        make_prediction("critical", "complaint"),
        make_prediction("hardship", "hardship"),
        make_prediction("human_review", "human_review"),
    ]
    assert type_accuracy(predictions) == 0.75


def test_type_accuracy_of_nothing_is_zero() -> None:
    assert type_accuracy([]) == 0.0


def test_per_type_accuracy_groups_by_expected_type() -> None:
    predictions = [
        make_prediction("critical", "critical"),
        make_prediction("critical", "complaint"),
        make_prediction("hardship", "hardship"),
    ]
    result = per_type_accuracy(predictions)
    assert result[FinalType.CRITICAL] == (1, 2)
    assert result[FinalType.HARDSHIP] == (1, 1)
    assert FinalType.COMPLAINT not in result


def test_prediction_knows_if_it_is_correct() -> None:
    assert make_prediction("critical", "critical").correct is True
    assert make_prediction("critical", "hardship").correct is False
