"""Evaluation metrics, kept apart from eval.py so the numbers can be unit-tested."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from wattle.models import FinalType, RequestType


class Prediction(BaseModel):
    """One evaluated request: what was expected and what the system decided."""

    id: str
    expected: FinalType
    predicted: FinalType
    ai_type: RequestType
    confidence: float

    @property
    def correct(self) -> bool:
        """True when the decided type matches the label."""
        return self.expected is self.predicted


def expected_final_type(expected_type: RequestType) -> FinalType:
    """Labels say `unclear`; the system's name for that outcome is `human_review`."""
    if expected_type is RequestType.UNCLEAR:
        return FinalType.HUMAN_REVIEW
    return FinalType(expected_type.value)


def type_accuracy(predictions: Sequence[Prediction]) -> float:
    """Share of requests whose decided type matches the label."""
    if not predictions:
        return 0.0
    return sum(p.correct for p in predictions) / len(predictions)


def per_type_accuracy(predictions: Sequence[Prediction]) -> dict[FinalType, tuple[int, int]]:
    """(correct, total) per expected type, as counts because the sets are small."""
    result: dict[FinalType, tuple[int, int]] = {}
    for final_type in FinalType:
        of_type = [p for p in predictions if p.expected is final_type]
        if of_type:
            result[final_type] = (sum(p.correct for p in of_type), len(of_type))
    return result
