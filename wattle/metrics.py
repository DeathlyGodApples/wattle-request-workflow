"""Evaluation metrics, kept apart from eval.py so the numbers can be unit-tested."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

from wattle.models import CaseResult, FinalType, RequestType, SendMode

# The deliberately messy test requests (typos, slang, mixed intent): REQ-051 to REQ-058.
_MESSY_ID = re.compile(r"REQ-05[1-8]")


class Prediction(BaseModel):
    """One evaluated request: the labels next to what the system decided."""

    id: str
    expected: FinalType
    predicted: FinalType
    ai_type: RequestType
    confidence: float
    rules_fired: list[str] = []
    keyword_hits: list[str] = []
    requires_human: bool
    auto_customer_draft: bool
    hardship_days: int | None = None  # the clock the system would start, if it chose hardship
    human_before_reply: bool  # label: nothing may be auto-sent for this request
    statutory_21d: bool | None = None  # label: set on hardship rows only

    @property
    def correct(self) -> bool:
        """True when the decided type matches the label."""
        return self.expected is self.predicted


def has_auto_customer_draft(result: CaseResult) -> bool:
    """True if the case produced a customer message that would go out with no person involved."""
    return any(
        draft.audience == "customer" and draft.send_mode is SendMode.AUTO
        for draft in result.branch.drafts
    )


def hardship_days_started(result: CaseResult) -> int | None:
    """The length of the clock the hardship branch actually started, or None for other paths."""
    if result.decision.final_type is not FinalType.HARDSHIP:
        return None
    for step in result.branch.steps:
        if step.name == "start_clock":
            return step.detail["days"]
    return None


def llm_failure(result: CaseResult) -> str | None:
    """Why an LLM call failed in this case, or None; a failed call must not be scored."""
    if result.classification.llm_error:
        return result.classification.rationale
    for step in result.branch.steps:
        if step.detail.get("llm_error"):
            return f"{step.name}: {step.detail.get('reason')}"
    return None


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


def confusion_matrix(predictions: Sequence[Prediction]) -> dict[FinalType, dict[FinalType, int]]:
    """Counts with expected types as rows and predicted types as columns."""
    matrix = {row: dict.fromkeys(FinalType, 0) for row in FinalType}
    for p in predictions:
        matrix[p.expected][p.predicted] += 1
    return matrix


def caught(predictions: Sequence[Prediction], final_type: FinalType) -> tuple[int, int]:
    """Recall as (caught, total): of the requests that are this type, how many were found."""
    of_type = [p for p in predictions if p.expected is final_type]
    return sum(p.correct for p in of_type), len(of_type)


def false_alarms(predictions: Sequence[Prediction], final_type: FinalType) -> list[str]:
    """IDs of requests decided as this type that are labelled as something else."""
    return [p.id for p in predictions if p.predicted is final_type and p.expected is not final_type]


def hardship_clock(predictions: Sequence[Prediction]) -> tuple[int, int]:
    """(correct, total) for 21-day vs 10-day, over hardship requests the system caught."""
    scored = [
        p
        for p in predictions
        if p.expected is FinalType.HARDSHIP and p.correct and p.statutory_21d is not None
    ]
    correct = sum((p.hardship_days == 21) == p.statutory_21d for p in scored)
    return correct, len(scored)


def unsafe_auto_sends(predictions: Sequence[Prediction]) -> list[str]:
    """IDs where a person had to be involved but a customer message would go out anyway."""
    return [p.id for p in predictions if p.human_before_reply and p.auto_customer_draft]


def human_review_rate(predictions: Sequence[Prediction]) -> float:
    """Share of requests that need a person (critical, hardship or human review).

    This is the operational cost of being careful, so it counts every human-gated type,
    not only the ones routed to the human_review queue.
    """
    if not predictions:
        return 0.0
    return sum(p.requires_human for p in predictions) / len(predictions)


def is_messy(request_id: str) -> bool:
    """True for the deliberately messy test requests, REQ-051 to REQ-058."""
    return _MESSY_ID.fullmatch(request_id) is not None


def clean_messy_accuracy(
    predictions: Sequence[Prediction],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """(correct, total) for clean requests, then for messy ones."""
    messy = [p for p in predictions if is_messy(p.id)]
    clean = [p for p in predictions if not is_messy(p.id)]
    return (
        (sum(p.correct for p in clean), len(clean)),
        (sum(p.correct for p in messy), len(messy)),
    )


def build_metrics(predictions: Sequence[Prediction]) -> dict:
    """Every reported number in one JSON-ready dict, so all outputs agree."""
    critical_caught, critical_total = caught(predictions, FinalType.CRITICAL)
    hardship_caught, hardship_total = caught(predictions, FinalType.HARDSHIP)
    clock_correct, clock_total = hardship_clock(predictions)
    clean, messy = clean_messy_accuracy(predictions)
    return {
        "n": len(predictions),
        "correct": sum(p.correct for p in predictions),
        "type_accuracy": round(type_accuracy(predictions), 4),
        "per_type": {
            t.value: {"correct": c, "total": n}
            for t, (c, n) in per_type_accuracy(predictions).items()
        },
        "critical": {
            "caught": critical_caught,
            "total": critical_total,
            "false_alarms": false_alarms(predictions, FinalType.CRITICAL),
        },
        "hardship": {
            "caught": hardship_caught,
            "total": hardship_total,
            "clock_correct": clock_correct,
            "clock_total": clock_total,
        },
        "unsafe_auto_sends": unsafe_auto_sends(predictions),
        "human_review_rate": round(human_review_rate(predictions), 4),
        "clean": {"correct": clean[0], "total": clean[1]},
        "messy": {"correct": messy[0], "total": messy[1]},
        "confusion": {
            row.value: {col.value: count for col, count in cols.items()}
            for row, cols in confusion_matrix(predictions).items()
        },
        "misses": [
            {"id": p.id, "expected": p.expected.value, "predicted": p.predicted.value}
            for p in predictions
            if not p.correct
        ],
    }
