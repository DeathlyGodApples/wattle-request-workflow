"""Deterministic rules that turn an AI classification into the final decision.

M2 holds only the "AI alone" baseline; the R0-R4 rules are added in M3.
"""

from __future__ import annotations

from wattle.models import Classification, Decision, FinalType, RequestType, Urgency

URGENCY_BY_TYPE: dict[FinalType, Urgency] = {
    FinalType.CRITICAL: Urgency.CRITICAL,
    FinalType.HARDSHIP: Urgency.HIGH,
    FinalType.COMPLAINT: Urgency.HIGH,
    FinalType.SERVICE_REQUEST: Urgency.MEDIUM,
    FinalType.GENERAL_ENQUIRY: Urgency.LOW,
    FinalType.HUMAN_REVIEW: Urgency.MEDIUM,
}

HUMAN_REQUIRED_TYPES = frozenset({FinalType.CRITICAL, FinalType.HARDSHIP, FinalType.HUMAN_REVIEW})


def build_decision(final_type: FinalType, rules_fired: list[str]) -> Decision:
    """Derive urgency and the human gate from the final type, so the LLM never sets either."""
    return Decision(
        final_type=final_type,
        urgency=URGENCY_BY_TYPE[final_type],
        rules_fired=rules_fired,
        requires_human=final_type in HUMAN_REQUIRED_TYPES,
    )


def ai_only_decision(c: Classification) -> Decision:
    """The ablation baseline: trust the LLM's type, with no safety rules applied."""
    if c.llm_error or c.request_type is RequestType.UNCLEAR:
        return build_decision(FinalType.HUMAN_REVIEW, ["AI_ONLY"])
    return build_decision(FinalType(c.request_type.value), ["AI_ONLY"])
