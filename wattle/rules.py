"""Deterministic rules that turn an AI classification into the final decision.

The AI classifies; this module decides. Rules are checked in a fixed order and the first
one that applies sets the outcome.
"""

from __future__ import annotations

from wattle.config import CONFIDENCE_THRESHOLD
from wattle.keywords import KeywordHits
from wattle.models import Classification, Decision, FinalType, RequestType, RiskFlag, Urgency

URGENCY_BY_TYPE: dict[FinalType, Urgency] = {
    FinalType.CRITICAL: Urgency.CRITICAL,
    FinalType.HARDSHIP: Urgency.HIGH,
    FinalType.COMPLAINT: Urgency.HIGH,
    FinalType.SERVICE_REQUEST: Urgency.MEDIUM,
    FinalType.GENERAL_ENQUIRY: Urgency.LOW,
    FinalType.HUMAN_REVIEW: Urgency.MEDIUM,
}

HUMAN_REQUIRED_TYPES = frozenset({FinalType.CRITICAL, FinalType.HARDSHIP, FinalType.HUMAN_REVIEW})

# Flags that make a request critical whatever type the LLM chose.
CRITICAL_FLAGS = frozenset(
    {
        RiskFlag.FRAUD,
        RiskFlag.SCAM,
        RiskFlag.UNAUTHORISED_TRANSACTION,
        RiskFlag.SELF_HARM,
        RiskFlag.FINANCIAL_ABUSE,
    }
)

# Types that are upgraded to hardship when there is a sign of financial difficulty.
HARDSHIP_UPGRADE_TYPES = frozenset(
    {RequestType.COMPLAINT, RequestType.SERVICE_REQUEST, RequestType.GENERAL_ENQUIRY}
)


def apply_rules(
    c: Classification,
    hits: KeywordHits,
    *,
    use_rules: bool = True,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> Decision:
    """Decide the final type: the first rule that applies wins, and all that apply are logged."""
    if not use_rules:
        return ai_only_decision(c)
    fired = [
        (rule_id, result) for rule_id, result, applies in _checks(c, hits, threshold) if applies
    ]
    if not fired:
        return build_decision(FinalType(c.request_type.value), [])
    return build_decision(fired[0][1], [rule_id for rule_id, _ in fired])


def _checks(
    c: Classification, hits: KeywordHits, threshold: float
) -> list[tuple[str, FinalType, bool]]:
    """The rule table in precedence order: (rule ID, outcome, does it apply?)."""
    flags = set(c.risk_flags)
    is_critical = (
        c.request_type is RequestType.CRITICAL
        or bool(flags & CRITICAL_FLAGS)
        or bool(hits.critical)
    )
    shows_difficulty = RiskFlag.FINANCIAL_DIFFICULTY in flags or bool(hits.hardship)
    is_uncertain = c.confidence < threshold or c.request_type is RequestType.UNCLEAR
    # An LLM outage must not hide a risky message: a critical keyword still makes it critical.
    outage_result = FinalType.CRITICAL if hits.critical else FinalType.HUMAN_REVIEW
    return [
        ("R0", outage_result, c.llm_error),
        ("R1", FinalType.CRITICAL, is_critical),
        ("R2", FinalType.HUMAN_REVIEW, RiskFlag.PROMPT_INJECTION in flags),
        ("R3", FinalType.HARDSHIP, c.request_type in HARDSHIP_UPGRADE_TYPES and shows_difficulty),
        ("R4", FinalType.HUMAN_REVIEW, is_uncertain),
    ]


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
