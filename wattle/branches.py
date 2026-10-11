"""One handler per final type, each returning the steps and drafts for that path.

Steps are plain records for the audit trail. Nothing is sent from here: a draft only says
whether it may go out automatically, must wait for approval, or must not go out at all.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime

from wattle import deadlines, templates
from wattle.faq import select_faq
from wattle.llm import LLMClient
from wattle.models import (
    BranchResult,
    CaseStatus,
    Classification,
    Decision,
    Draft,
    FinalType,
    Product,
    Request,
    RiskFlag,
    SendMode,
    Step,
)

FRAUD_AND_SAFETY_TEAM = "fraud_and_safety_team"
HARDSHIP_TEAM = "hardship_team"
COMPLAINTS_TEAM = "complaints_team"
SERVICE_TEAM = "service_team"
HUMAN_REVIEW_QUEUE = "human_review"

# With these flags a message to the customer could put them at risk, so none goes out.
SUPPRESS_FLAGS = frozenset({RiskFlag.SELF_HARM, RiskFlag.FINANCIAL_ABUSE})

Handler = Callable[[Request, Classification, Decision, datetime], BranchResult]


def _step(name: str, **detail: object) -> Step:
    """One audit-trail record; details are JSON-friendly values so they can be stored."""
    return Step(name=name, detail=detail)


def _customer_draft(text: str, send_mode: SendMode) -> Draft:
    """A message addressed to the customer."""
    return Draft(audience="customer", text=text, send_mode=send_mode)


def handle_critical(
    req: Request, c: Classification, decision: Decision, now: datetime
) -> BranchResult:
    """Stop automation, alert a supervisor and hold (or suppress) the reply."""
    followup = deadlines.critical_followup_due(now)
    suppress = bool(SUPPRESS_FLAGS & set(c.risk_flags))
    send_mode = SendMode.SUPPRESSED if suppress else SendMode.HELD_FOR_APPROVAL
    template, text = _critical_draft(c)
    flags = [flag.value for flag in c.risk_flags]
    steps = [
        _step("pause_automation", reason="critical request: nothing is sent without a person"),
        _step(
            "alert_supervisor",
            summary=c.extracted.summary,
            risk_flags=flags,
            rules_fired=decision.rules_fired,
        ),
    ]
    if suppress:
        reason = "self-harm or financial abuse flagged: contact could put the customer at risk"
        steps.append(_step("suppress_outbound_contact", reason=reason))
    steps.append(_step("draft_reply", template=template, send_mode=send_mode.value))
    steps.append(
        _step(
            "set_followup",
            minutes=deadlines.CRITICAL_FOLLOWUP_MINUTES,
            due_at=followup.isoformat(),
        )
    )
    return BranchResult(
        steps=steps,
        drafts=[_customer_draft(text, send_mode)],
        route_to=FRAUD_AND_SAFETY_TEAM,
        due_at=followup,
        status=CaseStatus.AWAITING_HUMAN,
    )


def _critical_draft(c: Classification) -> tuple[str, str]:
    """Pick the wording for the kind of risk: (template name, text). Self-harm comes first."""
    if RiskFlag.SELF_HARM in c.risk_flags:
        return "self_harm_reply", templates.self_harm_reply()
    if RiskFlag.FINANCIAL_ABUSE in c.risk_flags:
        return "financial_abuse_reply", templates.financial_abuse_reply()
    return "critical_reply", templates.critical_reply()


def handle_hardship(
    req: Request, c: Classification, decision: Decision, now: datetime
) -> BranchResult:
    """Hold a supportive reply, route to the hardship team and start the right clock."""
    days = deadlines.hardship_clock_days(c.product)
    due = deadlines.hardship_due(now, c.product)
    kind = "statutory" if days == deadlines.HARDSHIP_STATUTORY_DAYS else "internal"
    steps = [
        _step(
            "draft_reply",
            template="hardship_reply",
            send_mode=SendMode.HELD_FOR_APPROVAL.value,
        ),
        _step("route", to=HARDSHIP_TEAM),
        _step(
            "start_clock",
            days=days,
            kind=kind,
            product=c.product.value,
            reason=_hardship_clock_reason(c.product),
            due_at=due.isoformat(),
        ),
    ]
    draft = _customer_draft(templates.hardship_reply(due), SendMode.HELD_FOR_APPROVAL)
    return BranchResult(
        steps=steps,
        drafts=[draft],
        route_to=HARDSHIP_TEAM,
        due_at=due,
        status=CaseStatus.AWAITING_HUMAN,
    )


def _hardship_clock_reason(product: Product) -> str:
    """Say in words why this clock was chosen, for the reviewer and the audit trail."""
    if product in deadlines.NON_CREDIT_PRODUCTS:
        return "non-credit product: internal 10-day target"
    if product is Product.UNKNOWN:
        return "product not identified, so the stricter 21-day statutory clock applies"
    return "credit product: 21 days under National Credit Code s72"


def handle_complaint(
    req: Request, c: Classification, decision: Decision, now: datetime
) -> BranchResult:
    """Acknowledge at once, summarise for the complaints team and start the 30-day clock."""
    acknowledge_by = deadlines.complaint_ack_due(now)
    due = deadlines.complaint_resolution_due(now)
    extracted = c.extracted
    steps = [
        _step(
            "draft_acknowledgement",
            send_mode=SendMode.AUTO.value,
            acknowledge_by=acknowledge_by.isoformat(),
        ),
        _step(
            "summarise",
            what=extracted.summary,
            amount=extracted.amount,
            date=extracted.date,
            desired_outcome=extracted.desired_outcome,
        ),
        _step("route", to=COMPLAINTS_TEAM),
        _step(
            "start_clock",
            days=deadlines.COMPLAINT_RESOLUTION_DAYS,
            kind="resolution",
            due_at=due.isoformat(),
        ),
    ]
    acknowledgement = templates.complaint_acknowledgement(extracted, due)
    summary = templates.complaint_internal_summary(req, c, due)
    drafts = [
        _customer_draft(acknowledgement, SendMode.AUTO),
        Draft(audience="internal", text=summary, send_mode=SendMode.AUTO),
    ]
    return BranchResult(
        steps=steps, drafts=drafts, route_to=COMPLAINTS_TEAM, due_at=due, status=CaseStatus.ROUTED
    )


def handle_service_request(
    req: Request, c: Classification, decision: Decision, now: datetime
) -> BranchResult:
    """Ask for what is missing, or route the request and confirm it with a reference."""
    if c.missing_info:
        text = templates.missing_info_request(c.missing_info)
        return BranchResult(
            steps=[_step("request_missing_info", missing=c.missing_info)],
            drafts=[_customer_draft(text, SendMode.AUTO)],
            route_to=SERVICE_TEAM,
            due_at=None,  # the clock starts once the customer has replied
            status=CaseStatus.AWAITING_CUSTOMER,
        )
    due = deadlines.service_request_due(now)
    reference = service_reference(req.id)
    steps = [
        _step("route", to=SERVICE_TEAM),
        _step("confirm", reference=reference),
        _step("start_clock", days=deadlines.SERVICE_REQUEST_DAYS, due_at=due.isoformat()),
    ]
    text = templates.service_confirmation(reference, due)
    return BranchResult(
        steps=steps,
        drafts=[_customer_draft(text, SendMode.AUTO)],
        route_to=SERVICE_TEAM,
        due_at=due,
        status=CaseStatus.ROUTED,
    )


def service_reference(request_id: str) -> str:
    """WB- plus 8 hex digits taken from the request ID, so re-running gives the same one."""
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    return f"WB-{digest[:8].upper()}"


def handle_general_enquiry(
    req: Request, c: Classification, decision: Decision, now: datetime, llm: LLMClient
) -> BranchResult:
    """Answer from the FAQ word for word, or hand off when no entry clearly fits."""
    match = select_faq(req, llm)
    selected = _step(
        "select_faq",
        faq_id=match.faq_id,
        confidence=match.confidence,
        matched=match.matched,
        reason=match.reason,
        llm_error=match.llm_error,
    )
    if match.entry is None:
        return BranchResult(
            steps=[selected, _step("handoff_no_faq_match", reason=match.reason)],
            drafts=[],
            route_to=HUMAN_REVIEW_QUEUE,
            due_at=None,
            status=CaseStatus.AWAITING_HUMAN,
        )
    answer = templates.faq_answer(match.entry)
    return BranchResult(
        steps=[selected, _step("answer_from_faq", faq_id=match.entry.id)],
        drafts=[_customer_draft(answer, SendMode.AUTO)],
        route_to=None,
        due_at=None,
        status=CaseStatus.AUTO_RESOLVED,
    )


def handle_human_review(
    req: Request, c: Classification, decision: Decision, now: datetime
) -> BranchResult:
    """Queue the case for a person with the reasons; nothing is drafted or sent."""
    queued = _step(
        "queue_for_review",
        rules_fired=decision.rules_fired,
        rationale=c.rationale,
        ai_type=c.request_type.value,
        confidence=c.confidence,
    )
    return BranchResult(
        steps=[queued],
        drafts=[],
        route_to=HUMAN_REVIEW_QUEUE,
        due_at=None,
        status=CaseStatus.AWAITING_HUMAN,
    )


# The enquiry branch is not in this table because it alone needs the LLM (to pick an FAQ).
HANDLERS: dict[FinalType, Handler] = {
    FinalType.CRITICAL: handle_critical,
    FinalType.HARDSHIP: handle_hardship,
    FinalType.COMPLAINT: handle_complaint,
    FinalType.SERVICE_REQUEST: handle_service_request,
    FinalType.HUMAN_REVIEW: handle_human_review,
}


def run_branch(
    req: Request, c: Classification, decision: Decision, now: datetime, llm: LLMClient
) -> BranchResult:
    """Send the case down the path the decision chose."""
    if decision.final_type is FinalType.GENERAL_ENQUIRY:
        return handle_general_enquiry(req, c, decision, now, llm)
    return HANDLERS[decision.final_type](req, c, decision, now)
