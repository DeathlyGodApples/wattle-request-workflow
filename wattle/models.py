"""Pydantic models and enums: the single source of truth for every data shape."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class RequestType(StrEnum):
    """What the LLM is allowed to return."""

    CRITICAL = "critical"
    HARDSHIP = "hardship"
    COMPLAINT = "complaint"
    SERVICE_REQUEST = "service_request"
    GENERAL_ENQUIRY = "general_enquiry"
    UNCLEAR = "unclear"


class FinalType(StrEnum):
    """What code decides; `unclear` becomes `human_review` because a person must look."""

    CRITICAL = "critical"
    HARDSHIP = "hardship"
    COMPLAINT = "complaint"
    SERVICE_REQUEST = "service_request"
    GENERAL_ENQUIRY = "general_enquiry"
    HUMAN_REVIEW = "human_review"


class Product(StrEnum):
    """Bank product the request is about; drives the hardship clock."""

    HOME_LOAN = "home_loan"
    PERSONAL_LOAN = "personal_loan"
    CAR_LOAN = "car_loan"
    CREDIT_CARD = "credit_card"
    TRANSACTION_ACCOUNT = "transaction_account"
    SAVINGS = "savings"
    UNKNOWN = "unknown"


class RiskFlag(StrEnum):
    """Signals the rules act on regardless of the request type."""

    FRAUD = "fraud"
    SCAM = "scam"
    UNAUTHORISED_TRANSACTION = "unauthorised_transaction"
    SELF_HARM = "self_harm"
    FINANCIAL_ABUSE = "financial_abuse"
    FINANCIAL_DIFFICULTY = "financial_difficulty"
    PROMPT_INJECTION = "prompt_injection"


class Urgency(StrEnum):
    """Derived from the final type in code, never taken from the LLM."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SendMode(StrEnum):
    """Whether a draft may go out without a person looking at it."""

    AUTO = "auto"
    HELD_FOR_APPROVAL = "held_for_approval"
    SUPPRESSED = "suppressed"


class CaseStatus(StrEnum):
    """Where a case sits after its branch has run."""

    AUTO_RESOLVED = "auto_resolved"
    ROUTED = "routed"
    AWAITING_CUSTOMER = "awaiting_customer"
    AWAITING_HUMAN = "awaiting_human"
    APPROVED = "approved"


Channel = Literal["email", "web_form", "chat"]
Audience = Literal["customer", "internal"]


class Request(BaseModel):
    """A customer request as it arrives, with no labels attached."""

    id: str
    channel: Channel
    subject: str
    body: str


class LabelledRequest(Request):
    """A row from the dev/test files: a request plus the expected answers used by eval."""

    expected_type: RequestType
    expected_urgency: Urgency
    human_before_reply: bool
    statutory_21d: bool | None = None
    edge_case: str | None = None

    def to_request(self) -> Request:
        """Drop the labels so the pipeline can never see the expected answer."""
        return Request(id=self.id, channel=self.channel, subject=self.subject, body=self.body)


class FaqEntry(BaseModel):
    """One FAQ item; its answer text is shown to customers verbatim."""

    id: str
    topic: str
    question: str
    answer: str


class Extracted(BaseModel):
    """Facts pulled from the message that fill the draft templates."""

    amount: str | None = None
    date: str | None = None
    account_hint: str | None = None
    desired_outcome: str | None = None
    summary: str


class LLMClassification(BaseModel):
    """Exactly the fields the LLM is asked to fill; this is the response schema it is given."""

    request_type: RequestType
    product: Product
    risk_flags: list[RiskFlag]
    extracted: Extracted
    missing_info: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class Classification(LLMClassification):
    """The LLM's reading plus `llm_error`, which only our code sets; an input to the rules."""

    llm_error: bool = False


class Decision(BaseModel):
    """The final path chosen by code, with the rules that led there."""

    final_type: FinalType
    urgency: Urgency
    rules_fired: list[str]
    requires_human: bool


class Draft(BaseModel):
    """A templated message and whether it may be sent automatically."""

    audience: Audience
    text: str
    send_mode: SendMode


class Step(BaseModel):
    """One logged action in a branch, kept as a plain record for the audit trail."""

    name: str
    detail: dict


class BranchResult(BaseModel):
    """Everything a branch produced for one case."""

    steps: list[Step]
    drafts: list[Draft]
    route_to: str | None
    due_at: datetime | None
    status: CaseStatus


class CaseResult(BaseModel):
    """The full record of one request going through the pipeline."""

    request: Request
    classification: Classification
    decision: Decision
    branch: BranchResult
    keyword_hits: list[str]
