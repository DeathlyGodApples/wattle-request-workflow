"""Templated drafts: regulated customer wording must be predictable, not LLM-written.

Customer-facing text is fixed wording. The only things filled in are due dates, a reference,
and the amount and date copied from the customer's own message. Nothing the LLM wrote itself
reaches a customer: its summary appears in internal notes only, and what it says is missing
is translated into our own fixed labels.
"""

from __future__ import annotations

import re
from datetime import datetime

from wattle.models import Classification, Extracted, FaqEntry, Request

SUPPORT_LINE = "13 00 00"  # Wattle Bank's 24/7 line, as given in the FAQ
LIFELINE = "13 11 14"
NATIONAL_DEBT_HELPLINE = "1800 007 007"
RESPECT_LINE = "1800RESPECT (1800 737 732)"
NOT_STATED = "not stated"

# Limit on anything copied into a customer message, so it stays a short fact on one line.
MAX_FILLED_CHARS = 60

# The only details a customer can be asked for: (our label, wording of the LLM's that means it).
# Passwords, PINs and one-time codes are deliberately absent, so they can never be requested.
MISSING_INFO_LABELS: tuple[tuple[str, str], ...] = (
    ("account number", r"account (number|no|num)\b"),
    ("BSB", r"\bbsb\b"),
    ("card last 4 digits", r"card (number|digits)|last (4|four)"),
    ("new address", r"(?<!email )(?<!e-mail )\baddress\b"),
    ("ID document", r"\bid\b|\bidenti|document|licen[cs]e|passport|certificate"),
    ("account name", r"account name|account holder|name on|full name|new name|surname"),
    ("date of birth", r"birth|\bdob\b"),
)
GENERIC_MISSING_INFO = "the details needed to complete your request"


def short(value: str) -> str:
    """One line, at most 60 characters: a filled-in value can never become a paragraph."""
    return " ".join(value.split())[:MAX_FILLED_CHARS]


def format_date(moment: datetime) -> str:
    """Dates as customers read them, e.g. 5 October 2026."""
    return f"{moment.day} {moment:%B %Y}"


def critical_reply() -> str:
    """Fraud or security concern: safe next steps only, and no promise about the outcome."""
    return (
        "Hello,\n\n"
        "Thank you for contacting Wattle Bank. We are treating your message as urgent and our "
        "Fraud and Safety team is reviewing it now.\n\n"
        "What you can do right away:\n"
        "- Lock your card in the app under Cards > Lock card.\n"
        f"- Call our 24/7 line on {SUPPORT_LINE} if you see a transaction you do not recognise."
        "\n\n"
        "We will never ask you for your password, PIN or a one-time code.\n\n"
        "Wattle Bank Fraud and Safety Team"
    )


def self_harm_reply() -> str:
    """The customer may be at risk: put support first and leave the money for later."""
    return (
        "Hello,\n\n"
        "Thank you for telling us how you are feeling. We are concerned about you, and one of "
        "our team would like to talk with you personally.\n\n"
        "If you are in immediate danger, please call 000. You can talk to Lifeline at any "
        f"time on {LIFELINE}.\n\n"
        "Any money matters can wait. We will work through them with you.\n\n"
        "Wattle Bank Customer Care"
    )


def financial_abuse_reply() -> str:
    """Someone else may read this, so it says nothing about what the customer reported."""
    return (
        "Hello,\n\n"
        "Thank you for contacting Wattle Bank. A specialist from our team would like to speak "
        "with you privately about your message.\n\n"
        f"Please call us on {SUPPORT_LINE} at a time, and from a phone, that is safe for you. "
        f"If you are in immediate danger, call 000. {RESPECT_LINE} offers confidential support "
        "at any time.\n\n"
        "Wattle Bank Customer Care"
    )


def hardship_reply(statutory_due: datetime | None) -> str:
    """Supportive, with no promised outcome; a date is stated only for the statutory clock.

    The 10-day target for non-credit products is an internal goal, not a commitment to the
    customer, so for those cases (`statutory_due` is None) the reply names no date.
    """
    if statutory_due is None:
        timing = "They will be in touch with you soon."
    else:
        timing = f"We will respond to you in writing by {format_date(statutory_due)}."
    return (
        "Hello,\n\n"
        "Thank you for letting us know about your situation. We are sorry things are difficult "
        "at the moment, and we want to help.\n\n"
        "Your request has been passed to our Financial Hardship team. A team member will "
        "review it and contact you to talk through your circumstances and the options that "
        f"may be available. {timing}\n\n"
        "You do not need to do anything else right now. If you would like free, independent "
        f"advice, the National Debt Helpline is available on {NATIONAL_DEBT_HELPLINE}.\n\n"
        "Wattle Bank Financial Hardship Team"
    )


def complaint_acknowledgement(extracted: Extracted, resolution_due: datetime) -> str:
    """Confirm the complaint is recorded and say when the response is due."""
    paragraphs = [
        "Hello,",
        "Thank you for contacting Wattle Bank. We are sorry you have had reason to complain.",
        "We have recorded your complaint and passed it to our Complaints team, who will "
        "investigate it.",
        *_noted_details(extracted),
        f"We will give you our response by {format_date(resolution_due)} at the latest.",
        "Wattle Bank Complaints Team",
    ]
    return "\n\n".join(paragraphs)


def _noted_details(extracted: Extracted) -> list[str]:
    """One paragraph with the amount and date the customer gave, or nothing if neither."""
    lines = []
    if extracted.amount:
        lines.append(f"- Amount: {short(extracted.amount)}")
    if extracted.date:
        lines.append(f"- Date: {short(extracted.date)}")
    if not lines:
        return []
    return ["What we have noted:\n" + "\n".join(lines)]


def complaint_internal_summary(req: Request, c: Classification, resolution_due: datetime) -> str:
    """Staff note: what happened, the amount, the date and what the customer wants."""
    extracted = c.extracted
    return (
        f"Complaint {req.id} ({req.channel})\n"
        f"Subject: {req.subject or NOT_STATED}\n"
        f"What: {extracted.summary}\n"
        f"Product: {c.product.value}\n"
        f"Amount: {extracted.amount or NOT_STATED}\n"
        f"Date: {extracted.date or NOT_STATED}\n"
        f"Desired outcome: {extracted.desired_outcome or NOT_STATED}\n"
        f"Respond by: {format_date(resolution_due)}"
    )


def missing_info_labels(missing: list[str]) -> list[str]:
    """Translate the LLM's wording into our fixed labels; anything unrecognised is dropped."""
    return [
        label
        for label, pattern in MISSING_INFO_LABELS
        if any(re.search(pattern, item, re.IGNORECASE) for item in missing)
    ]


def missing_info_request(labels: list[str]) -> str:
    """Ask for the missing details by our own labels, or in general terms if none is known."""
    if labels:
        items = "\n".join(f"- {label}" for label in labels)
        ask = f"we need a little more information from you:\n{items}"
    else:
        ask = f"we need {GENERIC_MISSING_INFO}."
    return (
        "Hello,\n\n"
        f"Thank you for your request. So that we can action it, {ask}\n\n"
        "Please reply with these details. We will never ask for your PIN or password.\n\n"
        "Wattle Bank Service Team"
    )


def service_confirmation(reference: str, due: datetime) -> str:
    """Confirm the request is with the team, with a reference the customer can quote."""
    return (
        "Hello,\n\n"
        "Thank you for your request. It has been passed to our Service team.\n\n"
        f"Your reference is {reference}. We aim to complete it by {format_date(due)} and will "
        "contact you if we need anything further.\n\n"
        "Wattle Bank Service Team"
    )


def faq_answer(entry: FaqEntry) -> str:
    """The FAQ answer exactly as written, plus its source so it can be checked."""
    return f"{entry.answer}\n\nSource: {entry.id}"
