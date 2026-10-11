"""Due-date helpers so regulatory clocks are computed in one auditable place.

Calendar days only (no business-day or public-holiday calendar), in Melbourne time.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from wattle.config import TIMEZONE
from wattle.models import Product

LOCAL_TZ = ZoneInfo(TIMEZONE)

COMPLAINT_ACK_DAYS = 1  # ASIC RG 271: acknowledge promptly
COMPLAINT_RESOLUTION_DAYS = 30  # ASIC RG 271: respond within 30 calendar days
HARDSHIP_STATUTORY_DAYS = 21  # National Credit Code s72, credit products
HARDSHIP_INTERNAL_DAYS = 10  # internal target where the statutory clock does not apply
CRITICAL_FOLLOWUP_MINUTES = 15
SERVICE_REQUEST_DAYS = 5

NON_CREDIT_PRODUCTS = frozenset({Product.TRANSACTION_ACCOUNT, Product.SAVINGS})


def now_local() -> datetime:
    """The current time in Melbourne; the default when no `now` is injected."""
    return datetime.now(LOCAL_TZ)


def to_local(moment: datetime) -> datetime:
    """Put a time on the Melbourne clock; a time with no zone is read as Melbourne time."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=LOCAL_TZ)
    return moment.astimezone(LOCAL_TZ)


def _add_days(received: datetime, days: int) -> datetime:
    """Same clock time N calendar days later, even if daylight saving changes in between."""
    return to_local(received) + timedelta(days=days)


def hardship_clock_days(product: Product) -> int:
    """Statutory 21 days unless the product is clearly non-credit: when unsure, be stricter."""
    if product in NON_CREDIT_PRODUCTS:
        return HARDSHIP_INTERNAL_DAYS
    return HARDSHIP_STATUTORY_DAYS


def complaint_ack_due(received: datetime) -> datetime:
    """When a complaint should be acknowledged by."""
    return _add_days(received, COMPLAINT_ACK_DAYS)


def complaint_resolution_due(received: datetime) -> datetime:
    """When the bank's response to a complaint is due."""
    return _add_days(received, COMPLAINT_RESOLUTION_DAYS)


def hardship_due(received: datetime, product: Product) -> datetime:
    """When the written hardship response is due, on the clock that fits the product."""
    return _add_days(received, hardship_clock_days(product))


def service_request_due(received: datetime) -> datetime:
    """The internal target for completing a service request."""
    return _add_days(received, SERVICE_REQUEST_DAYS)


def critical_followup_due(received: datetime) -> datetime:
    """Fifteen real minutes later; done in UTC so a clock change cannot stretch or skip it."""
    later = to_local(received).astimezone(UTC) + timedelta(minutes=CRITICAL_FOLLOWUP_MINUTES)
    return later.astimezone(LOCAL_TZ)
