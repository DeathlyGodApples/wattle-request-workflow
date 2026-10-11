"""Tests for deadlines: calendar days in Melbourne time, and the stricter clock when unsure."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from wattle import deadlines
from wattle.deadlines import hardship_clock_days
from wattle.models import Product

MELBOURNE = ZoneInfo("Australia/Melbourne")
RECEIVED = datetime(2026, 10, 12, 9, 30, tzinfo=MELBOURNE)


@pytest.mark.parametrize(
    ("product", "days"),
    [
        (Product.HOME_LOAN, 21),
        (Product.PERSONAL_LOAN, 21),
        (Product.CAR_LOAN, 21),
        (Product.CREDIT_CARD, 21),
        (Product.UNKNOWN, 21),  # when unsure, apply the stricter statutory clock
        (Product.TRANSACTION_ACCOUNT, 10),
        (Product.SAVINGS, 10),
    ],
)
def test_hardship_clock_is_statutory_unless_clearly_non_credit(product: Product, days: int) -> None:
    assert hardship_clock_days(product) == days


def test_every_product_has_a_clock() -> None:
    for product in Product:
        assert hardship_clock_days(product) in (10, 21)


def test_complaint_is_acknowledged_in_one_day_and_resolved_in_thirty() -> None:
    assert deadlines.complaint_ack_due(RECEIVED) == datetime(2026, 10, 13, 9, 30, tzinfo=MELBOURNE)
    assert deadlines.complaint_resolution_due(RECEIVED) == datetime(
        2026, 11, 11, 9, 30, tzinfo=MELBOURNE
    )


def test_hardship_due_date_follows_the_product() -> None:
    assert deadlines.hardship_due(RECEIVED, Product.HOME_LOAN) == RECEIVED + timedelta(days=21)
    assert deadlines.hardship_due(RECEIVED, Product.UNKNOWN) == RECEIVED + timedelta(days=21)
    assert deadlines.hardship_due(RECEIVED, Product.SAVINGS) == RECEIVED + timedelta(days=10)


def test_critical_follow_up_is_fifteen_minutes_later() -> None:
    assert deadlines.critical_followup_due(RECEIVED) == RECEIVED + timedelta(minutes=15)


def test_service_request_is_due_in_five_days() -> None:
    assert deadlines.service_request_due(RECEIVED) == datetime(
        2026, 10, 17, 9, 30, tzinfo=MELBOURNE
    )


def test_calendar_days_keep_the_local_time_across_a_daylight_saving_change() -> None:
    # Daylight saving ends on 4 April 2027, inside this 30-day window.
    received = datetime(2027, 3, 20, 9, 30, tzinfo=MELBOURNE)
    due = deadlines.complaint_resolution_due(received)
    assert (due.year, due.month, due.day, due.hour, due.minute) == (2027, 4, 19, 9, 30)
    assert received.utcoffset() == timedelta(hours=11)
    assert due.utcoffset() == timedelta(hours=10)


def test_fifteen_minutes_is_real_elapsed_time_across_a_daylight_saving_change() -> None:
    # Clocks jump from 02:00 to 03:00 on 4 October 2026.
    received = datetime(2026, 10, 4, 1, 50, tzinfo=MELBOURNE)
    due = deadlines.critical_followup_due(received)
    assert (due.hour, due.minute) == (3, 5)
    assert due.astimezone(UTC) - received.astimezone(UTC) == timedelta(minutes=15)


def test_times_from_another_zone_are_converted_to_melbourne() -> None:
    received_utc = datetime(2026, 10, 11, 22, 30, tzinfo=UTC)  # 09:30 on the 12th in Melbourne
    due = deadlines.service_request_due(received_utc)
    assert due == datetime(2026, 10, 17, 9, 30, tzinfo=MELBOURNE)
    assert due.utcoffset() == timedelta(hours=11)


def test_a_time_without_a_zone_is_read_as_melbourne_time() -> None:
    assert deadlines.to_local(datetime(2026, 10, 12, 9, 30)) == RECEIVED


def test_now_local_is_in_melbourne_time() -> None:
    assert deadlines.now_local().tzinfo == MELBOURNE


def test_the_follow_up_also_reads_a_time_without_a_zone_as_melbourne_time() -> None:
    naive = datetime(2026, 10, 12, 9, 30)
    assert deadlines.critical_followup_due(naive) == RECEIVED + timedelta(minutes=15)
