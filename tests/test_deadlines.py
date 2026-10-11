"""Tests for deadlines. M3 covers the hardship clock length; due dates are added in M4."""

from __future__ import annotations

import pytest

from wattle.deadlines import hardship_clock_days
from wattle.models import Product


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
