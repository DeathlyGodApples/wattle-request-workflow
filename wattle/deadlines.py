"""Due-date helpers so regulatory clocks are computed in one auditable place.

M3 holds only the hardship clock length (needed by the eval); due dates are added in M4.
"""

from __future__ import annotations

from wattle.models import Product

HARDSHIP_STATUTORY_DAYS = 21  # National Credit Code s72, credit products
HARDSHIP_INTERNAL_DAYS = 10  # internal target where the statutory clock does not apply

NON_CREDIT_PRODUCTS = frozenset({Product.TRANSACTION_ACCOUNT, Product.SAVINGS})


def hardship_clock_days(product: Product) -> int:
    """Statutory 21 days unless the product is clearly non-credit: when unsure, be stricter."""
    if product in NON_CREDIT_PRODUCTS:
        return HARDSHIP_INTERNAL_DAYS
    return HARDSHIP_STATUTORY_DAYS
