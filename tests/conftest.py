"""Shared pytest setup: keeps the suite offline unless a live run is asked for."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked `live` unless RUN_LIVE=1, so unit runs never call a real LLM."""
    if os.getenv("RUN_LIVE") == "1":
        return
    skip_live = pytest.mark.skip(reason="live LLM test; set RUN_LIVE=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
