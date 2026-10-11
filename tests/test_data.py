"""Tests for data loading: the files parse fully and bad rows fail loudly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wattle import data
from wattle.models import FaqEntry, LabelledRequest

VALID_ROW = {
    "id": "X-1",
    "channel": "chat",
    "subject": "",
    "body": "hi",
    "expected_type": "unclear",
    "expected_urgency": "medium",
    "human_before_reply": True,
    "statutory_21d": None,
    "edge_case": None,
}


def test_dev_set_has_20_labelled_requests() -> None:
    rows = data.load_dev()
    assert len(rows) == 20
    assert all(isinstance(r, LabelledRequest) for r in rows)


def test_test_set_has_58_labelled_requests() -> None:
    assert len(data.load_test()) == 58


def test_dev_and_test_ids_do_not_overlap() -> None:
    dev_ids = {r.id for r in data.load_dev()}
    test_ids = {r.id for r in data.load_test()}
    assert dev_ids.isdisjoint(test_ids)


def test_faq_has_15_entries() -> None:
    entries = data.load_faq()
    assert len(entries) == 15
    assert all(isinstance(e, FaqEntry) for e in entries)
    assert len({e.id for e in entries}) == 15


def test_load_requests_skips_blank_lines(tmp_path: Path) -> None:
    line = json.dumps(VALID_ROW)
    path = tmp_path / "rows.jsonl"
    path.write_text(f"{line}\n\n{line}\n", encoding="utf-8")
    assert len(data.load_requests(path)) == 2


def test_load_requests_names_the_bad_line(tmp_path: Path) -> None:
    bad_row = {**VALID_ROW, "channel": "fax"}
    path = tmp_path / "rows.jsonl"
    path.write_text(f"{json.dumps(VALID_ROW)}\n{json.dumps(bad_row)}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rows.jsonl line 2"):
        data.load_requests(path)
