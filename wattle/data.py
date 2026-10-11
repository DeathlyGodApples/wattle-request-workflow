"""Loaders for the labelled request files and the FAQ, so file formats live in one place."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from wattle.config import DATA_DIR
from wattle.models import FaqEntry, LabelledRequest

DEV_PATH = DATA_DIR / "dev_requests.jsonl"
TEST_PATH = DATA_DIR / "test_requests.jsonl"
FAQ_PATH = DATA_DIR / "faq.json"


def load_requests(path: Path) -> list[LabelledRequest]:
    """Validate every row up front so a bad label fails here, not halfway through an eval."""
    rows: list[LabelledRequest] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(LabelledRequest.model_validate_json(line))
            except ValidationError as exc:
                raise ValueError(f"{path.name} line {line_no}: {exc}") from exc
    return rows


def load_dev() -> list[LabelledRequest]:
    """Dev set: the only data allowed in prompts, few-shot examples and tuning."""
    return load_requests(DEV_PATH)


def load_test() -> list[LabelledRequest]:
    """Held-out test set: for evaluation only, never for prompts or fixtures."""
    return load_requests(TEST_PATH)


def load_faq(path: Path = FAQ_PATH) -> list[FaqEntry]:
    """FAQ entries whose answers are shown verbatim, so they are validated on load."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [FaqEntry.model_validate(entry) for entry in raw["entries"]]
