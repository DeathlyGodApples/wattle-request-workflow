"""Tests for config: the scaffold imports cleanly and points at the real data folder."""

from __future__ import annotations

from wattle import config


def test_confidence_threshold_is_a_probability() -> None:
    assert isinstance(config.CONFIDENCE_THRESHOLD, float)
    assert 0.0 <= config.CONFIDENCE_THRESHOLD <= 1.0


def test_llm_model_has_a_default() -> None:
    assert config.LLM_MODEL


def test_data_dir_contains_dev_set() -> None:
    assert (config.DATA_DIR / "dev_requests.jsonl").is_file()
