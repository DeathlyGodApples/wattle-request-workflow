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


def test_the_default_model_is_the_one_the_committed_cache_was_made_with() -> None:
    assert config.DEFAULT_LLM_MODEL == "gemini-3.5-flash-lite"


def test_the_committed_cache_is_inside_the_data_folder_and_holds_only_json() -> None:
    assert config.CACHE_DIR == config.DATA_DIR / "llm_cache"
    files = list(config.CACHE_DIR.iterdir())
    assert files, "the committed LLM cache is missing"
    assert all(path.suffix == ".json" for path in files)
