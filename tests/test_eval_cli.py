"""Tests for the eval script's plumbing, using FakeLLM so no real call is made."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import eval as eval_cli
from wattle import data
from wattle.llm import FakeLLM, LLMError
from wattle.models import Extracted, LLMClassification, Product, RequestType
from wattle.prompts import FEW_SHOT_IDS, classifier_user_prompt


def always(request_type: RequestType) -> FakeLLM:
    """A fake that gives every request the same type, so the expected score is easy to state."""
    return FakeLLM(
        default=LLMClassification(
            request_type=request_type,
            product=Product.UNKNOWN,
            risk_flags=[],
            extracted=Extracted(summary="summary"),
            missing_info=[],
            confidence=0.9,
            rationale="rationale",
        )
    )


def test_dev_excludes_the_few_shot_examples_by_default() -> None:
    rows, excluded = eval_cli.load_rows("dev")
    assert excluded == 6
    assert len(rows) == 14
    assert {row.id for row in rows}.isdisjoint(FEW_SHOT_IDS)


def test_include_fewshot_scores_all_dev_rows() -> None:
    rows, excluded = eval_cli.load_rows("dev", include_fewshot=True)
    assert excluded == 0
    assert len(rows) == 20


def test_test_set_is_never_reduced() -> None:
    rows, excluded = eval_cli.load_rows("test")
    assert excluded == 0
    assert len(rows) == 58


def test_limit_applies_after_the_exclusion() -> None:
    rows, excluded = eval_cli.load_rows("dev", 3)
    assert excluded == 6
    assert len(rows) == 3
    assert {row.id for row in rows}.isdisjoint(FEW_SHOT_IDS)


def test_include_fewshot_flag_defaults_to_off() -> None:
    assert eval_cli.parse_args(["--set", "dev", "--no-rules"]).include_fewshot is False
    args = eval_cli.parse_args(["--set", "dev", "--no-rules", "--include-fewshot"])
    assert args.include_fewshot is True


def test_report_states_how_many_were_excluded() -> None:
    rows, excluded = eval_cli.load_rows("dev")
    predictions = eval_cli.run(rows, always(RequestType.COMPLAINT))
    summary = eval_cli.summarise(predictions, "dev", "no-rules", excluded)
    assert summary["n"] == 14
    assert summary["excluded_few_shot"] == 6
    assert "Few-shot examples excluded: 6" in eval_cli.to_markdown(summary)


def test_main_prints_the_excluded_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(eval_cli, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(eval_cli, "GeminiClient", lambda: always(RequestType.COMPLAINT))
    monkeypatch.setattr(eval_cli, "CachedLLM", lambda inner, model: inner)
    assert eval_cli.main(["--set", "dev", "--no-rules"]) == 0
    assert "Excluded 6 few-shot example(s)" in capsys.readouterr().out


def test_run_scores_each_row_against_its_label() -> None:
    rows = data.load_dev()
    predictions = eval_cli.run(rows, always(RequestType.COMPLAINT))
    summary = eval_cli.summarise(predictions, "dev", "no-rules")
    expected_correct = sum(r.expected_type is RequestType.COMPLAINT for r in rows)
    assert summary["n"] == 20
    assert summary["correct"] == expected_correct
    assert len(summary["misses"]) == 20 - expected_correct


def test_llm_never_sees_the_labels() -> None:
    rows = data.load_dev()
    llm = always(RequestType.COMPLAINT)
    eval_cli.run(rows, llm)
    for row, call in zip(rows, llm.calls, strict=True):
        assert call.user == classifier_user_prompt(row.to_request())
        if row.edge_case:
            assert row.edge_case not in call.user


def test_a_failed_llm_call_stops_the_run_instead_of_being_scored() -> None:
    # The last dev rows are labelled unclear; scoring a failure there would count as correct.
    with pytest.raises(LLMError, match="DEV-018"):
        eval_cli.run(data.load_dev()[17:], FakeLLM())


def test_main_writes_no_report_when_the_llm_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(eval_cli, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(eval_cli, "GeminiClient", FakeLLM)
    monkeypatch.setattr(eval_cli, "CachedLLM", lambda inner, model: inner)
    assert eval_cli.main(["--set", "dev", "--no-rules", "--limit", "2"]) == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("limit", ["0", "-3"])
def test_limit_must_be_positive(limit: str) -> None:
    with pytest.raises(SystemExit):
        eval_cli.parse_args(["--set", "dev", "--no-rules", "--limit", limit])


def test_write_outputs_creates_report_json_and_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(eval_cli, "RESULTS_DIR", tmp_path)
    predictions = eval_cli.run(data.load_dev()[:4], always(RequestType.CRITICAL))
    summary = eval_cli.summarise(predictions, "dev", "no-rules")

    report = eval_cli.write_outputs(summary, predictions)

    assert report.name.startswith("dev_no-rules_")
    assert report.suffix == ".md"
    assert json.loads(report.with_suffix(".json").read_text(encoding="utf-8"))["n"] == 4
    (raw,) = (tmp_path / "raw").glob("*.jsonl")
    assert len(raw.read_text(encoding="utf-8").splitlines()) == 4


def test_running_with_rules_is_refused_until_m3() -> None:
    assert eval_cli.main(["--set", "dev"]) == 2
