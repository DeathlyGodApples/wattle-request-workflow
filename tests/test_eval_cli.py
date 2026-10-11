"""Tests for the eval script's plumbing, using FakeLLM so no real call is made."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import eval as eval_cli
from helpers import make_classification, scripted_llm
from wattle import data
from wattle.llm import FakeLLM, LLMError
from wattle.models import FaqSelection, RequestType
from wattle.prompts import FEW_SHOT_IDS, classifier_user_prompt

# The FAQ entry the fake picks for every enquiry, so an enquiry gets an automatic answer.
FAQ_PICK = FaqSelection(faq_id="FAQ-12", confidence=0.9)
NO_FAQ_PICK = FaqSelection(faq_id=None, confidence=0.0)


def always(request_type: RequestType, faq: FaqSelection | None = FAQ_PICK) -> FakeLLM:
    """A fake that gives every request the same type, so the expected score is easy to state."""
    return scripted_llm(make_classification(request_type.value), faq)


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
    predictions = eval_cli.run(rows, always(RequestType.COMPLAINT), use_rules=False)
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


def dev_row(request_id: str) -> list:
    """One dev row by ID, as a list ready for eval_cli.run."""
    return [row for row in data.load_dev() if row.id == request_id]


def test_rules_catch_a_critical_request_the_llm_called_an_enquiry() -> None:
    # DEV-003 is labelled critical and mentions a stolen phone; the fake LLM misses it.
    llm = always(RequestType.GENERAL_ENQUIRY)
    (with_rules,) = eval_cli.run(dev_row("DEV-003"), llm, use_rules=True)
    (ai_alone,) = eval_cli.run(dev_row("DEV-003"), llm, use_rules=False)

    assert with_rules.predicted.value == "critical"
    assert with_rules.rules_fired[0] == "R1"
    assert "stolen" in with_rules.keyword_hits
    assert with_rules.requires_human is True
    assert with_rules.auto_customer_draft is False

    assert ai_alone.predicted.value == "general_enquiry"
    assert ai_alone.rules_fired == ["AI_ONLY"]
    assert ai_alone.auto_customer_draft is True


def test_unsafe_auto_sends_show_up_without_rules_and_not_with_them() -> None:
    llm = always(RequestType.GENERAL_ENQUIRY)
    ai_alone = eval_cli.summarise(
        eval_cli.run(dev_row("DEV-003"), llm, use_rules=False), "dev", "x"
    )
    with_rules = eval_cli.summarise(eval_cli.run(dev_row("DEV-003"), llm), "dev", "x")
    assert ai_alone["unsafe_auto_sends"] == ["DEV-003"]
    assert with_rules["unsafe_auto_sends"] == []


def test_an_enquiry_handed_to_a_person_is_not_an_auto_send() -> None:
    # No FAQ entry fits, so nothing goes to the customer even though the type is wrong.
    llm = always(RequestType.GENERAL_ENQUIRY, NO_FAQ_PICK)
    (ai_alone,) = eval_cli.run(dev_row("DEV-003"), llm, use_rules=False)
    assert ai_alone.predicted.value == "general_enquiry"
    assert ai_alone.auto_customer_draft is False


def test_a_failed_faq_call_stops_the_run_too() -> None:
    llm = always(RequestType.GENERAL_ENQUIRY, faq=None)
    with pytest.raises(LLMError, match="DEV-016: select_faq"):
        eval_cli.run(dev_row("DEV-016"), llm)


def test_hardship_clock_is_recorded_only_for_hardship_decisions() -> None:
    (hardship,) = eval_cli.run(dev_row("DEV-005"), always(RequestType.HARDSHIP))
    (complaint,) = eval_cli.run(dev_row("DEV-009"), always(RequestType.COMPLAINT))
    assert hardship.hardship_days == 21  # the fake reports product unknown, so the strict clock
    assert hardship.statutory_21d is True
    assert complaint.hardship_days is None


def test_main_runs_with_rules_and_names_the_report_for_the_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(eval_cli, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(eval_cli, "GeminiClient", lambda: always(RequestType.COMPLAINT))
    monkeypatch.setattr(eval_cli, "CachedLLM", lambda inner, model: inner)
    assert eval_cli.main(["--set", "dev"]) == 0
    (report,) = tmp_path.glob("dev_rules_*.json")
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["mode"] == "rules"
    assert "confusion" in saved
    assert "AI + rules" in report.with_suffix(".md").read_text(encoding="utf-8")


def write_summary(path: Path, mode: str, use_rules: bool) -> Path:
    """Save a small real summary to disk so compare() reads what eval writes."""
    llm = always(RequestType.GENERAL_ENQUIRY)
    predictions = eval_cli.run(data.load_dev()[:5], llm, use_rules=use_rules)
    path.write_text(json.dumps(eval_cli.summarise(predictions, "dev", mode)), encoding="utf-8")
    return path


def test_compare_puts_two_reports_side_by_side(tmp_path: Path) -> None:
    a = write_summary(tmp_path / "a.json", "no-rules", use_rules=False)
    b = write_summary(tmp_path / "b.json", "rules", use_rules=True)
    table = eval_cli.compare(a, b)
    assert "| Metric | AI alone (dev) | AI + rules (dev) |" in table
    for label in ("Type accuracy", "Critical caught", "Unsafe auto-sends", "Needs a human"):
        assert label in table
    assert "| Critical caught | 0/4 | 3/4 |" in table
    assert "WARNING" not in table


def test_compare_shows_na_for_metrics_an_older_report_lacks(tmp_path: Path) -> None:
    old = tmp_path / "old.json"
    old_format = {"dataset": "dev", "mode": "no-rules", "model": eval_cli.LLM_MODEL, "n": 5,
                  "correct": 1, "type_accuracy": 0.2, "per_type": {}, "misses": []}  # fmt: skip
    old.write_text(json.dumps(old_format), encoding="utf-8")
    new = write_summary(tmp_path / "new.json", "rules", use_rules=True)
    table = eval_cli.compare(old, new)
    assert "| Type accuracy | 1/5 (20.0%) |" in table
    assert "| Critical caught | n/a | 3/4 |" in table
    # DEV-004 has no risky keyword, so with this fake LLM even the rules let one through.
    assert "| Unsafe auto-sends (target 0) | n/a | 1 |" in table


def test_compare_warns_when_reports_are_not_comparable(tmp_path: Path) -> None:
    a = write_summary(tmp_path / "a.json", "no-rules", use_rules=False)
    other = json.loads(a.read_text(encoding="utf-8")) | {"model": "another-model"}
    b = tmp_path / "b.json"
    b.write_text(json.dumps(other), encoding="utf-8")
    assert "WARNING: reports differ in model" in eval_cli.compare(a, b)


def test_main_compare_prints_the_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = write_summary(tmp_path / "a.json", "no-rules", use_rules=False)
    b = write_summary(tmp_path / "b.json", "rules", use_rules=True)
    assert eval_cli.main(["--compare", str(a), str(b)]) == 0
    assert "AI + rules (dev)" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [[], ["--set", "dev", "--compare", "a.json", "b.json"]])
def test_exactly_one_of_set_or_compare_is_required(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        eval_cli.parse_args(argv)
