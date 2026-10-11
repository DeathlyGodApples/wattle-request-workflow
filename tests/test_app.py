"""Tests for the app: the logic behind the buttons, and that both pages render and work."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import app
from helpers import NOW, make_classification, make_request, scripted_llm
from wattle import db
from wattle.llm import FakeLLM
from wattle.llm import GeminiClient as RealGeminiClient
from wattle.models import FaqSelection

FAQ_PICK = FaqSelection(faq_id="FAQ-02", confidence=0.9)
NEWLINE = chr(10)


def fake(request_type: str, **overrides: object) -> FakeLLM:
    """A fake LLM that classifies everything as one type."""
    return scripted_llm(make_classification(request_type, **overrides), FAQ_PICK)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "app.db"


# --- logic behind the buttons -------------------------------------------------------------


def test_samples_cover_dev_and_test_files_and_carry_no_labels() -> None:
    samples = app.sample_requests()
    assert len(samples) == 20 + 58
    label, request = next(iter(samples.items()))
    assert label.startswith(f"{request.id} — ")
    assert not hasattr(request, "expected_type")


def test_custom_text_becomes_a_request_with_a_unique_id() -> None:
    first = app.custom_request("My card is lost", NOW)
    second = app.custom_request("My card is lost", NOW.replace(second=1))
    assert first.body == "My card is lost"
    assert first.id.startswith("CUSTOM-")
    assert first.id != second.id


def test_process_and_log_saves_the_case(db_path: Path) -> None:
    saved = app.process_and_log(make_request(id="A-1"), fake("hardship"), db_path=db_path)
    assert saved is True
    case = db.get_case("A-1", db_path=db_path)
    assert case is not None
    assert case["final_type"] == "hardship"


def test_process_and_log_leaves_an_approved_case_alone_and_calls_no_llm(db_path: Path) -> None:
    app.process_and_log(make_request(id="A-2"), fake("hardship"), db_path=db_path)
    db.approve_case("A-2", None, "ok", db_path=db_path)

    llm = fake("complaint")
    saved = app.process_and_log(make_request(id="A-2"), llm, db_path=db_path)

    assert saved is False
    assert llm.calls == []
    case = db.get_case("A-2", db_path=db_path)
    assert case is not None
    assert (case["status"], case["final_type"]) == ("approved", "hardship")


def test_process_all_skips_approved_cases_and_reports_how_many(db_path: Path) -> None:
    requests = [make_request(id=f"B-{n}") for n in range(4)]
    app.process_all(requests[:2], fake("hardship"), db_path=db_path)
    db.approve_case("B-0", None, "", db_path=db_path)
    db.approve_case("B-1", None, "", db_path=db_path)

    seen: list[int] = []
    skipped = app.process_all(requests, fake("complaint"), db_path=db_path, on_progress=seen.append)

    assert skipped == 2
    assert seen == [1, 2, 3, 4]
    by_id = {case["id"]: case["final_type"] for case in db.list_cases(db_path=db_path)}
    assert by_id == {"B-0": "hardship", "B-1": "hardship", "B-2": "complaint", "B-3": "complaint"}


def test_process_all_can_run_without_rules(db_path: Path) -> None:
    stolen = make_request("My card was stolen.", id="B-9")
    app.process_all([stolen], fake("general_enquiry"), use_rules=False, db_path=db_path)
    case = db.get_case("B-9", db_path=db_path)
    assert case is not None
    assert case["final_type"] == "general_enquiry"
    assert case["rules_fired"] == ["AI_ONLY"]


def test_held_draft_finds_only_a_draft_waiting_for_approval(db_path: Path) -> None:
    app.process_and_log(make_request(id="C-1"), fake("hardship"), db_path=db_path)
    app.process_and_log(
        make_request(id="C-2"), fake("critical", risk_flags=["self_harm"]), db_path=db_path
    )
    app.process_and_log(make_request(id="C-3"), fake("unclear"), db_path=db_path)

    held = app.held_draft(db.get_case("C-1", db_path=db_path))
    assert held is not None
    assert held["send_mode"] == "held_for_approval"
    assert app.held_draft(db.get_case("C-2", db_path=db_path)) is None  # suppressed
    assert app.held_draft(db.get_case("C-3", db_path=db_path)) is None  # no draft at all


def test_summary_table_has_one_row_per_case_with_readable_columns(db_path: Path) -> None:
    app.process_all(
        [make_request(id="D-1"), make_request(id="D-2")], fake("complaint"), db_path=db_path
    )
    table = app.summary_table(db.list_cases(db_path=db_path))
    assert list(table["id"]) == ["D-1", "D-2"]
    assert {"final_type", "urgency", "status", "route_to", "due", "rules_fired"} <= set(
        table.columns
    )
    assert table.loc[0, "rules_fired"] == ""
    assert app.summary_table([]).empty


def test_counts_tally_a_column() -> None:
    cases = [{"status": "routed"}, {"status": "routed"}, {"status": "approved"}]
    assert app.counts(cases, "status") == {"routed": 2, "approved": 1}


def test_times_are_shown_in_a_readable_form() -> None:
    assert app.format_time(NOW.isoformat()) == "12 Oct 2026, 09:30"
    assert app.format_time(None) == "—"


def test_step_details_are_shown_as_one_plain_line() -> None:
    assert app.format_detail({"days": 21, "kind": "statutory"}) == "days: 21 · kind: statutory"
    assert app.format_detail({"missing": ["a", "b"], "amount": None}) == "missing: a, b · amount: —"
    assert app.format_detail({}) == ""


# --- the pages themselves -------------------------------------------------------------------


@pytest.fixture
def run_app(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AppTest]:
    """The real app script, wired to a temp database and a fake LLM (hardship for everything)."""
    monkeypatch.setattr("wattle.config.DB_PATH", str(db_path))
    monkeypatch.setattr("wattle.llm.GeminiClient", lambda: fake("hardship", product="home_loan"))
    monkeypatch.setattr("wattle.llm.CachedLLM", lambda inner, model: inner)
    st.cache_resource.clear()
    yield AppTest.from_file(app.__file__, default_timeout=10)
    st.cache_resource.clear()


def button(at: AppTest, label: str) -> Any:
    (found,) = [b for b in at.button if b.label == label]
    return found


def all_text(at: AppTest) -> str:
    """Everything written on the page, for simple "is it shown" checks."""
    elements = [*at.markdown, *at.text, *at.caption, *at.subheader, *at.success, *at.warning]
    return "\n".join(str(element.value) for element in elements)


def test_page_one_processes_a_sample_and_shows_the_result_card(
    run_app: AppTest, db_path: Path
) -> None:
    at = run_app.run()
    assert not at.exception
    button(at, "Process").click().run()

    assert not at.exception
    (case,) = db.list_cases(db_path=db_path)
    assert case["id"] == "DEV-001"
    assert case["final_type"] == "critical"  # "skimmed" is a critical keyword, so R1 wins
    shown = all_text(at)
    assert "DEV-001" in shown
    assert "pause_automation" in shown
    assert "HELD" in shown


def test_the_rules_toggle_switches_to_ai_only(run_app: AppTest, db_path: Path) -> None:
    at = run_app.run()
    at.sidebar.toggle[0].set_value(False).run()
    button(at, "Process").click().run()

    assert not at.exception
    (case,) = db.list_cases(db_path=db_path)
    assert case["final_type"] == "hardship"
    assert case["rules_fired"] == ["AI_ONLY"]


def test_custom_text_can_be_processed(run_app: AppTest, db_path: Path) -> None:
    at = run_app.run()
    at.radio(key="source").set_value("Paste custom text").run()
    at.text_area(key="custom_text").set_value("I lost my job and need help.").run()
    button(at, "Process").click().run()

    assert not at.exception
    (case,) = db.list_cases(db_path=db_path)
    assert case["id"].startswith("CUSTOM-")
    assert case["body"] == "I lost my job and need help."


def test_process_all_reports_processed_and_skipped_counts(run_app: AppTest, db_path: Path) -> None:
    at = run_app.run()
    button(at, "Process").click().run()
    db.approve_case("DEV-001", None, "", db_path=db_path)

    button(at, "Process all samples").click().run()

    assert not at.exception
    assert len(db.list_cases(db_path=db_path)) == 20
    assert "Processed 19" in all_text(at)
    assert "skipped 1" in all_text(at)
    assert db.get_case("DEV-001", db_path=db_path)["status"] == "approved"  # type: ignore[index]


def test_approving_on_page_two_moves_the_case_out_of_the_queue(
    run_app: AppTest, db_path: Path
) -> None:
    request = make_request("I lost my job.", id="Q-1")
    app.process_and_log(request, fake("hardship", product="home_loan"), db_path=db_path)

    at = run_app.run()
    at.sidebar.radio(key="page").set_value(app.PAGE_REVIEW).run()
    assert not at.exception
    assert "Q-1" in "\n".join(e.label for e in at.expander)

    at.text_area(key="edit-Q-1").set_value("Edited by the reviewer.")
    at.text_input(key="note-Q-1").set_value("Checked")
    button(at, "Approve").click().run()

    assert not at.exception
    case = db.get_case("Q-1", db_path=db_path)
    assert case is not None
    assert case["status"] == "approved"
    assert case["reviewer_note"] == "Checked"
    assert case["drafts"][0]["text"] == "Edited by the reviewer."
    assert db.list_cases("awaiting_human", db_path=db_path) == []
    assert "The review queue is empty" in all_text(at)


def test_reset_demo_data_empties_the_log(run_app: AppTest, db_path: Path) -> None:
    app.process_and_log(make_request(id="R-1"), fake("complaint"), db_path=db_path)
    at = run_app.run()
    button(at, "Reset demo data").click().run()
    assert not at.exception
    assert db.list_cases(db_path=db_path) == []


def test_a_failed_llm_call_is_explained_and_the_safe_path_is_taken(
    run_app: AppTest, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("wattle.llm.GeminiClient", lambda: FakeLLM())  # every call fails
    at = run_app.run()
    at.radio(key="source").set_value("Paste custom text").run()
    at.text_area(key="custom_text").set_value("Where is my new card?").run()
    button(at, "Process").click().run()

    assert not at.exception
    assert "The LLM call failed, so the safe path was taken." in at.error[0].value
    (case,) = db.list_cases(db_path=db_path)
    assert case["final_type"] == "human_review"
    assert case["rules_fired"] == ["R0", "R4"]


def test_the_demo_works_with_no_api_key_from_the_committed_cache(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real cache and the real Gemini client, but with no key: DEV-001 is a cache hit.
    monkeypatch.setattr("wattle.config.DB_PATH", str(db_path))
    monkeypatch.setattr("wattle.config.LLM_MODEL", app.config.DEFAULT_LLM_MODEL)
    monkeypatch.setattr("wattle.llm.GeminiClient", lambda: RealGeminiClient(api_key=""))
    st.cache_resource.clear()
    at = AppTest.from_file(app.__file__, default_timeout=10).run()
    button(at, "Process").click().run()
    st.cache_resource.clear()

    assert not at.exception
    assert not at.error
    case = db.get_case("DEV-001", db_path=db_path)
    assert case is not None
    assert case["final_type"] == "critical"
    assert not case["rationale"].startswith("LLM error")


def test_no_api_key_and_no_cached_answer_says_the_key_is_missing(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("wattle.config.DB_PATH", str(db_path))
    monkeypatch.setattr("wattle.llm.GeminiClient", lambda: RealGeminiClient(api_key=""))
    monkeypatch.setattr("wattle.llm.CachedLLM", lambda inner, model: inner)  # always a miss
    st.cache_resource.clear()
    at = AppTest.from_file(app.__file__, default_timeout=10).run()
    button(at, "Process").click().run()
    st.cache_resource.clear()

    assert not at.exception
    assert "GEMINI_API_KEY is not set" in at.error[0].value


def test_llm_failure_note_reads_the_classifier_and_the_faq_step() -> None:
    fine = {"rationale": "Explicit complaint.", "steps": [{"detail": {"llm_error": False}}]}
    classify_failed = {"rationale": "LLM error: quota exceeded"}
    faq_failed = {
        "rationale": "An enquiry.",
        "steps": [{"detail": {"llm_error": True, "reason": "LLM error: timeout"}}],
    }
    assert app.llm_failure_note(fine) is None
    assert app.llm_failure_note(classify_failed) == "LLM error: quota exceeded"
    assert app.llm_failure_note(faq_failed) == "LLM error: timeout"


def test_customer_and_llm_text_is_never_rendered_as_markdown(
    run_app: AppTest, db_path: Path
) -> None:
    # A link typed by a customer, or written by the LLM, must stay plain text on screen.
    link = "[Click to verify](http://evil.example)"
    request = make_request(f"Body {link}", subject=f"Subject {link}", id="M-1")
    app.process_and_log(request, fake("unclear", rationale=f"Rationale {link}"), db_path=db_path)

    at = run_app.run()
    at.sidebar.radio(key="page").set_value(app.PAGE_REVIEW).run()

    assert not at.exception
    rendered = [*at.markdown, *at.caption, *at.subheader, *at.header]
    assert not any(link in str(element.value) for element in rendered)
    plain = NEWLINE.join(str(element.value) for element in at.text)
    for part in ("Subject", "Body", "Rationale"):
        assert f"{part} {link}" in plain


def test_processing_an_approved_sample_again_warns_and_changes_nothing(
    run_app: AppTest, db_path: Path
) -> None:
    at = run_app.run()
    button(at, "Process").click().run()
    db.approve_case("DEV-001", None, "signed off", db_path=db_path)

    button(at, "Process").click().run()

    assert not at.exception
    assert "already approved" in at.warning[0].value
    case = db.get_case("DEV-001", db_path=db_path)
    assert case is not None
    assert (case["status"], case["reviewer_note"]) == ("approved", "signed off")


def test_a_blank_edit_is_refused_and_the_case_stays_in_the_queue(
    run_app: AppTest, db_path: Path
) -> None:
    app.process_and_log(make_request(id="Q-2"), fake("hardship"), db_path=db_path)
    at = run_app.run()
    at.sidebar.radio(key="page").set_value(app.PAGE_REVIEW).run()
    at.text_area(key="edit-Q-2").set_value("   ")
    button(at, "Approve").click().run()

    assert not at.exception
    assert "empty" in at.error[0].value
    assert [c["id"] for c in db.list_cases("awaiting_human", db_path=db_path)] == ["Q-2"]
