"""Tests for the decision log: a case round-trips, and approval is recorded step by step."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from helpers import NOW, make_classification, make_request, scripted_llm
from wattle import db
from wattle.models import CaseResult, FaqSelection
from wattle.pipeline import process

LATER = NOW + timedelta(hours=2)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A fresh database file per test, so tests never share state."""
    path = tmp_path / "test.db"
    db.init_db(path)
    return path


def make_case(
    request_type: str, case_id: str, body: str = "Customer text", **overrides: object
) -> CaseResult:
    """Run a crafted request through the real pipeline so the saved shape is the real one."""
    classification = make_classification(request_type, **overrides)
    llm = scripted_llm(classification, FaqSelection(faq_id="FAQ-01", confidence=0.9))
    return process(make_request(body, id=case_id), llm, now=NOW)


def test_init_db_creates_the_three_tables(db_path: Path) -> None:
    assert db.list_cases(db_path=db_path) == []
    assert db.get_case("nope", db_path=db_path) is None


def test_a_case_round_trips_with_its_steps_and_drafts(db_path: Path) -> None:
    result = make_case("hardship", "C-1", "I lost my job.", product="home_loan")
    db.save_case(result, db_path=db_path)

    case = db.get_case("C-1", db_path=db_path)
    assert case is not None
    assert case["id"] == "C-1"
    assert case["received_at"] == NOW.isoformat()
    assert case["body"] == "I lost my job."
    assert case["ai_type"] == "hardship"
    assert case["ai_confidence"] == 0.9
    assert case["final_type"] == "hardship"
    assert case["urgency"] == "high"
    assert case["rules_fired"] == []
    assert case["keyword_hits"] == ["lost my job"]
    assert case["rationale"] == "Why this type was chosen."
    assert case["route_to"] == "hardship_team"
    assert case["due_at"] == (NOW + timedelta(days=21)).isoformat()
    assert case["status"] == "awaiting_human"
    assert case["requires_human"] is True
    assert case["approved_at"] is None
    assert [step["name"] for step in case["steps"]] == ["draft_reply", "route", "start_clock"]
    assert [step["seq"] for step in case["steps"]] == [1, 2, 3]
    assert case["steps"][2]["detail"]["days"] == 21
    assert case["steps"][0]["created_at"] == NOW.isoformat()
    (draft,) = case["drafts"]
    assert draft["audience"] == "customer"
    assert draft["send_mode"] == "held_for_approval"
    assert draft["text"] == result.branch.drafts[0].text


def test_a_case_with_no_due_date_or_route_is_stored_with_nulls(db_path: Path) -> None:
    db.save_case(make_case("general_enquiry", "C-2"), db_path=db_path)
    case = db.get_case("C-2", db_path=db_path)
    assert case is not None
    assert case["due_at"] is None
    assert case["route_to"] is None
    assert case["requires_human"] is False


def test_saving_the_same_request_again_replaces_it(db_path: Path) -> None:
    db.save_case(make_case("complaint", "C-3"), db_path=db_path)
    db.save_case(make_case("service_request", "C-3"), db_path=db_path)

    (listed,) = db.list_cases(db_path=db_path)
    assert listed["final_type"] == "service_request"
    case = db.get_case("C-3", db_path=db_path)
    assert case is not None
    assert [step["name"] for step in case["steps"]] == ["route", "confirm", "start_clock"]
    assert len(case["drafts"]) == 1


def test_list_cases_can_filter_by_status(db_path: Path) -> None:
    db.save_case(make_case("complaint", "C-4"), db_path=db_path)
    db.save_case(make_case("hardship", "C-5"), db_path=db_path)
    waiting = db.list_cases("awaiting_human", db_path=db_path)
    assert [case["id"] for case in waiting] == ["C-5"]
    assert len(db.list_cases(db_path=db_path)) == 2


def test_list_cases_sorts_by_urgency_then_due_date(db_path: Path) -> None:
    db.save_case(make_case("unclear", "medium-no-due"), db_path=db_path)
    db.save_case(make_case("hardship", "high-21-days", product="home_loan"), db_path=db_path)
    db.save_case(make_case("hardship", "high-10-days", product="savings"), db_path=db_path)
    db.save_case(make_case("critical", "critical"), db_path=db_path)
    db.save_case(make_case("general_enquiry", "low"), db_path=db_path)

    assert [case["id"] for case in db.list_cases(db_path=db_path)] == [
        "critical",
        "high-10-days",
        "high-21-days",
        "medium-no-due",
        "low",
    ]


def test_due_dates_sort_by_real_time_across_a_daylight_saving_change(db_path: Path) -> None:
    # On 4 April 2027 clocks go back from 03:00 to 02:00, so 02:00-03:00 happens twice.
    # "first" is due at 02:35 (UTC+11) and "second" at 02:20 (UTC+10), 45 minutes later in
    # real time, although as text "02:20" sorts before "02:35".
    melbourne = ZoneInfo("Australia/Melbourne")
    before_change = datetime(2027, 4, 4, 2, 20, tzinfo=melbourne)
    after_change = datetime(2027, 4, 4, 2, 5, fold=1, tzinfo=melbourne)
    for case_id, received in (("second", after_change), ("first", before_change)):
        classification = make_classification("critical")
        result = process(make_request(id=case_id), scripted_llm(classification), now=received)
        db.save_case(result, db_path=db_path)

    cases = db.list_cases(db_path=db_path)
    assert [case["id"] for case in cases] == ["first", "second"]


def test_approving_updates_the_held_draft_and_logs_the_step(db_path: Path) -> None:
    db.save_case(make_case("hardship", "C-6"), db_path=db_path)

    db.approve_case("C-6", "Edited reply.", "Checked by Sam", now=LATER, db_path=db_path)

    case = db.get_case("C-6", db_path=db_path)
    assert case is not None
    assert case["status"] == "approved"
    assert case["approved_at"] == LATER.isoformat()
    assert case["reviewer_note"] == "Checked by Sam"
    assert case["drafts"][0]["text"] == "Edited reply."
    last = case["steps"][-1]
    assert last["name"] == "human_approved"
    assert last["seq"] == 4
    assert last["created_at"] == LATER.isoformat()
    assert last["detail"] == {"note": "Checked by Sam", "draft_edited": True}
    assert db.list_cases("awaiting_human", db_path=db_path) == []


def test_approving_without_new_text_keeps_the_draft(db_path: Path) -> None:
    result = make_case("hardship", "C-7")
    db.save_case(result, db_path=db_path)
    db.approve_case("C-7", None, "", now=LATER, db_path=db_path)
    case = db.get_case("C-7", db_path=db_path)
    assert case is not None
    assert case["drafts"][0]["text"] == result.branch.drafts[0].text
    assert case["steps"][-1]["detail"]["draft_edited"] is False


def test_approving_never_rewrites_a_suppressed_draft(db_path: Path) -> None:
    result = make_case("critical", "C-8", risk_flags=["self_harm"])
    db.save_case(result, db_path=db_path)
    db.approve_case("C-8", "New text", "note", now=LATER, db_path=db_path)
    case = db.get_case("C-8", db_path=db_path)
    assert case is not None
    assert case["status"] == "approved"
    assert case["drafts"][0]["send_mode"] == "suppressed"
    assert "Lifeline" in case["drafts"][0]["text"]
    assert case["steps"][-1]["detail"]["draft_edited"] is False


def test_a_case_with_no_draft_can_still_be_approved(db_path: Path) -> None:
    db.save_case(make_case("unclear", "C-9"), db_path=db_path)
    db.approve_case("C-9", None, "Spam, closed", now=LATER, db_path=db_path)
    case = db.get_case("C-9", db_path=db_path)
    assert case is not None
    assert case["status"] == "approved"
    assert case["drafts"] == []


def test_approving_an_unknown_case_is_an_error(db_path: Path) -> None:
    with pytest.raises(KeyError, match="C-404"):
        db.approve_case("C-404", None, "", db_path=db_path)


def test_only_a_case_waiting_for_a_human_can_be_approved(db_path: Path) -> None:
    db.save_case(make_case("complaint", "C-10"), db_path=db_path)
    with pytest.raises(ValueError, match="routed"):
        db.approve_case("C-10", None, "", db_path=db_path)

    db.save_case(make_case("hardship", "C-11"), db_path=db_path)
    db.approve_case("C-11", None, "", db_path=db_path)
    with pytest.raises(ValueError, match="approved"):
        db.approve_case("C-11", None, "", db_path=db_path)


def test_an_empty_edit_is_rejected_so_a_blank_reply_cannot_be_approved(db_path: Path) -> None:
    db.save_case(make_case("hardship", "C-12"), db_path=db_path)
    with pytest.raises(ValueError, match="empty"):
        db.approve_case("C-12", "   ", "", db_path=db_path)
    assert db.get_case("C-12", db_path=db_path)["status"] == "awaiting_human"  # type: ignore[index]


def test_reset_db_removes_everything(db_path: Path) -> None:
    db.save_case(make_case("hardship", "C-13"), db_path=db_path)
    db.reset_db(db_path)
    assert db.list_cases(db_path=db_path) == []
    assert db.get_case("C-13", db_path=db_path) is None


def test_functions_work_on_a_database_that_was_never_initialised(tmp_path: Path) -> None:
    path = tmp_path / "new.db"
    db.save_case(make_case("complaint", "C-14"), db_path=path)
    assert len(db.list_cases(db_path=path)) == 1
