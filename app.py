"""Streamlit app: process requests on page 1, review the queue and the log on page 2.

Run with `streamlit run app.py`. The functions above the "pages" section hold the logic
behind the buttons and use no Streamlit calls, so they are unit-tested directly.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from wattle import config, data, db, llm
from wattle.deadlines import now_local
from wattle.llm import LLMClient
from wattle.models import CaseStatus, Request, SendMode
from wattle.pipeline import process

PAGE_PROCESS = "Process requests"
PAGE_REVIEW = "Review queue & log"
SOURCE_SAMPLE = "Pick a sample"
SOURCE_CUSTOM = "Paste custom text"

TYPE_COLOURS = {
    "critical": "red",
    "hardship": "orange",
    "complaint": "violet",
    "service_request": "blue",
    "general_enquiry": "green",
    "human_review": "gray",
}
# Send mode -> (badge text, badge colour).
SEND_MODE_BADGES = {
    SendMode.AUTO.value: ("AUTO", "green"),
    SendMode.HELD_FOR_APPROVAL.value: ("HELD", "orange"),
    SendMode.SUPPRESSED.value: ("SUPPRESSED", "red"),
}
EMPTY = "—"

Case = dict[str, Any]


# --- logic behind the buttons (no Streamlit calls) --------------------------------------


def sample_requests() -> dict[str, Request]:
    """The dev and test requests by "id — subject", with their labels stripped off."""
    rows = [*data.load_dev(), *data.load_test()]
    return {f"{row.id} — {row.subject or '(no subject)'}": row.to_request() for row in rows}


def custom_request(text: str, now: datetime) -> Request:
    """Wrap pasted text as a request with its own ID, so it gets its own case and reference."""
    digest = hashlib.sha256(f"{now.isoformat()}|{text}".encode()).hexdigest()
    return Request(id=f"CUSTOM-{digest[:8].upper()}", channel="web_form", subject="", body=text)


def is_approved(case_id: str, db_path: str | Path) -> bool:
    """True if a reviewer has already signed this case off."""
    case = db.get_case(case_id, db_path=db_path)
    return case is not None and case["status"] == CaseStatus.APPROVED.value


def process_and_log(
    req: Request, client: LLMClient, *, use_rules: bool = True, db_path: str | Path
) -> bool:
    """Run one request and log it; an approved case is left alone and False is returned."""
    if is_approved(req.id, db_path):
        return False
    result = process(req, client, use_rules=use_rules)
    return db.save_case(result, db_path=db_path)


def process_all(
    requests: Sequence[Request],
    client: LLMClient,
    *,
    use_rules: bool = True,
    db_path: str | Path,
    on_progress: Callable[[int], None] | None = None,
) -> int:
    """Run every request and log it; return how many were skipped because already approved."""
    skipped = 0
    for done, req in enumerate(requests, start=1):
        if not process_and_log(req, client, use_rules=use_rules, db_path=db_path):
            skipped += 1
        if on_progress is not None:
            on_progress(done)
    return skipped


def held_draft(case: Case | None) -> Case | None:
    """The draft a reviewer may edit: the one held for approval, if the case has one."""
    for draft in (case or {}).get("drafts", []):
        if draft["send_mode"] == SendMode.HELD_FOR_APPROVAL.value:
            return draft
    return None


def format_time(iso_text: str | None) -> str:
    """A stored ISO time as people read it, e.g. 12 Oct 2026, 09:30."""
    if not iso_text:
        return EMPTY
    return datetime.fromisoformat(iso_text).strftime("%d %b %Y, %H:%M")


def format_detail(detail: dict[str, Any]) -> str:
    """A step's detail on one plain line, e.g. "days: 21 · kind: statutory"."""
    return " · ".join(f"{key}: {_plain(value)}" for key, value in detail.items())


def _plain(value: object) -> str:
    """Lists as comma-separated text and missing values as a dash."""
    if value is None or value == []:
        return EMPTY
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def llm_failure_note(case: Case) -> str | None:
    """The LLM error behind a case, if a call failed, so the page can say so plainly."""
    if case["rationale"].startswith("LLM error:"):
        return case["rationale"]
    for step in case.get("steps", []):
        if step["detail"].get("llm_error"):
            return str(step["detail"].get("reason"))
    return None


def summary_table(cases: Sequence[Case]) -> pd.DataFrame:
    """One row per case with the columns an operations team scans first."""
    rows = [
        {
            "id": case["id"],
            "subject": case["subject"],
            "final_type": case["final_type"],
            "urgency": case["urgency"],
            "ai_type": case["ai_type"],
            "confidence": case["ai_confidence"],
            "rules_fired": ", ".join(case["rules_fired"]),
            "status": case["status"],
            "route_to": case["route_to"] or EMPTY,
            "due": format_time(case["due_at"]),
        }
        for case in cases
    ]
    return pd.DataFrame(rows)


def counts(cases: Sequence[Case], column: str) -> dict[str, int]:
    """How many cases have each value of a column, e.g. per status."""
    return dict(Counter(case[column] for case in cases))


# --- pages ------------------------------------------------------------------------------


@st.cache_resource
def get_llm() -> LLMClient:
    """One Gemini client for the whole server process; repeat requests come from the cache."""
    return llm.CachedLLM(llm.GeminiClient(), model=config.LLM_MODEL)


def render_card(case: Case) -> None:
    """The result card: the decision, why it was made, what was done and what was drafted."""
    # Customer and LLM text is always shown with st.text, which never renders Markdown or
    # links, so a message cannot put a clickable link or fake formatting on a reviewer's screen.
    st.subheader(case["id"])
    st.text(f"Subject: {case['subject'] or '(no subject)'}")
    st.text(case["body"])
    st.badge(case["final_type"].upper(), color=TYPE_COLOURS[case["final_type"]])
    left, middle, right = st.columns(3)
    left.metric("Urgency", case["urgency"])
    middle.metric("AI type (confidence)", f"{case['ai_type']} ({case['ai_confidence']:.2f})")
    right.metric("Status", case["status"])
    st.markdown(f"**Rules fired:** {', '.join(case['rules_fired']) or 'none'}")
    st.text(f"Keyword hits: {', '.join(case['keyword_hits']) or 'none'}")
    st.markdown(f"**Route:** {case['route_to'] or EMPTY} · **Due:** {format_time(case['due_at'])}")
    st.markdown("**AI rationale:**")
    st.text(case["rationale"])
    _render_steps(case)
    _render_drafts(case)


def _render_steps(case: Case) -> None:
    """The audit trail, one numbered line per step."""
    st.markdown("**Steps**")
    for step in case["steps"]:
        st.markdown(f"{step['seq']}. `{step['name']}`")
        if step["detail"]:
            st.text(format_detail(step["detail"]))


def _render_drafts(case: Case) -> None:
    """Each draft with a badge saying whether it may be sent."""
    st.markdown("**Drafts**")
    if not case["drafts"]:
        st.caption("No draft: nothing is sent for this case.")
    for draft in case["drafts"]:
        label, colour = SEND_MODE_BADGES[draft["send_mode"]]
        st.badge(f"{label} · {draft['audience']}", color=colour)
        st.text(draft["text"])


def render_sidebar() -> tuple[str, bool]:
    """Page switch, the rules on/off toggle (the ablation, live) and the demo reset."""
    page = st.sidebar.radio("Page", [PAGE_PROCESS, PAGE_REVIEW], key="page")
    use_rules = st.sidebar.toggle("Rules on", value=True, key="use_rules")
    if not use_rules:
        st.sidebar.caption("AI only: the keyword check and rules R0-R4 are switched off.")
    if st.sidebar.button("Reset demo data"):
        db.reset_db(config.DB_PATH)
        st.session_state.pop("last_case_id", None)
        st.session_state.pop("last_batch", None)
        st.session_state.pop("last_batch_ids", None)
        st.sidebar.success("Demo data cleared.")
    return page, use_rules


def choose_request(samples: dict[str, Request]) -> Request | None:
    """The request to process: a chosen sample, or pasted text (None while it is empty)."""
    source = st.radio("Request", [SOURCE_SAMPLE, SOURCE_CUSTOM], horizontal=True, key="source")
    if source == SOURCE_SAMPLE:
        return samples[st.selectbox("Sample", list(samples), key="sample")]
    text = st.text_area("Customer message", key="custom_text")
    return custom_request(text, now_local()) if text.strip() else None


def page_process(use_rules: bool) -> None:
    """Page 1: process one request or a whole file, and show what the system decided."""
    st.header(PAGE_PROCESS)
    req = choose_request(sample_requests())
    one, everything = st.columns(2)
    if one.button("Process", type="primary", disabled=req is None) and req is not None:
        _process_one(req, use_rules)
    file_name = everything.selectbox("File", ["dev", "test"], key="batch_file")
    if everything.button("Process all samples"):
        _process_file(file_name, use_rules)
    if "last_batch" in st.session_state:
        st.success(st.session_state["last_batch"])
        batch_ids = st.session_state["last_batch_ids"]
        cases = [c for c in db.list_cases(db_path=config.DB_PATH) if c["id"] in batch_ids]
        st.dataframe(summary_table(cases), hide_index=True)
        failed = [c["id"] for c in cases if llm_failure_note(c)]
        if failed:
            st.error(f"The LLM call failed for {len(failed)} request(s): {', '.join(failed)}.")
    case = db.get_case(st.session_state.get("last_case_id", ""), db_path=config.DB_PATH)
    if case is not None:
        st.divider()
        _render_llm_failure(case)
        render_card(case)


def _render_llm_failure(case: Case) -> None:
    """Say plainly when the LLM call failed (for example no API key and no cached answer)."""
    note = llm_failure_note(case)
    if note:
        st.error(f"The LLM call failed, so the safe path was taken. {note}")


def _process_one(req: Request, use_rules: bool) -> None:
    """Handle the Process button; an approved case is shown as it stands, not re-run."""
    saved = process_and_log(req, get_llm(), use_rules=use_rules, db_path=config.DB_PATH)
    if not saved:
        st.warning(f"{req.id} is already approved, so it was not processed again.")
    st.session_state["last_case_id"] = req.id
    st.session_state.pop("last_batch", None)


def _process_file(file_name: str, use_rules: bool) -> None:
    """Handle the Process all button, with a progress bar because uncached calls are slow."""
    rows = data.load_dev() if file_name == "dev" else data.load_test()
    requests = [row.to_request() for row in rows]
    bar = st.progress(0.0, text=f"Processing {len(requests)} requests")
    skipped = process_all(
        requests,
        get_llm(),
        use_rules=use_rules,
        db_path=config.DB_PATH,
        on_progress=lambda done: bar.progress(done / len(requests)),
    )
    bar.empty()
    processed = len(requests) - skipped
    st.session_state["last_batch"] = (
        f"Processed {processed} request(s) from the {file_name} file; "
        f"skipped {skipped} already approved."
    )
    st.session_state["last_batch_ids"] = [req.id for req in requests]
    st.session_state.pop("last_case_id", None)


def page_review() -> None:
    """Page 2: the human review queue, counts, and the full decision log."""
    st.header(PAGE_REVIEW)
    cases = db.list_cases(db_path=config.DB_PATH)
    queue = [case for case in cases if case["status"] == CaseStatus.AWAITING_HUMAN.value]
    st.subheader(f"Waiting for a person ({len(queue)})")
    if not queue:
        st.caption("The review queue is empty.")
    for listed in queue:  # already sorted: most urgent first, then earliest due date
        title = f"{listed['urgency'].upper()} · {listed['id']} · {listed['final_type']}"
        with st.expander(f"{title} · due {format_time(listed['due_at'])}"):
            _render_review(listed["id"])
    _render_counts(cases)
    _render_log(cases)


def _render_review(case_id: str) -> None:
    """One queued case with its card, an editable held draft and the Approve button."""
    case = db.get_case(case_id, db_path=config.DB_PATH)
    if case is None:
        return
    render_card(case)
    draft = held_draft(case)
    edited = None
    if draft is not None:
        edited = st.text_area(
            "Edit the held draft", draft["text"], height=300, key=f"edit-{case_id}"
        )
    else:
        st.caption("No held draft to edit. Approving records that a person has reviewed the case.")
    note = st.text_input("Reviewer note", key=f"note-{case_id}")
    if st.button("Approve", key=f"approve-{case_id}"):
        try:
            db.approve_case(case_id, edited, note, db_path=config.DB_PATH)
        except (KeyError, ValueError) as exc:  # the case was reset meanwhile, or a blank edit
            st.error(str(exc))
        else:
            st.rerun()


def _render_counts(cases: Sequence[Case]) -> None:
    """How many cases sit in each status, and how many of each final type."""
    st.subheader("Counts")
    if not cases:
        st.caption("No cases yet. Process some requests on the first page.")
        return
    by_status = counts(cases, "status")
    for column, (status, number) in zip(st.columns(len(by_status)), by_status.items(), strict=True):
        column.metric(status, number)
    st.bar_chart(pd.Series(counts(cases, "final_type"), name="cases"))


def _render_log(cases: Sequence[Case]) -> None:
    """Every case in a table; pick one to see its steps and drafts."""
    st.subheader("Decision log")
    if not cases:
        return
    st.dataframe(summary_table(cases), hide_index=True)
    chosen = st.selectbox("Show the steps for", [case["id"] for case in cases], key="log_case")
    case = db.get_case(chosen, db_path=config.DB_PATH)
    if case is not None:
        with st.expander(f"Steps and drafts for {chosen}"):
            render_card(case)


def main() -> None:
    """Draw the sidebar, then the page it selects."""
    st.set_page_config(page_title="Wattle Bank — Request Workflow", layout="wide")
    st.title("Wattle Bank — Request Workflow")
    st.caption("The AI classifies. Code decides. A human approves anything risky.")
    db.init_db(config.DB_PATH)
    page, use_rules = render_sidebar()
    if page == PAGE_PROCESS:
        page_process(use_rules)
    else:
        page_review()


if __name__ == "__main__":
    main()
