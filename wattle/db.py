"""SQLite decision log so every case, step and draft can be reviewed later.

Each function opens its own connection and closes it, so there is no shared state. Times are
stored as ISO-8601 text with their UTC offset; lists are stored as JSON text.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from wattle.config import DB_PATH
from wattle.deadlines import now_local, to_local
from wattle.models import CaseResult, CaseStatus, SendMode

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases(
    id TEXT PRIMARY KEY, received_at TEXT, channel TEXT, subject TEXT, body TEXT,
    ai_type TEXT, ai_confidence REAL, final_type TEXT, urgency TEXT,
    rules_fired TEXT, keyword_hits TEXT, rationale TEXT,
    route_to TEXT, due_at TEXT, status TEXT, requires_human INTEGER,
    approved_at TEXT, reviewer_note TEXT);
CREATE TABLE IF NOT EXISTS steps(
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, seq INTEGER,
    name TEXT, detail TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS drafts(
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, audience TEXT,
    text TEXT, send_mode TEXT);
"""

# Most urgent first, then the earliest deadline; cases with no deadline come last.
# datetime() converts the stored offset to UTC, so the order is by real time.
QUEUE_ORDER = """
ORDER BY CASE urgency WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
         due_at IS NULL, datetime(due_at), id
"""

JSON_COLUMNS = ("rules_fired", "keyword_hits")


def _connect(db_path: str | Path) -> sqlite3.Connection:
    """Open the database and make sure the tables exist, so no caller can forget to."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def init_db(db_path: str | Path = DB_PATH) -> None:
    """Create the database file and its tables if they are not there yet."""
    with closing(_connect(db_path)):
        pass


def reset_db(db_path: str | Path = DB_PATH) -> None:
    """Empty the log for a fresh demo."""
    with closing(_connect(db_path)) as conn, conn:
        conn.execute("DELETE FROM cases")
        conn.execute("DELETE FROM steps")
        conn.execute("DELETE FROM drafts")


def save_case(result: CaseResult, *, db_path: str | Path = DB_PATH) -> bool:
    """Store a processed case; return False, changing nothing, if it was already approved.

    Processing a request again replaces its earlier record, except once a person has signed
    it off: an approval and its audit trail must never be overwritten.
    """
    case_id = result.request.id
    created_at = result.received_at.isoformat()
    steps = [
        (case_id, seq, step.name, json.dumps(step.detail), created_at)
        for seq, step in enumerate(result.branch.steps, start=1)
    ]
    drafts = [
        (case_id, draft.audience, draft.text, draft.send_mode.value)
        for draft in result.branch.drafts
    ]
    with closing(_connect(db_path)) as conn, conn:
        # Take the write lock before checking, so an approval cannot slip in between.
        conn.execute("BEGIN IMMEDIATE")
        if _is_approved(conn, case_id):
            logger.info("Case %s is approved; not overwriting it", case_id)
            return False
        conn.execute("DELETE FROM steps WHERE case_id = ?", (case_id,))
        conn.execute("DELETE FROM drafts WHERE case_id = ?", (case_id,))
        conn.execute(
            "INSERT OR REPLACE INTO cases VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            _case_row(result),
        )
        conn.executemany(
            "INSERT INTO steps(case_id, seq, name, detail, created_at) VALUES (?,?,?,?,?)", steps
        )
        conn.executemany(
            "INSERT INTO drafts(case_id, audience, text, send_mode) VALUES (?,?,?,?)", drafts
        )
    return True


def _is_approved(conn: sqlite3.Connection, case_id: str) -> bool:
    """True if this case exists and a reviewer has already approved it."""
    row = conn.execute("SELECT status FROM cases WHERE id = ?", (case_id,)).fetchone()
    return row is not None and row["status"] == CaseStatus.APPROVED.value


def _case_row(result: CaseResult) -> tuple[Any, ...]:
    """One value per column of `cases`, in the order the table declares them."""
    req, c, decision, branch = (
        result.request,
        result.classification,
        result.decision,
        result.branch,
    )
    return (
        req.id,
        result.received_at.isoformat(),
        req.channel,
        req.subject,
        req.body,
        c.request_type.value,
        c.confidence,
        decision.final_type.value,
        decision.urgency.value,
        json.dumps(decision.rules_fired),
        json.dumps(result.keyword_hits),
        c.rationale,
        branch.route_to,
        branch.due_at.isoformat() if branch.due_at else None,
        branch.status.value,
        int(decision.requires_human),
        None,  # approved_at
        None,  # reviewer_note
    )


def list_cases(
    status: CaseStatus | str | None = None, *, db_path: str | Path = DB_PATH
) -> list[dict[str, Any]]:
    """Cases in work-queue order, optionally only those with one status."""
    with closing(_connect(db_path)) as conn:
        if status is None:
            rows = conn.execute(f"SELECT * FROM cases {QUEUE_ORDER}").fetchall()
        else:
            query = f"SELECT * FROM cases WHERE status = ? {QUEUE_ORDER}"
            rows = conn.execute(query, (str(status),)).fetchall()
    return [_case_dict(row) for row in rows]


def get_case(case_id: str, *, db_path: str | Path = DB_PATH) -> dict[str, Any] | None:
    """One case with its steps (in order) and drafts, or None if it is not in the log."""
    with closing(_connect(db_path)) as conn:
        row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        if row is None:
            return None
        steps = conn.execute(
            "SELECT seq, name, detail, created_at FROM steps WHERE case_id = ? ORDER BY seq",
            (case_id,),
        ).fetchall()
        drafts = conn.execute(
            "SELECT audience, text, send_mode FROM drafts WHERE case_id = ? ORDER BY id",
            (case_id,),
        ).fetchall()
    case = _case_dict(row)
    case["steps"] = [{**dict(step), "detail": json.loads(step["detail"])} for step in steps]
    case["drafts"] = [dict(draft) for draft in drafts]
    return case


def _case_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Turn a stored row back into Python values: JSON text to lists, 0/1 to bool."""
    case = dict(row)
    for column in JSON_COLUMNS:
        case[column] = json.loads(case[column])
    case["requires_human"] = bool(case["requires_human"])
    return case


def approve_case(
    case_id: str,
    edited_text: str | None,
    note: str,
    *,
    now: datetime | None = None,
    db_path: str | Path = DB_PATH,
) -> None:
    """Record a reviewer's sign-off: save their edit to the held draft and log who did what.

    Only a draft held for approval is ever rewritten; a suppressed draft stays as it is.
    """
    if edited_text is not None and not edited_text.strip():
        raise ValueError("The edited draft is empty; pass None to keep the draft as it is.")
    approved_at = (to_local(now) if now is not None else now_local()).isoformat()
    with closing(_connect(db_path)) as conn, conn:
        # Take the write lock before checking, so two reviewers cannot both approve one case.
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status FROM cases WHERE id = ?", (case_id,)).fetchone()
        if row is None:
            raise KeyError(f"No case with id {case_id}")
        if row["status"] != CaseStatus.AWAITING_HUMAN.value:
            raise ValueError(f"Case {case_id} is {row['status']}, not awaiting a human.")
        edited = _update_held_draft(conn, case_id, edited_text)
        conn.execute(
            "UPDATE cases SET status = ?, approved_at = ?, reviewer_note = ? WHERE id = ?",
            (CaseStatus.APPROVED.value, approved_at, note, case_id),
        )
        detail = json.dumps({"note": note, "draft_edited": edited})
        conn.execute(
            "INSERT INTO steps(case_id, seq, name, detail, created_at) "
            "SELECT ?, COALESCE(MAX(seq), 0) + 1, 'human_approved', ?, ? "
            "FROM steps WHERE case_id = ?",
            (case_id, detail, approved_at, case_id),
        )


def _update_held_draft(conn: sqlite3.Connection, case_id: str, edited_text: str | None) -> bool:
    """Replace the held draft's text with the reviewer's version; True if a draft changed."""
    if edited_text is None:
        return False
    cursor = conn.execute(
        "UPDATE drafts SET text = ? WHERE case_id = ? AND send_mode = ? AND text != ?",
        (edited_text, case_id, SendMode.HELD_FOR_APPROVAL.value, edited_text),
    )
    return cursor.rowcount > 0
