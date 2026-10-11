"""Orchestrator: classify, scan keywords, apply rules, run the branch.

Pure orchestration with no database access; the caller decides whether to log the result.
"""

from __future__ import annotations

from datetime import datetime

from wattle.branches import run_branch
from wattle.classifier import classify
from wattle.deadlines import now_local, to_local
from wattle.keywords import scan
from wattle.llm import LLMClient
from wattle.models import CaseResult, Request
from wattle.rules import apply_rules


def process(
    req: Request, llm: LLMClient, *, use_rules: bool = True, now: datetime | None = None
) -> CaseResult:
    """Take one request from arrival to a decided, drafted case; never raises on LLM failure."""
    received_at = to_local(now) if now is not None else now_local()
    classification = classify(req, llm)
    hits = scan(f"{req.subject}\n{req.body}")
    decision = apply_rules(classification, hits, use_rules=use_rules)
    branch = run_branch(req, classification, decision, received_at, llm)
    return CaseResult(
        request=req,
        received_at=received_at,
        classification=classification,
        decision=decision,
        branch=branch,
        keyword_hits=hits.all(),
    )
