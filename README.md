# Wattle Bank Request Workflow

A proof-of-concept that classifies customer requests for a fictional Australian retail bank,
applies deterministic rules to decide the path, and holds anything risky for human approval.

**Core principle:** the AI classifies. Code decides. A human approves anything risky.

> Status: pipeline, evaluation and app are built (milestones M0 to M5). The full README is
> written in M6; see `CLAUDE.md` §11.

## Rule order note

Rules run in a fixed order (R0 to R4, see `wattle/rules.py`) and the first one that applies
decides. One deliberate exception to "LLM failure goes to human review": if the LLM call fails
(R0) but the message contains a critical keyword such as "stolen" or "didn't make", the request
is still treated as **critical**, and the log shows both R0 and R1. An outage must not hide a
risky message. With no critical keyword, a failed call goes to human review.

## Quick start

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env   # then add your GEMINI_API_KEY
pytest -q
streamlit run app.py
```
