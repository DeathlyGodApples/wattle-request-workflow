# Wattle Bank Request Workflow

A proof-of-concept that classifies customer requests for a fictional Australian retail bank,
applies deterministic rules to decide the path, and holds anything risky for human approval.

**Core principle:** the AI classifies. Code decides. A human approves anything risky.

> Status: scaffold only (milestone M0). The full README is written in M6; see `CLAUDE.md` §11.

## Quick start

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env   # then add your GEMINI_API_KEY
pytest -q
streamlit run app.py
```
