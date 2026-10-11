"""Central place for env-driven settings so no other module reads os.environ."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
# Tracked in git: model answers to the synthetic dev and test requests, so the demo and
# the evaluation run without an API key. It holds no secrets.
CACHE_DIR = DATA_DIR / "llm_cache"
RESULTS_DIR = ROOT_DIR / "results"

# Pinned id (not the "-latest" alias) so cached eval runs stay reproducible.
# It is also the model the committed cache and the published results were made with.
DEFAULT_LLM_MODEL = "gemini-3.5-flash-lite"

# `or` (not a getenv default) so an empty value in .env falls back to the default.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or ""
LLM_MODEL = os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD") or 0.7)
LLM_RPM = int(os.getenv("LLM_RPM") or 10)
DB_PATH = os.getenv("DB_PATH") or "wattle.db"

TIMEZONE = "Australia/Melbourne"
