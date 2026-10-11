"""Turns a request into a validated Classification and never crashes the pipeline."""

from __future__ import annotations

import logging

from pydantic import ValidationError

from wattle.llm import LLMClient, LLMError
from wattle.models import (
    Classification,
    Extracted,
    LLMClassification,
    Product,
    Request,
    RequestType,
)
from wattle.prompts import classifier_system_prompt, classifier_user_prompt

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2  # the first try plus one retry


def classify(req: Request, llm: LLMClient) -> Classification:
    """Ask the LLM to classify; on repeated failure return a safe result instead of raising."""
    system = classifier_system_prompt()
    user = classifier_user_prompt(req)
    last_error = "unknown error"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            output = llm.generate_json(system, user, LLMClassification)
            return Classification(**output.model_dump())
        except (LLMError, ValidationError) as exc:
            last_error = str(exc)
            logger.warning("Classify attempt %d failed for %s: %s", attempt, req.id, exc)
    return _error_classification(last_error)


def _error_classification(error: str) -> Classification:
    """The fallback the rules send to human review (R0): nothing is guessed."""
    return Classification(
        request_type=RequestType.UNCLEAR,
        product=Product.UNKNOWN,
        risk_flags=[],
        extracted=Extracted(summary="Not classified because the LLM call failed."),
        missing_info=[],
        confidence=0.0,
        rationale=f"LLM error: {error}",
        llm_error=True,
    )
