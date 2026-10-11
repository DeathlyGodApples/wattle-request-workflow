"""LLM access behind one small interface so providers can be swapped and tests stay offline."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from wattle.config import CACHE_DIR, GEMINI_API_KEY, LLM_MODEL, LLM_RPM

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_SECONDS = 5  # doubles each retry: 5 s, 10 s, 20 s
REQUEST_TIMEOUT_MS = 120_000

# Everything the SDK can raise for a failed call: API errors, a non-JSON reply (for example a
# proxy's HTML error page), and network or timeout errors from its httpx transport.
RETRYABLE_SDK_ERRORS = (errors.APIError, errors.UnknownApiResponseError, httpx.HTTPError)

Responder = Callable[[str, str, type[BaseModel]], BaseModel]


class LLMError(Exception):
    """The one failure type callers catch, whatever the provider raised underneath."""


class LLMClient(Protocol):
    """Anything that can turn a prompt into a validated instance of `schema`."""

    def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        """Return the model's answer as a validated `schema` object, or raise LLMError."""
        ...


@dataclass(frozen=True)
class LLMCall:
    """A recorded FakeLLM call, so tests can assert on what was sent."""

    system: str
    user: str
    schema_name: str


class FakeLLM:
    """Offline stand-in: answers by substring of the user text, or by a callable."""

    def __init__(
        self,
        responses: Mapping[str, BaseModel | Exception] | Responder | None = None,
        *,
        default: BaseModel | None = None,
    ) -> None:
        self._responses = responses
        self._default = default
        self.calls: list[LLMCall] = []

    def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        """Return the first programmed response whose key is in the user text, else the default."""
        self.calls.append(LLMCall(system, user, schema.__name__))
        if callable(self._responses):
            return self._responses(system, user, schema)
        for needle, response in (self._responses or {}).items():
            if needle in user:
                if isinstance(response, Exception):
                    raise response
                return self._checked_copy(response, schema)
        if self._default is not None:
            return self._checked_copy(self._default, schema)
        raise LLMError(f"FakeLLM has no response programmed for: {user[:60]!r}")

    @staticmethod
    def _checked_copy(response: BaseModel, schema: type[BaseModel]) -> BaseModel:
        """Fail on a wrong-typed fixture, and copy so one test cannot mutate another's response."""
        if not isinstance(response, schema):
            raise LLMError(
                f"FakeLLM was asked for {schema.__name__} but programmed with "
                f"{type(response).__name__}"
            )
        return response.model_copy(deep=True)


class CachedLLM:
    """Wraps a client so identical prompts are answered from disk, keeping evals cheap."""

    def __init__(self, inner: LLMClient, *, model: str, cache_dir: Path = CACHE_DIR) -> None:
        self._inner = inner
        self._model = model
        self._cache_dir = cache_dir

    def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        """Serve from cache when possible; only successful responses are stored."""
        path = self._cache_dir / f"{self._key(system, user, schema)}.json"
        cached = self._read(path, schema)
        if cached is not None:
            return cached
        result = self._inner.generate_json(system, user, schema)
        self._write(path, result)
        return result

    def _key(self, system: str, user: str, schema: type[BaseModel]) -> str:
        """Hash everything that changes the answer; JSON keeps the four parts unambiguous."""
        parts = json.dumps([self._model, system, user, schema.__name__])
        return hashlib.sha256(parts.encode("utf-8")).hexdigest()

    @staticmethod
    def _read(path: Path, schema: type[BaseModel]) -> BaseModel | None:
        """Treat an unreadable or out-of-date entry as a miss so it gets replaced, not stuck."""
        if not path.exists():
            return None
        try:
            return schema.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, OSError, UnicodeDecodeError) as exc:
            logger.warning("Ignoring bad LLM cache entry %s: %s", path.name, exc)
            return None

    def _write(self, path: Path, result: BaseModel) -> None:
        """Write to a temp file then swap, so an interrupted run cannot leave half an entry."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            # The answer is already in hand; a full disk must not throw it away.
            logger.warning("Could not write LLM cache entry %s: %s", path.name, exc)


class GeminiClient:
    """Google Gemini provider: JSON mode at temperature 0, with retries and a simple throttle."""

    def __init__(
        self,
        model: str = LLM_MODEL,
        *,
        api_key: str = GEMINI_API_KEY,
        rpm: int = LLM_RPM,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if client is None and not api_key:
            raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")
        self.model = model
        self._client = client if client is not None else _build_sdk_client(api_key)
        self._min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._sleep = sleep
        self._clock = clock
        self._last_call: float | None = None

    def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        """Call Gemini and validate the reply; every failure surfaces as LLMError."""
        text = self._call_with_retries(system, user, schema)
        if not text:
            raise LLMError("Gemini returned no text (the response may have been blocked).")
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise LLMError(f"Gemini output failed validation: {exc}") from exc

    def _call_with_retries(self, system: str, user: str, schema: type[BaseModel]) -> str | None:
        """Retry rate limits, server errors and network drops; anything else fails at once."""
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            response_mime_type="application/json",
            response_schema=schema,
        )
        for attempt in range(MAX_RETRIES + 1):
            self._throttle()
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=user, config=config
                )
                return response.text
            except RETRYABLE_SDK_ERRORS as exc:
                if attempt == MAX_RETRIES or not _is_retryable(exc):
                    raise LLMError(f"Gemini call failed: {exc}") from exc
                delay = BACKOFF_SECONDS * 2**attempt
                logger.warning("Gemini call failed (%s); retrying in %ss", exc, delay)
                self._sleep(delay)
        raise LLMError("Gemini call failed: retries exhausted")  # unreachable; keeps types honest

    def _throttle(self) -> None:
        """Space calls out so a free-tier requests-per-minute limit is not hit."""
        now = self._clock()
        if self._last_call is not None:
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                self._sleep(wait)
                now += wait
        self._last_call = now


def _build_sdk_client(api_key: str) -> genai.Client:
    """Set a request timeout: the SDK's default is none, so a stalled call would hang forever."""
    options = types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
    return genai.Client(api_key=api_key, http_options=options)


def _is_retryable(exc: Exception) -> bool:
    """Rate limits (429), server errors (5xx) and network errors are worth another try."""
    if isinstance(exc, errors.APIError):
        return exc.code == 429 or 500 <= exc.code < 600
    return True
