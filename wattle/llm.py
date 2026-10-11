"""LLM access behind one small interface so providers can be swapped and tests stay offline."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ValidationError

from wattle.config import CACHE_DIR, LLM_MODEL

logger = logging.getLogger(__name__)

Responder = Callable[[str, str, type[BaseModel]], BaseModel]


class LLMError(Exception):
    """The one failure type callers catch, whatever the provider raised underneath."""


class LLMClient(Protocol):
    """Anything that can turn a prompt into a validated instance of `schema`."""

    def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel: ...


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
        except (ValidationError, OSError) as exc:
            logger.warning("Ignoring bad LLM cache entry %s: %s", path.name, exc)
            return None

    def _write(self, path: Path, result: BaseModel) -> None:
        """Write to a temp file then swap, so an interrupted run cannot leave half an entry."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)


class GeminiClient:
    """Real provider; implemented in M2."""

    def __init__(self, model: str = LLM_MODEL) -> None:
        self.model = model

    def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        """Placeholder so the interface exists before the real call is written."""
        raise NotImplementedError("GeminiClient is implemented in milestone M2.")
