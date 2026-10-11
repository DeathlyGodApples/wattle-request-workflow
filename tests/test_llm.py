"""Tests for the LLM layer: the fake is predictable and the cache avoids repeat calls."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors
from pydantic import BaseModel

from wattle.llm import CachedLLM, FakeLLM, GeminiClient, LLMError


class Answer(BaseModel):
    """Small schema standing in for a real response model."""

    text: str


class OtherAnswer(BaseModel):
    """Second schema with the same fields, to prove the schema name is in the cache key."""

    text: str


def echo_schema(system: str, user: str, schema: type[BaseModel]) -> BaseModel:
    """Responder that builds whichever schema was asked for."""
    return schema(text="fresh")


def test_fake_llm_matches_on_substring_of_user_text() -> None:
    llm = FakeLLM({"card": Answer(text="card reply"), "loan": Answer(text="loan reply")})
    assert llm.generate_json("sys", "my loan is late", Answer) == Answer(text="loan reply")


def test_fake_llm_accepts_a_callable() -> None:
    llm = FakeLLM(lambda system, user, schema: Answer(text=user.upper()))
    assert llm.generate_json("sys", "hello", Answer) == Answer(text="HELLO")


def test_fake_llm_falls_back_to_default() -> None:
    llm = FakeLLM({"card": Answer(text="card reply")}, default=Answer(text="fallback"))
    assert llm.generate_json("sys", "something else", Answer) == Answer(text="fallback")


def test_fake_llm_without_a_match_raises_llm_error() -> None:
    llm = FakeLLM({"card": Answer(text="card reply")})
    with pytest.raises(LLMError):
        llm.generate_json("sys", "something else", Answer)


def test_fake_llm_raises_programmed_exceptions() -> None:
    llm = FakeLLM({"boom": LLMError("rate limited")})
    with pytest.raises(LLMError, match="rate limited"):
        llm.generate_json("sys", "boom", Answer)


def test_fake_llm_rejects_a_response_of_the_wrong_schema() -> None:
    llm = FakeLLM({"card": Answer(text="card reply")})
    with pytest.raises(LLMError, match="OtherAnswer"):
        llm.generate_json("sys", "card", OtherAnswer)


def test_fake_llm_returns_a_fresh_copy_each_call() -> None:
    llm = FakeLLM(default=Answer(text="ok"))
    first = llm.generate_json("sys", "one", Answer)
    first.text = "mutated"
    assert llm.generate_json("sys", "two", Answer) == Answer(text="ok")


def test_fake_llm_records_calls() -> None:
    llm = FakeLLM(default=Answer(text="ok"))
    llm.generate_json("sys", "one", Answer)
    llm.generate_json("sys", "two", Answer)
    assert [call.user for call in llm.calls] == ["one", "two"]


def test_cache_miss_calls_inner_then_hit_does_not(tmp_path: Path) -> None:
    inner = FakeLLM(default=Answer(text="fresh"))
    cached = CachedLLM(inner, model="m1", cache_dir=tmp_path)

    first = cached.generate_json("sys", "question", Answer)
    second = cached.generate_json("sys", "question", Answer)

    assert first == second == Answer(text="fresh")
    assert len(inner.calls) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_cache_survives_a_new_wrapper(tmp_path: Path) -> None:
    first_inner = FakeLLM(default=Answer(text="fresh"))
    CachedLLM(first_inner, model="m1", cache_dir=tmp_path).generate_json("sys", "q", Answer)

    second_inner = FakeLLM(default=Answer(text="should not be used"))
    result = CachedLLM(second_inner, model="m1", cache_dir=tmp_path).generate_json(
        "sys", "q", Answer
    )

    assert result == Answer(text="fresh")
    assert second_inner.calls == []


@pytest.mark.parametrize(
    ("model", "system", "user", "schema"),
    [
        ("m2", "sys", "question", Answer),
        ("m1", "other sys", "question", Answer),
        ("m1", "sys", "other question", Answer),
        ("m1", "sys", "question", OtherAnswer),
    ],
)
def test_cache_key_changes_with_each_input(
    tmp_path: Path, model: str, system: str, user: str, schema: type[BaseModel]
) -> None:
    inner = FakeLLM(echo_schema)
    CachedLLM(inner, model="m1", cache_dir=tmp_path).generate_json("sys", "question", Answer)
    CachedLLM(inner, model=model, cache_dir=tmp_path).generate_json(system, user, schema)
    assert len(inner.calls) == 2


def test_cache_does_not_store_failures(tmp_path: Path) -> None:
    cached = CachedLLM(FakeLLM({"q": LLMError("down")}), model="m1", cache_dir=tmp_path)
    with pytest.raises(LLMError):
        cached.generate_json("sys", "q", Answer)
    assert list(tmp_path.glob("*.json")) == []


def test_corrupt_cache_entry_is_replaced_not_fatal(tmp_path: Path) -> None:
    inner = FakeLLM(default=Answer(text="fresh"))
    cached = CachedLLM(inner, model="m1", cache_dir=tmp_path)
    cached.generate_json("sys", "question", Answer)
    (entry,) = tmp_path.glob("*.json")
    entry.write_text('{"text": "tru', encoding="utf-8")

    assert cached.generate_json("sys", "question", Answer) == Answer(text="fresh")
    assert len(inner.calls) == 2
    assert Answer.model_validate_json(entry.read_text(encoding="utf-8")) == Answer(text="fresh")
    assert list(tmp_path.glob("*.tmp")) == []


class FakeGeminiSDK:
    """Stands in for the google-genai client: plays back a scripted list of outcomes."""

    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[dict] = []
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, **kwargs: object) -> SimpleNamespace:
        self.requests.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


def api_error(code: int) -> errors.APIError:
    """Build the SDK's own error type so the retry logic sees what it would see for real."""
    return errors.APIError(code, {"error": {"message": f"error {code}", "status": "TEST"}})


def make_gemini(outcomes: list[object], **kwargs: object) -> tuple[GeminiClient, FakeGeminiSDK]:
    """Build a GeminiClient around the fake SDK with sleeping switched off."""
    sdk = FakeGeminiSDK(outcomes)
    kwargs.setdefault("sleep", lambda seconds: None)
    return GeminiClient("test-model", client=sdk, **kwargs), sdk


def test_gemini_returns_a_validated_schema_instance() -> None:
    client, _ = make_gemini(['{"text": "hello"}'])
    assert client.generate_json("sys", "user", Answer) == Answer(text="hello")


def test_gemini_asks_for_json_at_temperature_zero() -> None:
    client, sdk = make_gemini(['{"text": "hello"}'])
    client.generate_json("the system prompt", "the user prompt", Answer)
    (request,) = sdk.requests
    assert request["model"] == "test-model"
    assert request["contents"] == "the user prompt"
    assert request["config"].system_instruction == "the system prompt"
    assert request["config"].temperature == 0
    assert request["config"].response_mime_type == "application/json"
    assert request["config"].response_schema is Answer


@pytest.mark.parametrize("code", [429, 500, 503])
def test_gemini_retries_rate_limits_and_server_errors(code: int) -> None:
    sleeps: list[float] = []
    client, sdk = make_gemini([api_error(code), '{"text": "ok"}'], sleep=sleeps.append, rpm=0)
    assert client.generate_json("sys", "user", Answer) == Answer(text="ok")
    assert len(sdk.requests) == 2
    assert sleeps == [5]


def test_gemini_retries_network_errors() -> None:
    client, sdk = make_gemini([httpx.ConnectError("no route"), '{"text": "ok"}'])
    assert client.generate_json("sys", "user", Answer) == Answer(text="ok")
    assert len(sdk.requests) == 2


def test_gemini_turns_a_non_json_api_reply_into_llm_error() -> None:
    client, sdk = make_gemini([errors.UnknownApiResponseError("<html>proxy error</html>")] * 4)
    with pytest.raises(LLMError, match="proxy error"):
        client.generate_json("sys", "user", Answer)
    assert len(sdk.requests) == 4


def test_cache_ignores_an_entry_that_is_not_valid_utf8(tmp_path: Path) -> None:
    inner = FakeLLM(default=Answer(text="fresh"))
    cached = CachedLLM(inner, model="m1", cache_dir=tmp_path)
    cached.generate_json("sys", "question", Answer)
    (entry,) = tmp_path.glob("*.json")
    entry.write_bytes(b"\xff\xfe\x00bad")

    assert cached.generate_json("sys", "question", Answer) == Answer(text="fresh")
    assert len(inner.calls) == 2


def test_cache_write_failure_still_returns_the_answer(tmp_path: Path) -> None:
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("a file where the cache directory should be", encoding="utf-8")
    cached = CachedLLM(FakeLLM(default=Answer(text="fresh")), model="m1", cache_dir=blocker)
    assert cached.generate_json("sys", "question", Answer) == Answer(text="fresh")


def test_gemini_backs_off_exponentially_then_gives_up() -> None:
    sleeps: list[float] = []
    client, sdk = make_gemini([api_error(429)] * 4, sleep=sleeps.append, rpm=0)
    with pytest.raises(LLMError, match="429"):
        client.generate_json("sys", "user", Answer)
    assert len(sdk.requests) == 4  # first try + 3 retries
    assert sleeps == [5, 10, 20]


def test_gemini_does_not_retry_a_bad_request() -> None:
    client, sdk = make_gemini([api_error(400), '{"text": "never reached"}'])
    with pytest.raises(LLMError, match="400"):
        client.generate_json("sys", "user", Answer)
    assert len(sdk.requests) == 1


@pytest.mark.parametrize("text", ["not json", '{"wrong_field": 1}', "", None])
def test_gemini_turns_unusable_output_into_llm_error(text: str | None) -> None:
    client, _ = make_gemini([text])
    with pytest.raises(LLMError):
        client.generate_json("sys", "user", Answer)


def test_gemini_throttles_to_the_requests_per_minute_limit() -> None:
    sleeps: list[float] = []
    client, _ = make_gemini(
        ['{"text": "a"}', '{"text": "b"}'], sleep=sleeps.append, rpm=10, clock=lambda: 100.0
    )
    client.generate_json("sys", "one", Answer)
    client.generate_json("sys", "two", Answer)
    assert sleeps == [6.0]  # 60 s / 10 rpm, and the frozen clock says no time has passed


def test_gemini_without_an_api_key_fails_only_when_it_is_actually_called() -> None:
    client = GeminiClient("test-model", api_key="")  # building it is not an error
    with pytest.raises(LLMError, match="GEMINI_API_KEY is not set"):
        client.generate_json("sys", "user", Answer)


def test_no_key_and_a_cache_hit_works(tmp_path: Path) -> None:
    # Store an answer using a working client, then read it back with no key at all.
    with_key = CachedLLM(FakeLLM(echo_schema), model="m", cache_dir=tmp_path)
    stored = with_key.generate_json("sys", "cached question", Answer)

    no_key = CachedLLM(GeminiClient("m", api_key=""), model="m", cache_dir=tmp_path)

    assert no_key.generate_json("sys", "cached question", Answer) == stored


def test_no_key_and_a_cache_miss_says_the_key_is_missing(tmp_path: Path) -> None:
    no_key = CachedLLM(GeminiClient("m", api_key=""), model="m", cache_dir=tmp_path)
    with pytest.raises(LLMError, match="not in the answer cache"):
        no_key.generate_json("sys", "never asked before", Answer)
    assert list(tmp_path.iterdir()) == []  # a failure is never cached
