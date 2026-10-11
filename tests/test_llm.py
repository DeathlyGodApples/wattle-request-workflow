"""Tests for the LLM layer: the fake is predictable and the cache avoids repeat calls."""

from __future__ import annotations

from pathlib import Path

import pytest
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


def test_gemini_client_is_a_stub_until_m2() -> None:
    with pytest.raises(NotImplementedError):
        GeminiClient().generate_json("sys", "user", Answer)
