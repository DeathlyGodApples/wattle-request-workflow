"""Tests for keywords: risky wording is caught, ordinary wording is not."""

from __future__ import annotations

import pytest

from wattle import data
from wattle.keywords import KeywordHits, scan
from wattle.models import RequestType


@pytest.mark.parametrize(
    "text",
    [
        "There is a transfer I didn't make.",
        "I did not make this payment.",
        "I don't recognise this transaction",
        "I do not recognize this charge",
        "an unauthorised withdrawal",
        "an unauthorized withdrawal",
        "I think I was scammed",
        "Is this a scam?",
        "two scams this week",
        "someone is scamming me",
        "I do not feel safe",
        "I do not see the point",
        "my card was stolen",
        "this looks like fraud",
        "a fraudulent transaction",
        "my account has been taken over",
        "I gave them the code from the text message",
        "I don't feel safe at home",
        "I don't see the point anymore",
        "he controls all our money",
        "she controls my money",
        "a loan was opened in my name",
        "I think my card was skimmed",
        "he asked for remote access to my computer",
    ],
)
def test_critical_phrases_are_caught(text: str) -> None:
    hits = scan(text)
    assert hits.critical, f"no critical hit for: {text}"


@pytest.mark.parametrize(
    "text",
    [
        "I lost my job last month",
        "I was made redundant",
        "after my redundancy",
        "I can't keep up with repayments",
        "I cant afford the repayments",
        "I cannot pay this month",
        "I can't make the minimum payment",
        "I can't work for three months",
        "I am in financial hardship",
        "I'm behind on my loan",
        "I'm on reduced hours",
        "my reduced pay doesn't cover it",
        "I had my hours cut",
        "I am on Centrelink",
        "I'm between jobs at the moment",
    ],
)
def test_hardship_phrases_are_caught(text: str) -> None:
    hits = scan(text)
    assert hits.hardship, f"no hardship hit for: {text}"


@pytest.mark.parametrize(
    "text",
    [
        "What is your SWIFT code?",
        "Please send a replacement card to my new address.",
        "Can I open a savings account online?",
        "",
    ],
)
def test_ordinary_text_has_no_critical_hits(text: str) -> None:
    assert scan(text).critical == []


def test_matching_is_whole_word_only() -> None:
    assert scan("the scampi was fine").critical == []
    assert scan("a hardshipping label").hardship == []


def test_matching_ignores_case() -> None:
    assert scan("MY CARD WAS STOLEN").critical == ["stolen"]


@pytest.mark.parametrize("apostrophe", ["\u2019", "\u2018", "\u02bc", "\u00b4", "`"])
def test_apostrophe_look_alikes_are_treated_as_straight(apostrophe: str) -> None:
    assert scan(f"a payment I didn{apostrophe}t make").critical == ["didn't make"]


@pytest.mark.parametrize("gap", ["  ", "\n", " \n ", "\t"])
def test_a_line_break_or_extra_space_inside_a_phrase_still_matches(gap: str) -> None:
    assert scan(f"a payment I did{gap}not{gap}make").critical == ["did not make"]
    assert scan(f"I lost{gap}my{gap}job").hardship == ["lost my job"]


@pytest.mark.parametrize(
    ("text", "hit"),
    [
        ("I don't recognise this", "don't recognise"),
        ("an unauthorized payment", "unauthorized"),
        ("he controls all our money", "controls all our money"),
        ("I gave him the code", "gave him the code"),
        ("I can't keep up", "can't keep up"),
        ("my hours reduced again", "hours reduced"),
    ],
)
def test_the_hit_is_the_phrase_that_matched(text: str, hit: str) -> None:
    assert scan(text).all() == [hit]


def test_hits_report_the_matched_words_once_each() -> None:
    hits = scan("Stolen! My card was stolen and I think it is fraud.")
    assert hits.critical == ["stolen", "fraud"]
    assert hits.hardship == []


def test_a_message_can_hit_both_lists() -> None:
    hits = scan("I lost my job and now there is a transaction I don't recognise.")
    assert hits.critical == ["don't recognise"]
    assert hits.hardship == ["lost my job"]


def test_all_hits_lists_critical_first() -> None:
    hits = KeywordHits(critical=["stolen"], hardship=["lost my job"])
    assert hits.all() == ["stolen", "lost my job"]


def test_dev_set_has_no_keyword_false_alarms() -> None:
    """Tuning guard, dev set only: keywords must not fire on the wrong kind of request."""
    for row in data.load_dev():
        hits = scan(f"{row.subject}\n{row.body}")
        if row.expected_type is not RequestType.CRITICAL:
            assert hits.critical == [], f"{row.id}: unexpected critical hit {hits.critical}"
        if row.expected_type not in (RequestType.HARDSHIP, RequestType.CRITICAL):
            assert hits.hardship == [], f"{row.id}: unexpected hardship hit {hits.hardship}"
