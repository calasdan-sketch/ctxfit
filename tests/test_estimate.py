"""Tests for token estimation."""

import pytest

from ctxfit.estimate import (
    HeuristicEstimator,
    TiktokenEstimator,
    count_by_section,
    get_estimator,
    top_consumers,
)
from ctxfit.messages import Message


@pytest.fixture
def est():
    return HeuristicEstimator()


def test_empty_counts_zero(est):
    assert est.count("") == 0


def test_prose_heuristic_four_chars_per_token(est):
    # 400 latin chars -> ~100 tokens
    assert est.count("a" * 400) == 100


def test_cjk_counts_heavier(est):
    latin = est.count("a" * 40)  # 10
    cjk = est.count("漢" * 40)
    assert cjk > latin  # 40 CJK chars must exceed 10 tokens


def test_mixed_text(est):
    n = est.count("hello world, this is a test.")
    assert 5 <= n <= 12


def test_count_message_adds_overhead(est):
    m = Message(role="user", content="hi")
    assert est.count_message(m) == est.count("hi") + 4


def test_get_estimator_auto_falls_back_to_heuristic():
    # tiktoken is not installed in this env
    e = get_estimator("auto")
    assert isinstance(e, HeuristicEstimator)


def test_get_estimator_heuristic():
    assert isinstance(get_estimator("heuristic"), HeuristicEstimator)


def test_get_estimator_tiktoken_missing_raises():
    with pytest.raises(ImportError, match="tiktoken"):
        get_estimator("tiktoken")


def test_get_estimator_rejects_unknown():
    with pytest.raises(ValueError, match="unknown estimator"):
        get_estimator("gpt-5")


def test_get_estimator_accepts_custom():
    class Mine:
        name = "mine"

        def count(self, text):
            return 42

        def count_message(self, msg):
            return 42

    assert get_estimator(Mine()).count("x") == 42


def test_get_estimator_rejects_garbage():
    with pytest.raises(TypeError):
        get_estimator(123)


def test_tiktoken_estimator_import_error_message():
    with pytest.raises(ImportError, match="pip install tiktoken"):
        TiktokenEstimator()


def test_count_by_section(est):
    msgs = [
        Message(role="system", content="sys", section="system"),
        Message(role="tool", content="out", kind="tool_result", section="retrieved"),
        Message(role="user", content="hi", section="history"),
    ]
    usage = count_by_section(msgs, est)
    assert usage["system"] == est.count_message(msgs[0])
    assert usage["retrieved"] == est.count_message(msgs[1])
    assert usage["history"] == est.count_message(msgs[2])
    assert usage["scratchpad"] == 0


def test_top_consumers_sorted(est):
    msgs = [
        Message(role="user", content="a"),
        Message(role="user", content="b" * 1000),
        Message(role="user", content="c" * 100),
    ]
    top = top_consumers(msgs, est, n=2)
    assert len(top) == 2
    assert top[0]["index"] == 1
    assert top[1]["index"] == 2
    assert top[0]["est_tokens"] >= top[1]["est_tokens"]
    assert "preview" in top[0] and "section" in top[0]
