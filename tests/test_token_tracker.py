"""Unit tests for app.token_tracker."""

import time
from app import token_tracker
from app.token_tracker import APICall, APIFailureLimitExceeded, TokenTracker, _estimate_cost


def test_token_tracker_initial_stats():
    tracker = TokenTracker()
    s = tracker.stats()
    assert s["total_calls"] == 0
    assert s["total_input"] == 0
    assert s["total_output"] == 0
    assert s["total_tokens"] == 0
    assert s["avg_latency_ms"] == 0
    assert s["elapsed_seconds"] >= 0


def test_estimate_cost_rounding():
    # Ensure floating point arithmetic is stable
    cost = _estimate_cost("openai/gpt-4o", 123, 456)
    # (123*2.5 + 456*10)/1e6 = (307.5 + 4560)/1e6 = 4867.5/1e6 = 0.0048675
    expected = (123 * 2.50 + 456 * 10.00) / 1_000_000
    assert abs(cost - expected) < 1e-12


def test_token_tracker_record_single_call():
    tracker = TokenTracker()
    call = tracker.record(
        role="submitter",
        model="openai/gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=1234,
        ok=True,
    )
    assert tracker.total_calls == 1
    assert tracker.total_input == 100
    assert tracker.total_output == 50
    # `total_tokens` is exposed on the stats() dict (computed), not as an
    # attribute on the dataclass.
    assert tracker.stats()["total_tokens"] == 150
    assert tracker.failed_calls == 0
    assert call.seq == 1
    assert call.ok


def test_token_tracker_multiple_calls_aggregate():
    tracker = TokenTracker()
    tracker.record("a", "m1", 100, 50, 1000, True)
    tracker.record("b", "m2", 200, 100, 2000, True)
    tracker.record("a", "m1", 50, 25, 500, False)

    stats = tracker.stats()
    assert stats["total_calls"] == 3
    assert stats["failed_calls"] == 1
    assert stats["total_input"] == 350
    assert stats["total_output"] == 175
    assert stats["total_tokens"] == 525
    assert stats["by_model"]["m1"]["input"] == 150
    assert stats["by_model"]["m1"]["calls"] == 2
    assert stats["by_role"]["a"]["input"] == 150
    assert stats["by_role"]["a"]["calls"] == 2


def test_token_tracker_avg_latency():
    tracker = TokenTracker()
    tracker.record("x", "m", 100, 50, 1000, True)
    tracker.record("x", "m", 100, 50, 2000, True)
    stats = tracker.stats()
    assert stats["avg_latency_ms"] == 1500  # (1000+2000)/2


def test_token_tracker_cost_accumulation():
    tracker = TokenTracker()
    tracker.record("t", "openai/gpt-4o-mini", 1000, 500, 1000, True)  # cost = (1000*0.15+500*0.6)/1e6 = 0.00045
    tracker.record("t", "openai/gpt-4o-mini", 2000, 1000, 1000, True) # cost = (2000*0.15+1000*0.6)/1e6 = 0.0009
    stats = tracker.stats()
    assert abs(stats["total_cost_usd"] - 0.00135) < 1e-6


def test_api_call_to_dict():
    call = APICall(
        seq=1,
        timestamp=1234567890.0,
        role="validator",
        model="anthropic/claude-haiku-4.5",
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=500,
        cost_usd=0.001,
        ok=True,
        error=None,
        prompt_messages=[{"role": "user", "content": "Hello"}],
        response_text="Hi",
    )
    d = call.to_dict()
    assert d["seq"] == 1
    assert d["total_tokens"] == 150
    assert d["cost_usd"] == 0.001
    assert d["ok"] is True
    assert "Hello" in d["prompt"]
    assert "Hi" in d["response"]


def test_record_stores_bounded_prompt_and_response_previews(monkeypatch):
    monkeypatch.setattr(token_tracker, "MAX_CALL_TEXT_CHARS", 10)
    tracker = TokenTracker()

    call = tracker.record(
        role="submitter",
        model="m",
        prompt_tokens=1,
        completion_tokens=1,
        latency_ms=1,
        prompt_messages=[{"role": "user", "content": "P" * 40}],
        response_text="R" * 40,
    )

    payload = call.to_dict()
    assert len(call.prompt_messages[0]["content"]) < 40
    assert len(call.response_text) < 40
    assert "truncated" in payload["prompt"]
    assert "truncated" in payload["response"]


def test_record_caps_stored_call_list_without_losing_totals(monkeypatch):
    monkeypatch.setattr(token_tracker, "MAX_STORED_CALLS", 2)
    tracker = TokenTracker()

    for index in range(5):
        tracker.record("role", "m", index + 1, 1, 1, True)

    assert tracker.total_calls == 5
    assert tracker.stats()["total_input"] == 15
    assert [call.seq for call in tracker.calls] == [4, 5]


def test_failed_api_call_hard_limit_blocks_additional_records(monkeypatch):
    monkeypatch.setattr(token_tracker, "MAX_FAILED_API_CALLS", 2)
    tracker = TokenTracker()

    tracker.record("validator", "m", 0, 0, 1, False)
    tracker.record("validator", "m", 0, 0, 1, False)

    try:
        tracker.record("validator", "m", 0, 0, 1, True)
        assert False, "expected failure limit to stop further records"
    except APIFailureLimitExceeded as exc:
        assert exc.failed_calls == 2
        assert exc.limit == 2
