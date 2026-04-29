"""Tests for the full MOTO paper engine: convergence detector + adaptive loop.

Includes regression tests for bugs found during code review:
  - safety_guard_fires_below_floor: API-call/wall-clock guards must fire
    even when MIN_ACCEPTED_FLOOR isn't met (else permanent failures loop
    forever).
  - submitter_prompt_handles_empty_aggregate: prior_accepted entries that
    are empty/whitespace must not crash prompt construction.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app import paper_engine
from app.paper_engine import (
    MAX_API_CALLS_SOFT,
    MAX_WALL_CLOCK_SECONDS,
    MIN_ACCEPTED_FLOOR,
    PLATEAU_ROUNDS,
    Paper,
    SATURATION_ROUNDS,
    _check_convergence,
    run_paper_engine,
)
from app.prompts import submitter_prompt
from app.token_tracker import TokenTracker


# --- _check_convergence: pure function tests ---------------------------------

class TestCheckConvergence:
    def test_below_floor_does_not_stop(self):
        stop, reason = _check_convergence(
            accepted_count=MIN_ACCEPTED_FLOOR - 1,
            rounds_run=10,
            accepts_per_round=[0] * 10,
            novelty_per_round=[1.0] * 10,
            api_calls=10,
            elapsed_s=10.0,
        )
        assert stop is False
        assert reason == ""

    def test_saturation_after_floor(self):
        stop, reason = _check_convergence(
            accepted_count=MIN_ACCEPTED_FLOOR + 2,
            rounds_run=4,
            accepts_per_round=[2, 3] + [0] * SATURATION_ROUNDS,
            novelty_per_round=[5.0, 6.0, 5.0, 4.0],
            api_calls=10,
            elapsed_s=10.0,
        )
        assert stop is True
        assert "saturation" in reason

    def test_saturation_window_too_short(self):
        # Even with zero accepts, if rounds_run < SATURATION_ROUNDS we don't fire.
        stop, _ = _check_convergence(
            accepted_count=MIN_ACCEPTED_FLOOR,
            rounds_run=1,
            accepts_per_round=[0],
            novelty_per_round=[1.0],
            api_calls=10,
            elapsed_s=10.0,
        )
        assert stop is False

    def test_plateau_triggers(self):
        stop, reason = _check_convergence(
            accepted_count=MIN_ACCEPTED_FLOOR + 5,
            rounds_run=5,
            accepts_per_round=[1] * 5,
            # last PLATEAU_ROUNDS values are non-increasing
            novelty_per_round=[3.0, 4.0] + [5.0] * PLATEAU_ROUNDS,
            api_calls=10,
            elapsed_s=10.0,
        )
        # The trailing window [5,5,5,...] is non-increasing → plateau.
        assert stop is True
        assert "plateau" in reason

    def test_strictly_increasing_novelty_continues(self):
        # Strictly improving novelty must NOT plateau.
        novelty = list(range(1, PLATEAU_ROUNDS + 5))  # always increasing
        stop, _ = _check_convergence(
            accepted_count=MIN_ACCEPTED_FLOOR + 5,
            rounds_run=len(novelty),
            accepts_per_round=[1] * len(novelty),
            novelty_per_round=[float(v) for v in novelty],
            api_calls=10,
            elapsed_s=10.0,
        )
        assert stop is False

    def test_api_call_safety_guard(self):
        stop, reason = _check_convergence(
            accepted_count=MIN_ACCEPTED_FLOOR + 2,
            rounds_run=4,
            accepts_per_round=[1, 1, 1, 1],
            novelty_per_round=[5.0, 6.0, 7.0, 8.0],
            api_calls=MAX_API_CALLS_SOFT + 5,
            elapsed_s=10.0,
        )
        assert stop is True
        assert "api_calls" in reason

    def test_wall_clock_safety_guard(self):
        stop, reason = _check_convergence(
            accepted_count=MIN_ACCEPTED_FLOOR + 2,
            rounds_run=4,
            accepts_per_round=[1, 1, 1, 1],
            novelty_per_round=[5.0, 6.0, 7.0, 8.0],
            api_calls=10,
            elapsed_s=MAX_WALL_CLOCK_SECONDS + 1,
        )
        assert stop is True
        assert "wall_clock" in reason

    def test_safety_guard_fires_below_floor_REGRESSION(self):
        """Regression: previously the floor check returned early before the
        safety guards, so a run with zero acceptances and exhausted API
        budget would loop forever. Safety guards must take precedence.
        """
        stop, reason = _check_convergence(
            accepted_count=0,  # well below floor
            rounds_run=50,
            accepts_per_round=[0] * 50,
            novelty_per_round=[0.0] * 50,
            api_calls=MAX_API_CALLS_SOFT + 100,
            elapsed_s=10.0,
        )
        assert stop is True, "safety guard must fire even when floor is unmet"
        assert "api_calls" in reason

        stop, reason = _check_convergence(
            accepted_count=0,
            rounds_run=50,
            accepts_per_round=[0] * 50,
            novelty_per_round=[0.0] * 50,
            api_calls=10,
            elapsed_s=MAX_WALL_CLOCK_SECONDS + 100,
        )
        assert stop is True
        assert "wall_clock" in reason

    def test_disabled_safety_guards(self, monkeypatch):
        """When MAX_API_CALLS_SOFT or MAX_WALL_CLOCK_SECONDS is 0, that
        guard is disabled."""
        monkeypatch.setattr(paper_engine, "MAX_API_CALLS_SOFT", 0)
        monkeypatch.setattr(paper_engine, "MAX_WALL_CLOCK_SECONDS", 0)
        stop, _ = _check_convergence(
            accepted_count=MIN_ACCEPTED_FLOOR + 1,
            rounds_run=1,
            accepts_per_round=[1],
            novelty_per_round=[5.0],
            api_calls=10**9,
            elapsed_s=10**9,
        )
        # No saturation/plateau triggered, guards disabled, floor met → continue.
        assert stop is False


# --- submitter_prompt aggregation -------------------------------------------

class TestSubmitterPromptAggregation:
    def test_no_prior_accepted_no_aggregate_block(self):
        msgs = submitter_prompt("cure cancer", [])
        assert "ALREADY-ACCEPTED" not in msgs[1]["content"]

    def test_prior_accepted_injects_aggregate(self):
        msgs = submitter_prompt(
            "cure cancer",
            [],
            prior_accepted=["KRAS G12C inhibitor + autophagy block",
                            "CAF-targeted IL6 axis"],
        )
        content = msgs[1]["content"]
        assert "ALREADY-ACCEPTED" in content
        assert "KRAS" in content
        assert "CAF-targeted" in content

    def test_previous_block_context_is_available_for_citation(self):
        msgs = submitter_prompt(
            "extend cell cinema",
            [],
            previous_block_context=(
                "[CancerHawk Block 1] Cell Cinema\n"
                "URL: https://asimog.github.io/cancerhawk/block-1/paper.html\n"
                "Useful prior findings: latent trajectories"
            ),
        )
        content = msgs[1]["content"]
        assert "PREVIOUS CANCERHAWK BLOCKS" in content
        assert "CancerHawk Block 1" in content
        assert "cite" in content.lower()

    def test_prior_rejections_appear(self):
        msgs = submitter_prompt(
            "cure cancer",
            ["too vague", "no falsifier"],
        )
        assert "PRIOR REJECTIONS" in msgs[1]["content"]
        assert "too vague" in msgs[1]["content"]

    def test_empty_submission_in_aggregate_does_not_crash_REGRESSION(self):
        """Regression: empty/whitespace strings in prior_accepted previously
        crashed because `''.splitlines()[0]` raised IndexError."""
        msgs = submitter_prompt(
            "cure cancer",
            [],
            prior_accepted=["", "   ", "\n\n", "real submission"],
        )
        content = msgs[1]["content"]
        assert "ALREADY-ACCEPTED" in content
        assert "real submission" in content
        assert content.count("(empty submission)") == 3

    def test_only_first_line_of_each_submission_used(self):
        msgs = submitter_prompt(
            "g",
            [],
            prior_accepted=["FIRST LINE\nsecondary content not shown"],
        )
        content = msgs[1]["content"]
        assert "FIRST LINE" in content
        assert "secondary content not shown" not in content

    def test_long_first_line_is_truncated_to_200(self):
        long_line = "X" * 500
        msgs = submitter_prompt("g", [], prior_accepted=[long_line])
        content = msgs[1]["content"]
        # 200-char snippet only.
        assert "X" * 200 in content
        assert "X" * 201 not in content


# --- run_paper_engine integration (mocked LLM) ------------------------------

class TestRunPaperEngineIntegration:
    @pytest.mark.asyncio
    async def test_converges_via_saturation(self, monkeypatch):
        """Submitter always succeeds; validator accepts the first 3 then
        rejects the rest → after SATURATION_ROUNDS empty rounds, stop."""
        # Force a tight floor so the test is fast.
        monkeypatch.setattr(paper_engine, "MIN_ACCEPTED_FLOOR", 3)
        monkeypatch.setattr(paper_engine, "SATURATION_ROUNDS", 2)
        monkeypatch.setattr(paper_engine, "PLATEAU_ROUNDS", 99)  # disable
        monkeypatch.setattr(paper_engine, "MAX_API_CALLS_SOFT", 0)
        monkeypatch.setattr(paper_engine, "MAX_WALL_CLOCK_SECONDS", 0)

        accept_counter = {"n": 0}

        async def fake_chat(api_key, model, messages, **kwargs):
            role = kwargs.get("role", "")
            if role == "submitter":
                return f"submission #{accept_counter['n']}"
            if role == "compiler_section":
                return "Section body."
            return "ok"

        async def fake_chat_json(api_key, model, messages, **kwargs):
            role = kwargs.get("role", "")
            if role == "validator":
                accept_counter["n"] += 1
                accept = accept_counter["n"] <= 3
                return {
                    "accept": accept,
                    "scores": {"novelty": 7 if accept else 3},
                    "reason": "ok" if accept else "duplicate",
                    "steering_feedback": "" if accept else "be more specific",
                }
            if role == "compiler_outline":
                return {
                    "title": "Test Paper",
                    "sections": [
                        {"heading": "1. Intro", "summary": "intro"},
                        {"heading": "2. Mechanism", "summary": "mech"},
                    ],
                }
            return {}

        with patch.object(paper_engine, "chat", side_effect=fake_chat), \
             patch.object(paper_engine, "chat_json", side_effect=fake_chat_json):
            emit = AsyncMock()
            on_call = AsyncMock()
            tracker = TokenTracker()
            paper = await run_paper_engine(
                api_key="sk-test",
                research_goal="cure pancreatic cancer",
                models={"submitter": "m", "validator": "m", "compiler": "m"},
                n_submitters=2,
                emit=emit,
                tracker=tracker,
                on_call=on_call,
            )

        assert isinstance(paper, Paper)
        assert paper.title == "Test Paper"
        assert len(paper.sections) == 2
        assert len(paper.accepted_submissions) >= 3
        assert "saturation" in paper.convergence_reason
        assert paper.rounds_run >= 2

    @pytest.mark.asyncio
    async def test_no_acceptances_raises_after_safety_guard(self, monkeypatch):
        """Regression: when nothing is ever accepted AND the safety guard
        eventually fires, the engine must surface a clear error rather
        than spinning forever."""
        monkeypatch.setattr(paper_engine, "MIN_ACCEPTED_FLOOR", 3)
        # Tight call budget so the safety guard fires quickly.
        monkeypatch.setattr(paper_engine, "MAX_API_CALLS_SOFT", 4)
        monkeypatch.setattr(paper_engine, "MAX_WALL_CLOCK_SECONDS", 0)
        monkeypatch.setattr(paper_engine, "SATURATION_ROUNDS", 99)
        monkeypatch.setattr(paper_engine, "PLATEAU_ROUNDS", 99)

        async def fake_chat(api_key, model, messages, **kwargs):
            # Record into tracker so api_calls increments.
            tracker = kwargs.get("tracker")
            if tracker is not None:
                tracker.record(role=kwargs.get("role", "x"), model=model,
                               prompt_tokens=10, completion_tokens=5,
                               latency_ms=10, ok=True)
            return "submission"

        async def fake_chat_json(api_key, model, messages, **kwargs):
            tracker = kwargs.get("tracker")
            if tracker is not None:
                tracker.record(role=kwargs.get("role", "x"), model=model,
                               prompt_tokens=10, completion_tokens=5,
                               latency_ms=10, ok=True)
            return {"accept": False, "scores": {"novelty": 1},
                    "reason": "no", "steering_feedback": "no"}

        with patch.object(paper_engine, "chat", side_effect=fake_chat), \
             patch.object(paper_engine, "chat_json", side_effect=fake_chat_json):
            emit = AsyncMock()
            on_call = AsyncMock()
            tracker = TokenTracker()
            with pytest.raises(RuntimeError, match="No submissions were accepted"):
                await run_paper_engine(
                    api_key="sk-test",
                    research_goal="g",
                    models={"submitter": "m", "validator": "m", "compiler": "m"},
                    n_submitters=2,
                    emit=emit,
                    tracker=tracker,
                    on_call=on_call,
                )
