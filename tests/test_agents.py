"""Tests for the agents submission module."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from app.agents import (
    AGENT_SUBMISSIONS_DIR,
    LEADERBOARD_FILE,
    PRIZE_AMOUNT_USDC,
    estimate_run_cost,
    get_leaderboard,
    get_prompts,
    parse_agent_run,
    submit_paper,
)


# ---------------------------------------------------------------------------
# get_prompts
# ---------------------------------------------------------------------------

def test_get_prompts_returns_required_keys():
    prompts = get_prompts()
    assert "domain_frame" in prompts
    assert "engines" in prompts
    assert "openrouter_endpoint" in prompts
    assert prompts["openrouter_endpoint"] == "https://openrouter.ai/api/v1/chat/completions"


def test_get_prompts_has_four_engine_roles():
    prompts = get_prompts()
    roles = {"submitter", "validator", "archetype", "peer_reviewer"}
    assert set(prompts["engines"].keys()) == roles


def test_get_prompts_engine_has_prompt_template_and_output_format():
    for engine in get_prompts()["engines"].values():
        assert "prompt_template" in engine
        assert "output_format" in engine
        assert isinstance(engine["prompt_template"], str)
        assert isinstance(engine["output_format"], str)


# ---------------------------------------------------------------------------
# estimate_run_cost
# ---------------------------------------------------------------------------

def test_estimate_run_cost_free_model():
    result = estimate_run_cost(model="openrouter/free", n_submitters=3)
    assert result["model"] == "openrouter/free"
    assert result["free_tier_available"] is True
    assert result["estimated_cost_usd"] == 0.0
    assert result["estimated_cost_usd_display"] == "$0.000000"
    assert result["estimated_calls"] > 0
    assert result["estimated_total_tokens"] > 0


def test_estimate_run_cost_paid_model():
    result = estimate_run_cost(model="anthropic/claude-sonnet-4-20250514", n_submitters=3)
    assert result["model"] == "anthropic/claude-sonnet-4-20250514"
    assert result["free_tier_available"] is False
    assert result["estimated_cost_usd"] >= 0


def test_estimate_run_cost_default_model():
    result = estimate_run_cost(n_submitters=3)
    assert result["model"] == "openrouter/free"
    assert result["free_tier_available"] is True


def test_estimate_run_cost_n_submitters_bounds():
    result = estimate_run_cost(n_submitters=1)
    calls_1 = result["estimated_calls"]
    result = estimate_run_cost(n_submitters=8)
    calls_8 = result["estimated_calls"]
    assert calls_8 > calls_1


def test_estimate_run_cost_unknown_model_zero_price():
    result = estimate_run_cost(model="totally/fake-model-xyz")
    assert result["input_price_per_m"] == 0.0
    assert result["output_price_per_m"] == 0.0
    assert result["estimated_cost_usd"] == 0.0


def test_estimate_run_cost_clamps_n_submitters():
    result = estimate_run_cost(n_submitters=0)
    assert result["estimated_calls"] > 0
    result = estimate_run_cost(n_submitters=999)
    assert result["estimated_calls"] > 0


# ---------------------------------------------------------------------------
# submit_paper
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_leaderboard():
    if LEADERBOARD_FILE.exists():
        LEADERBOARD_FILE.unlink()
    for f in AGENT_SUBMISSIONS_DIR.glob("*.json"):
        if f.name != "leaderboard.json":
            f.unlink()
    yield
    if LEADERBOARD_FILE.exists():
        LEADERBOARD_FILE.unlink()
    for f in AGENT_SUBMISSIONS_DIR.glob("*.json"):
        if f.name != "leaderboard.json":
            f.unlink()


def test_submit_paper_minimal(clean_leaderboard):
    result = submit_paper(
        agent_name="TestAgent",
        paper_title="Test Paper",
        paper_content="# Abstract\n\nThis is a test.",
        research_goal="Test goal",
    )
    assert result["success"] is True
    assert result["submission_id"]
    assert result["total_submissions"] > 0
    assert str(PRIZE_AMOUNT_USDC) in result["prize"]


def test_submit_paper_persists_to_disk(clean_leaderboard):
    result = submit_paper(
        agent_name="DiskAgent",
        paper_title="Disk Test",
        paper_content="# Content",
        research_goal="Disk persist goal",
    )
    sid = result["submission_id"]
    sub_path = AGENT_SUBMISSIONS_DIR / f"{sid}.json"
    assert sub_path.exists()
    data = json.loads(sub_path.read_text(encoding="utf-8"))
    assert data["submission_id"] == sid
    assert data["agent_name"] == "DiskAgent"
    assert data["status"] == "received"


def test_submit_paper_with_all_optional_fields(clean_leaderboard):
    result = submit_paper(
        agent_name="FullAgent",
        paper_title="Full Paper",
        paper_content="# Complete paper",
        research_goal="Full pipeline goal",
        agent_model="claude-sonnet-4-20250514",
        peer_reviews=[{"score": 8, "review": "Good"}],
        simulations=[{"type": "2d", "data": "test"}],
        wallet_address="0x1234567890abcdef",
    )
    assert result["success"] is True
    assert result["total_submissions"] > 0


def test_submit_paper_truncates_long_content(clean_leaderboard):
    long_content = "x" * 60000
    result = submit_paper(
        agent_name="TruncAgent",
        paper_title="Truncated",
        paper_content=long_content,
        research_goal="Truncation goal",
    )
    assert result["success"] is True
    sid = result["submission_id"]
    data = json.loads((AGENT_SUBMISSIONS_DIR / f"{sid}.json").read_text(encoding="utf-8"))
    assert len(data["paper_content"]) <= 50000


# ---------------------------------------------------------------------------
# get_leaderboard
# ---------------------------------------------------------------------------

def test_get_leaderboard_starts_empty(clean_leaderboard):
    lb = get_leaderboard()
    assert isinstance(lb, list)
    assert len(lb) == 0


def test_get_leaderboard_reflects_submissions(clean_leaderboard):
    submit_paper(
        agent_name="LB1",
        paper_title="First",
        paper_content="Content",
        research_goal="Goal 1",
    )
    submit_paper(
        agent_name="LB2",
        paper_title="Second",
        paper_content="Content",
        research_goal="Goal 2",
    )
    lb = get_leaderboard()
    assert len(lb) == 2
    assert lb[0]["agent_name"] == "LB1"
    assert lb[1]["agent_name"] == "LB2"
    assert "submitted_at" in lb[0]
    assert "submission_id" in lb[0]


# ---------------------------------------------------------------------------
# parse_agent_run
# ---------------------------------------------------------------------------

def test_parse_agent_run_valid_config():
    cfg = {
        "api_key": "sk-or-v1-testkey123456",
        "research_goal": "PD-1 resistance in melanoma",
        "model": "openrouter/free",
        "n_submitters": 3,
        "agent_name": "TestRunner",
    }
    api_key, goal, model, n_sub, name, mode = parse_agent_run(cfg)
    assert api_key == "sk-or-v1-testkey123456"
    assert goal == "PD-1 resistance in melanoma"
    assert model == "openrouter/free"
    assert n_sub == 3
    assert name == "TestRunner"
    assert mode == "openrouter"


def test_parse_agent_run_missing_api_key_raises():
    cfg = {"research_goal": "test", "n_submitters": 2}
    with pytest.raises(ValueError, match="API key"):
        parse_agent_run(cfg)


def test_parse_agent_run_missing_research_goal_raises():
    cfg = {"api_key": "sk-or-v1-testkey", "n_submitters": 2}
    with pytest.raises(ValueError, match="research_goal"):
        parse_agent_run(cfg)


def test_parse_agent_run_research_goal_too_long_raises():
    cfg = {"api_key": "sk-or-v1-testkey", "research_goal": "x" * 1001, "n_submitters": 2}
    with pytest.raises(ValueError, match="1000"):
        parse_agent_run(cfg)


def test_parse_agent_run_n_submitters_out_of_bounds_raises():
    cfg = {"api_key": "sk-or-v1-testkey", "research_goal": "test", "n_submitters": 9}
    with pytest.raises(ValueError, match="between 1 and 8"):
        parse_agent_run(cfg)

    cfg["n_submitters"] = -1
    with pytest.raises(ValueError, match="between 1 and 8"):
        parse_agent_run(cfg)


def test_parse_agent_run_defaults():
    cfg = {
        "api_key": "sk-or-v1-defaults",
        "research_goal": "Default model test",
    }
    api_key, goal, model, n_sub, name, mode = parse_agent_run(cfg)
    assert model == "openrouter/free"
    assert n_sub == 3
    assert name is None
    assert mode == "openrouter"


def test_parse_agent_run_agent_name_none_when_missing():
    cfg = {
        "api_key": "sk-or-v1-test",
        "research_goal": "No name test",
        "n_submitters": 2,
    }
    _, _, _, _, name, _ = parse_agent_run(cfg)
    assert name is None


def test_parse_agent_run_falls_back_to_env_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-env-fallback")
    cfg = {"research_goal": "env key test", "n_submitters": 2}
    api_key, _, _, _, _, _ = parse_agent_run(cfg)
    assert api_key == "sk-or-v1-env-fallback"
