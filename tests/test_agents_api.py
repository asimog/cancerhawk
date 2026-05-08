"""Tests for /api/agents/* endpoints — prompts, cost, submit, run (local mode), leaderboard."""

import json
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.agents import LEADERBOARD_FILE, AGENT_SUBMISSIONS_DIR

client = TestClient(app)


# ── GET /api/agents/prompts ────────────────────────────────

def test_prompts_returns_200_and_required_keys():
    resp = client.get("/api/agents/prompts")
    assert resp.status_code == 200
    data = resp.json()
    assert "domain_frame" in data
    assert "engines" in data
    assert "openrouter_endpoint" in data
    assert data["openrouter_endpoint"] == "https://openrouter.ai/api/v1/chat/completions"


def test_prompts_has_four_engine_roles():
    data = client.get("/api/agents/prompts").json()
    roles = {"submitter", "validator", "archetype", "peer_reviewer"}
    assert set(data["engines"].keys()) == roles


def test_prompts_engines_have_template_and_format():
    data = client.get("/api/agents/prompts").json()
    for engine in data["engines"].values():
        assert "prompt_template" in engine
        assert "output_format" in engine
        assert isinstance(engine["prompt_template"], str)
        assert isinstance(engine["output_format"], str)


# ── POST /api/agents/cost ─────────────────────────────────

def test_cost_free_model_is_coerced_to_paid_default():
    resp = client.post("/api/agents/cost", json={
        "model": "openrouter/free", "n_submitters": 3, "mode": "openrouter",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "deepseek/deepseek-v4-flash"
    assert data["free_tier_available"] is False
    assert data["estimated_cost_usd"] > 0.0


def test_cost_paid_model():
    resp = client.post("/api/agents/cost", json={
        "model": "deepseek/deepseek-v4-pro", "n_submitters": 5, "mode": "openrouter",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "deepseek/deepseek-v4-pro"
    assert data["free_tier_available"] is False


def test_cost_defaults():
    resp = client.post("/api/agents/cost", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "deepseek/deepseek-v4-flash"
    assert data["estimated_calls"] > 0


def test_cost_paysh_mode_includes_enrichment():
    resp = client.post("/api/agents/cost", json={
        "model": "openrouter/free", "n_submitters": 3, "mode": "paysh_cancerhawk",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "paysh_enrichment" in data
    assert "solana_wallet" in data


# ── POST /api/agents/submit ────────────────────────────────

@pytest.fixture
def clean_leaderboard(tmp_path, monkeypatch):
    """Keep submission API tests away from runtime results/agent_submissions."""
    import app.agents as agents_module

    monkeypatch.setattr(agents_module, "AGENT_SUBMISSIONS_DIR", tmp_path)
    monkeypatch.setattr(agents_module, "LEADERBOARD_FILE", tmp_path / "leaderboard.json")
    monkeypatch.setitem(globals(), "AGENT_SUBMISSIONS_DIR", tmp_path)
    monkeypatch.setitem(globals(), "LEADERBOARD_FILE", tmp_path / "leaderboard.json")
    yield


def test_submit_missing_agent_name_returns_400(clean_leaderboard):
    resp = client.post("/api/agents/submit", json={
        "paper_title": "Test", "paper_content": "# Test", "research_goal": "Test",
    })
    assert resp.status_code == 400


def test_submit_minimal_returns_200(clean_leaderboard):
    resp = client.post("/api/agents/submit", json={
        "agent_name": "TestAgent",
        "paper_title": "Minimal Paper",
        "paper_content": "# Abstract\n\nTest content.",
        "research_goal": "Minimal test goal",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "submission_id" in data


def test_submit_paper_content_capped_at_50k(clean_leaderboard):
    long_content = "X" * 60000
    resp = client.post("/api/agents/submit", json={
        "agent_name": "CapAgent",
        "paper_title": "Capped",
        "paper_content": long_content,
        "research_goal": "Cap test",
    })
    assert resp.status_code == 200
    data = resp.json()
    sid = data["submission_id"]
    sub_file = AGENT_SUBMISSIONS_DIR / f"{sid}.json"
    stored = json.loads(sub_file.read_text(encoding="utf-8"))
    assert len(stored["paper_content"]) <= 50000


def test_submit_defaults_to_givewell_wallet(clean_leaderboard):
    resp = client.post("/api/agents/submit", json={
        "agent_name": "NoWalletAgent",
        "paper_title": "No Wallet",
        "paper_content": "# Test",
        "research_goal": "Wallet default test",
    })
    assert resp.status_code == 200
    data = resp.json()
    sid = data["submission_id"]
    sub_file = AGENT_SUBMISSIONS_DIR / f"{sid}.json"
    stored = json.loads(sub_file.read_text(encoding="utf-8"))
    assert stored["wallet_address"] == "4Z2DBVoQCJZ42cCTDMNvYDUqRjA1C3vV7B155Mc6jGah"


# ── GET /api/agents/leaderboard ────────────────────────────

def test_leaderboard_returns_200(clean_leaderboard):
    resp = client.get("/api/agents/leaderboard")
    assert resp.status_code == 200
    data = resp.json()
    assert "submissions" in data
    assert "total" in data
    assert isinstance(data["submissions"], list)


def test_leaderboard_reflects_submission(clean_leaderboard):
    client.post("/api/agents/submit", json={
        "agent_name": "LBTest",
        "paper_title": "LB Paper",
        "paper_content": "# Test",
        "research_goal": "Leaderboard test",
    })
    data = client.get("/api/agents/leaderboard").json()
    assert data["total"] == 1
    assert data["submissions"][0]["agent_name"] == "LBTest"


# ── POST /api/agents/run (local mode) ──────────────────────

def test_run_local_mode_returns_guidance():
    resp = client.post("/api/agents/run", json={
        "research_goal": "Test local mode",
        "mode": "local",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "local"
    assert "prompts" in data["message"].lower()
    assert data["next_step"] == "GET /api/agents/prompts"


def test_run_missing_key_returns_400():
    resp = client.post("/api/agents/run", json={
        "research_goal": "Test no key",
        "mode": "openrouter",
    })
    assert resp.status_code == 400


def test_run_missing_goal_returns_400():
    resp = client.post("/api/agents/run", json={
        "api_key": "sk-or-v1-testkey1234567890abcdef123456",
        "mode": "openrouter",
    })
    assert resp.status_code == 400


# ── GET /api/models ────────────────────────────────────────

def test_models_excludes_openrouter_free():
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "openrouter/free" not in data["models"]


def test_models_includes_paid_models():
    resp = client.get("/api/models")
    data = resp.json()
    models = data["models"]
    assert any("deepseek" in m or "claude" in m or "gemini" in m for m in models)


# ── POST /api/agents/enrich ────────────────────────────────

def test_enrich_missing_goal_returns_400():
    resp = client.post("/api/agents/enrich", json={})
    assert resp.status_code == 400


def test_enrich_returns_available_endpoints():
    resp = client.post("/api/agents/enrich", json={"research_goal": "EZH2 T-cell exhaustion"})
    assert resp.status_code == 200
    data = resp.json()
    assert "available_endpoints" in data
    assert "perplexity_sonar" in data["available_endpoints"]
