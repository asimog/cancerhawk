"""Contract tests for human and agent interaction paths.

These tests keep the public participation surfaces deterministic without
calling OpenRouter, pay.sh, or any wallet/payment network.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import jobs
from app.agents import GIVEWELL_SOLANA, parse_agent_run, resolve_prize_wallet
from app.main import app


client = TestClient(app)


@dataclass
class AutoResult:
    title: str = "Auto Block"
    market_price: float = 0.8
    block: int = 3
    result_url: str = "/results/block-3/paper.html"
    calls: list = None
    git_status: str = "staged"

    def __post_init__(self):
        self.calls = self.calls or []
        self.stats = {
            "total_calls": 4,
            "total_tokens": 1200,
            "total_cost_usd": 0.0,
            "elapsed_seconds": 2,
        }


class AutoSupervisor:
    last_config = None

    def __init__(self, *, emit, on_call, tracker):
        self.emit = emit
        self.on_call = on_call
        self.tracker = tracker

    async def run(self, cfg):
        AutoSupervisor.last_config = cfg
        await self.emit("paper_done", "auto paper compiled", {"block": 3})
        return AutoResult()


def test_auto_generation_defaults_every_role_to_free_router(tmp_path, monkeypatch):
    """Production auto-blocks should not pin paid models unless explicitly set."""
    from app import main

    monkeypatch.setenv("HERMES_AUTO_GENERATE_ENABLED", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-server")
    monkeypatch.setenv("HERMES_AUTO_BLOCKS_PER_CYCLE", "1")
    for name in (
        "HERMES_MODEL_SUBMITTER",
        "HERMES_MODEL_VALIDATOR",
        "HERMES_MODEL_COMPILER",
        "HERMES_MODEL_ARCHETYPE",
        "HERMES_MODEL_TOPIC_DERIVER",
    ):
        monkeypatch.delenv(name, raising=False)

    with (
        patch.object(jobs, "JOBS_FILE", tmp_path / "jobs.json"),
        patch("app.main.HermesSupervisor", AutoSupervisor),
        patch("app.main._save_auto_block_to_db", autospec=True) as save_block,
    ):
        import asyncio

        asyncio.run(main.maybe_auto_generate())

    assert AutoSupervisor.last_config is not None
    assert AutoSupervisor.last_config.n_submitters == 3
    assert set(AutoSupervisor.last_config.models.values()) == {"openrouter/free"}
    save_block.assert_awaited_once()


def test_auto_generation_honors_explicit_operator_model_overrides(tmp_path, monkeypatch):
    from app import main

    monkeypatch.setenv("HERMES_AUTO_GENERATE_ENABLED", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-server")
    monkeypatch.setenv("HERMES_AUTO_BLOCKS_PER_CYCLE", "1")
    monkeypatch.setenv("HERMES_MODEL_VALIDATOR", "deepseek/deepseek-v4-pro")

    with (
        patch.object(jobs, "JOBS_FILE", tmp_path / "jobs.json"),
        patch("app.main.HermesSupervisor", AutoSupervisor),
        patch("app.main._save_auto_block_to_db", autospec=True),
    ):
        import asyncio

        asyncio.run(main.maybe_auto_generate())

    assert AutoSupervisor.last_config.models["validator"] == "deepseek/deepseek-v4-pro"
    assert AutoSupervisor.last_config.models["submitter"] == "openrouter/free"


def test_paysh_cancerhawk_mode_requires_server_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="server OpenRouter key"):
        parse_agent_run({
            "mode": "paysh_cancerhawk",
            "research_goal": "T-cell exhaustion reversal",
            "n_submitters": 3,
        })


def test_paysh_cancerhawk_mode_uses_server_key_when_configured(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-server")
    api_key, goal, model, n_submitters, agent_name, mode = parse_agent_run({
        "mode": "paysh_cancerhawk",
        "research_goal": "T-cell exhaustion reversal",
        "n_submitters": 3,
        "agent_name": "PayAgent",
    })

    assert api_key == "sk-or-v1-server"
    assert goal == "T-cell exhaustion reversal"
    assert model == "openrouter/free"
    assert n_submitters == 3
    assert agent_name == "PayAgent"
    assert mode == "paysh_cancerhawk"


def test_prize_wallet_accepts_solana_and_ethereum_and_rejects_junk():
    solana = "DfXygYQxEznVKtFmzVUaHbBiNyHPa1JL1y2jTnCvHRX"
    ethereum = "0x1234567890abcdef1234567890ABCDEF12345678"

    assert resolve_prize_wallet(solana) == solana
    assert resolve_prize_wallet(ethereum) == ethereum
    assert resolve_prize_wallet("not-a-wallet") == GIVEWELL_SOLANA
    assert resolve_prize_wallet(solana + "!!!!") == GIVEWELL_SOLANA


def test_agent_submit_sanitizes_invalid_wallet_and_does_not_claim_payment(tmp_path, monkeypatch):
    from app import agents

    monkeypatch.setattr(agents, "AGENT_SUBMISSIONS_DIR", tmp_path)
    monkeypatch.setattr(agents, "LEADERBOARD_FILE", tmp_path / "leaderboard.json")

    response = client.post("/api/agents/submit", json={
        "agent_name": "LocalSimAgent",
        "paper_title": "Local Simulation Paper",
        "paper_content": "# Abstract\n\nLocal simulated paper.",
        "research_goal": "local simulation",
        "wallet_address": "invalid-wallet",
    })

    assert response.status_code == 200
    payload = response.json()
    assert "submission_id" in payload
    assert "recorded" in " ".join(payload["next_steps"]).lower()
    assert "receives" not in " ".join(payload["next_steps"]).lower()

    stored = (tmp_path / f"{payload['submission_id']}.json").read_text(encoding="utf-8")
    assert GIVEWELL_SOLANA in stored


def test_websocket_bad_json_reports_error_without_job_creation(tmp_path):
    with patch.object(jobs, "JOBS_FILE", tmp_path / "jobs.json"):
        with client.websocket_connect("/ws/hermes/run") as ws:
            ws.send_text("{not-json")
            data = ws.receive_json()

    assert data["stage"] == "error"
    assert "bad config" in data["message"]
    assert not (tmp_path / "jobs.json").exists()
