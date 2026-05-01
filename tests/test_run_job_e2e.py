"""End-to-end regression for the user-triggered research job flow.

This keeps OpenRouter and GitHub mocked, but exercises the public FastAPI
entrypoint, persisted job card, completion events, and publish metadata shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import jobs
from app.main import app


@dataclass
class FakeResult:
    title: str = "E2E CancerHawk Block"
    market_price: float = 0.72
    block: int = 7
    result_url: str = "/results/block-7/paper.html"
    stats: dict = None
    calls: list = None
    git_status: str = "hermes pushed block 7; deploy hook skipped"

    def __post_init__(self):
        self.stats = self.stats or {
            "total_calls": 2,
            "total_tokens": 300,
            "total_cost_usd": 0.0012,
            "elapsed_seconds": 4,
        }
        self.calls = self.calls or []


class FakeSupervisor:
    def __init__(self, *, emit, on_call, tracker):
        self.emit = emit
        self.on_call = on_call
        self.tracker = tracker

    async def run(self, cfg):
        await self.emit("paper_done", "Paper compiled", {"title": "E2E CancerHawk Block"})
        await self.emit("publish_done", "Hermes wrote block 7 -> results/block-7", {"block": 7})
        await self.emit("git", "hermes pushed block 7; deploy hook skipped", {"status": "ok"})
        return FakeResult()


def test_start_job_creates_live_card_and_completes(tmp_path):
    test_jobs_file = tmp_path / "jobs.json"
    payload = {
        "api_key": "sk-test",
        "research_goal": "Trace an end-to-end oncology block publish",
        "n_submitters": 2,
        "auto_publish": True,
        "git_push": True,
        "submitter": "openrouter/free",
        "validator": "openrouter/free",
        "compiler": "openrouter/free",
        "archetype": "openrouter/free",
        "topic_deriver": "openrouter/free",
    }

    with (
        patch.object(jobs, "JOBS_FILE", test_jobs_file),
        patch("app.main.HermesSupervisor", FakeSupervisor),
    ):
        client = TestClient(app)
        response = client.post("/api/jobs/start", json=payload)
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        job_response = client.get(f"/api/jobs/{job_id}")
        assert job_response.status_code == 200
        job = job_response.json()

    assert job["status"] == "completed"
    assert job["research_goal"] == payload["research_goal"]
    assert job["result"]["block"] == 7
    assert job["result"]["result_url"] == "/results/block-7/paper.html"
    assert "hermes pushed block 7" in job["result"]["git_status"]
    stages = [event["stage"] for event in job["events"]]
    assert stages[:2] == ["start", "hermes"]
    assert "paper_done" in stages
    assert "publish_done" in stages
    assert "git" in stages
    assert stages[-1] == "done"

