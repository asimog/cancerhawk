"""Integration tests for WebSocket run endpoint."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.token_tracker import APICall

client = TestClient(app)


@pytest.fixture
def mock_engines():
    """Patch all heavy engine functions and optionally simulate API calls."""

    async def fake_paper_engine(*args, **kwargs):
        # Simulate a couple of LLM API calls to exercise token tracking
        emit = kwargs.get("emit")
        tracker = kwargs.get("tracker")
        on_call = kwargs.get("on_call")
        if tracker and on_call:
            # Record two fake calls
            c1 = tracker.record(
                role="submitter",
                model="openai/gpt-4o-mini",
                prompt_tokens=100,
                completion_tokens=50,
                latency_ms=800,
                ok=True,
            )
            await on_call(c1)
            c2 = tracker.record(
                role="validator",
                model="anthropic/claude-haiku-4.5",
                prompt_tokens=200,
                completion_tokens=100,
                latency_ms=1200,
                ok=True,
            )
            await on_call(c2)
        class FakePaper:
            title = "Test Paper"
            sections = [{"heading": "Intro", "content": "Introduction text"}]
            accepted_submissions = []
            rejections = []
            def full_text(self):
                return "# Test Paper\n\n## Intro\nIntroduction text"
        return FakePaper()

    async def fake_analysis_engine(*args, **kwargs):
        # Simulate one more API call
        emit = kwargs.get("emit")
        tracker = kwargs.get("tracker")
        on_call = kwargs.get("on_call")
        if tracker and on_call:
            c = tracker.record(
                role="archetype",
                model="anthropic/claude-haiku-4.5",
                prompt_tokens=300,
                completion_tokens=150,
                latency_ms=2000,
                ok=True,
            )
            await on_call(c)
        class FakeAnalysis:
            archetypes = []
            market_price = 0.75
            score_matrix = {}
            consensus_dim = {}
            headline_catalysts = []
        return FakeAnalysis()

    async def fake_peer_review_engine(*args, **kwargs):
        # One API call for the review itself
        emit = kwargs.get("emit")
        tracker = kwargs.get("tracker")
        on_call = kwargs.get("on_call")
        if tracker and on_call:
            c = tracker.record(
                role="peer_review:test",
                model="anthropic/claude-haiku-4.5",
                prompt_tokens=800,
                completion_tokens=400,
                latency_ms=3000,
                ok=True,
            )
            await on_call(c)
        from app.peer_review_engine import ConsolidatedReview, PeerReview
        review = PeerReview(
            archetype_id="test",
            archetype_name="TestReviewer",
            recommendation="accept",
            overall_confidence=0.9,
            summary="Good",
            dimension_scores={
                "mechanistic_plausibility": 5,
                "experimental_design": 5,
                "evidence_support": 5,
                "statistical_rigor": 5,
                "clarity_of_writing": 5,
            },
            criticisms=[],
            required_fixes=[],
            suggested_experiments=[],
            simulation_proposal=None,
        )
        cons = ConsolidatedReview(
            individual_reviews=[review],
            acceptance_probability=0.9,
            major_concerns=[],
            recommended_simulations=[
                {
                    "type": "statistical",
                    "description": "Run a bootstrap survival analysis.",
                    "rationale": "Check robustness of the claimed effect.",
                    "expected_metrics": ["hazard_ratio", "confidence_interval"],
                }
            ],
            revision_priorities=[],
        )
        return cons

    def fake_publish_block(*args, **kwargs):
        assert kwargs["simulations"] == [
            {
                "type": "statistical",
                "description": "Run a bootstrap survival analysis.",
                "rationale": "Check robustness of the claimed effect.",
                "expected_metrics": ["hazard_ratio", "confidence_interval"],
            }
        ]
        return {"block": 1, "path": "results/block-1/paper.html"}

    def fake_try_git_publish(*args, **kwargs):
        return "ok"

    patches = [
        patch("app.main.run_paper_engine", new=fake_paper_engine),
        patch("app.main.run_analysis_engine", new=fake_analysis_engine),
        patch("app.main.run_peer_review_engine", new=fake_peer_review_engine),
        patch("app.main.publish_block", new=fake_publish_block),
        patch("app.main.try_git_publish", new=fake_try_git_publish),
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


def test_websocket_run_success(mock_engines):
    with client.websocket_connect("/ws/run") as ws:
        # Send config
        cfg = {
            "api_key": "sk-test-fake",
            "research_goal": "Test cancer research",
            "n_submitters": 2,
            "auto_publish": True,
            "git_push": False,
            "submitter": "openai/gpt-4o-mini",
            "validator": "anthropic/claude-haiku-4.5",
            "compiler": "anthropic/claude-sonnet-4.6",
            "archetype": "anthropic/claude-haiku-4.5",
            "topic_deriver": "anthropic/claude-haiku-4.5",
        }
        ws.send_text(json.dumps(cfg))

        # Collect messages until done
        messages = []
        while True:
            msg = ws.receive_text()
            messages.append(msg)
            data = json.loads(msg)
            if data.get("stage") == "done":
                break

        # Verify we got the high-level orchestrator stages. The fake
        # engines don't emit per-engine stages (analyze/review/...) — those
        # are tested in their own engine-level unit tests. Here we only
        # assert main.py's orchestration sends start, paper_done,
        # publish_done, done.
        stages = [json.loads(m)["stage"] for m in messages]
        assert "start" in stages
        assert "paper_done" in stages
        assert "publish_done" in stages
        assert "done" in stages

        # Verify final done payload
        done_msg = json.loads(messages[-1])
        assert done_msg["stage"] == "done"
        assert "block" in done_msg["data"]
        assert "stats" in done_msg["data"]
        # Peer reviews should be included in stats? Actually stats doesn't include peer_reviews; they are separate in data payload? The done message sends stats, but not peer_reviews explicitly; peer reviews are part of publish_block result not directly in done. However, in done we include `stats`, `calls` list. That's fine.

        # Also check that api_call events appeared
        api_call_stages = [json.loads(m) for m in messages if json.loads(m).get("stage") == "api_call"]
        assert len(api_call_stages) > 0
        # Each api_call message should have data.call and data.totals
        first_call = api_call_stages[0]
        assert "data" in first_call
        assert "call" in first_call["data"]
        assert "totals" in first_call["data"]
        call_info = first_call["data"]["call"]
        assert "seq" in call_info
        assert "model" in call_info
        assert "prompt_tokens" in call_info
        assert "completion_tokens" in call_info


def test_websocket_invalid_config():
    with client.websocket_connect("/ws/run") as ws:
        ws.send_text(json.dumps({"api_key": ""}))
        msg = ws.receive_text()
        data = json.loads(msg)
        assert data["stage"] == "error"
        assert "API key required" in data["message"]
