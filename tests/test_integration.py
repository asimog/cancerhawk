"""Integration tests for WebSocket run endpoint."""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
client = TestClient(app)


@pytest.fixture
def mock_engines():
    """Patch all heavy engine functions and optionally simulate API calls."""

    async def fake_block_race(*args, **kwargs):
        # Simulate the block-race LLM calls to exercise token tracking
        tracker = kwargs.get("tracker")
        on_call = kwargs.get("on_call")
        if tracker and on_call:
            c1 = tracker.record(
                role="moto_worker:test",
                model="openai/gpt-4o-mini",
                prompt_tokens=100,
                completion_tokens=50,
                latency_ms=800,
                ok=True,
            )
            await on_call(c1)
            c2 = tracker.record(
                role="miroshark_review:test",
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

        class FakeCandidate:
            index = 1
            topic = {"id": 1, "title": "Test topic", "research_goal": "Test goal"}
            worker_id = "moto_worker_01"
            worker_wallet = {
                "agent_id": "moto_worker_01",
                "role": "moto_worker",
                "wallet_address": "WorkerWallet111111111111111111111111111111",
                "wallet_type": "subagent-ledger",
            }
            paper = FakePaper()

        review = {
            "archetype_id": "test",
            "archetype_name": "MiroShark TestReviewer",
            "reviewer_id": "miroshark_reviewer_01",
            "reviewer_wallet": {
                "agent_id": "miroshark_reviewer_01",
                "role": "miroshark_reviewer",
                "wallet_address": "ReviewerWallet11111111111111111111111111",
                "wallet_type": "subagent-ledger",
            },
            "candidate_index": 1,
            "candidate_title": "Test Paper",
            "recommendation": "accept",
            "confidence": 0.9,
            "overall_score": 8.0,
            "summary": "Good",
            "dimension_scores": {
                "mechanistic_plausibility": 5,
                "experimental_design": 5,
                "evidence_support": 5,
                "statistical_rigor": 5,
                "clarity_of_writing": 5,
            },
            "criticisms": [],
            "required_fixes": [],
            "suggested_experiments": [],
            "simulation_proposal": {
                "type": "statistical",
                "description": "Run a bootstrap survival analysis.",
                "rationale": "Check robustness of the claimed effect.",
                "expected_metrics": ["hazard_ratio", "confidence_interval"],
            },
        }

        class FakeRace:
            topics = [FakeCandidate.topic]
            base_metadata = {"source_repo": "https://github.com/Precigenetic/CancerHawk"}
            candidates = [FakeCandidate()]
            reviews = [review]
            validator_decision = {"winner_index": 1, "selection_rationale": "Best test candidate."}
            winner = candidates[0]
            next_topics = [
                {
                    "id": index,
                    "title": f"Next topic {index}",
                    "probability": 0.6,
                    "impact": 7,
                    "token_cost": 4000,
                    "rationale": "Keep the run deterministic.",
                }
                for index in range(1, 11)
            ]
            subagent_wallets = {
                "moto_worker_01": candidates[0].worker_wallet,
                "miroshark_reviewer_01": review["reviewer_wallet"],
                "moto_validator_01": {
                    "agent_id": "moto_validator_01",
                    "role": "moto_validator",
                    "wallet_address": "ValidatorWallet1111111111111111111111111",
                    "wallet_type": "subagent-ledger",
                },
            }

            def race_metadata(self):
                return {
                    "base_layer": self.base_metadata,
                    "topics": self.topics,
                    "candidate_count": len(self.candidates),
                    "review_count": len(self.reviews),
                    "winner_index": self.winner.index,
                    "validator_decision": self.validator_decision,
                    "subagent_wallets": self.subagent_wallets,
                }

        return FakeRace()

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

    def fake_publish_block(*args, **kwargs):
        simulations = kwargs["simulations"]
        race_metadata = kwargs["race_metadata"]
        # Dual-track output: 3 × html5_canvas + 3 × threejs.
        assert len(simulations) == 6
        canvas_sims = [s for s in simulations if s["type"] == "html5_canvas"]
        three_sims = [s for s in simulations if s["type"] == "threejs"]
        assert len(canvas_sims) == 3
        assert len(three_sims) == 3
        # Reviewer-supplied proposal still claims the first canvas slot.
        assert canvas_sims[0]["description"] == "Run a bootstrap survival analysis."
        assert all(sim.get("scene") for sim in canvas_sims)
        assert all(sim.get("three_scene") for sim in three_sims)
        assert race_metadata["winner_index"] == 1
        assert "subagent_wallets" in race_metadata
        return {"block": 1, "path": "results/block-1/paper.html"}

    def fake_try_git_publish(*args, **kwargs):
        return "ok"

    patches = [
        patch("app.hermes_supervisor.run_block_race", new=fake_block_race),
        patch("app.hermes_supervisor.run_analysis_engine", new=fake_analysis_engine),
        patch("app.hermes_supervisor.publish_block", new=fake_publish_block),
        patch("app.hermes_supervisor.try_git_publish", new=fake_try_git_publish),
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
        # stage_done, done.
        stages = [json.loads(m)["stage"] for m in messages]
        assert "start" in stages
        assert "paper_done" in stages
        assert "simulate" in stages
        assert "simulate_done" in stages
        assert "stage_done" in stages
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
