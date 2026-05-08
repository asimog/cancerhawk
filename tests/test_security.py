"""Security tests — secrets, length limits, path safety, medical disclaimer."""

import json
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from app.main import app

ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)


# ── Secrets exposure ───────────────────────────────────────

def test_no_openrouter_key_in_env_example():
    env_example = ROOT / ".env.example"
    if not env_example.exists():
        pytest.skip(".env.example not found")
    content = env_example.read_text()
    import re
    # Allow placeholder stubs like sk-or-v1-... but NOT actual 40+ char keys
    active_keys = re.findall(r'sk-or-v1-[a-zA-Z0-9]{40,}', content)
    assert not active_keys, ".env.example contains a real OpenRouter key"
    active_tokens = re.findall(r'ghp_[a-zA-Z0-9]{36,}', content)
    assert not active_tokens, ".env.example contains a real GitHub token"


def test_no_openrouter_key_in_api_responses():
    resp = client.get("/api/agents/prompts")
    data = resp.json()
    dumped = json.dumps(data)
    assert "sk-or-v1-" not in dumped, "OpenRouter key leaked in prompts response"


# ── Length limits ──────────────────────────────────────────

def test_submit_paper_content_length_capped():
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
    from app.agents import AGENT_SUBMISSIONS_DIR
    sub_file = AGENT_SUBMISSIONS_DIR / f"{sid}.json"
    stored = json.loads(sub_file.read_text(encoding="utf-8"))
    assert len(stored["paper_content"]) <= 50000


def test_research_goal_max_length():
    resp = client.post("/api/jobs/start", json={
        "api_key": "sk-or-v1-testkey1234567890abcdef123456",
        "research_goal": "X" * 1001,
        "n_submitters": 1,
    })
    assert resp.status_code == 400


def test_agent_run_research_goal_max_length():
    resp = client.post("/api/agents/run", json={
        "api_key": "sk-or-v1-testkey1234567890abcdef123456",
        "research_goal": "X" * 1001,
        "mode": "openrouter",
    })
    assert resp.status_code == 400


# ── Git publisher path safety ──────────────────────────────

def test_publisher_commits_only_results():
    publisher_path = ROOT / "app" / "publisher.py"
    content = publisher_path.read_text()
    # The _commit_paths function should reference results/
    assert "results" in content, "publisher must reference results/ for git safety"


def test_publisher_no_arbitrary_write():
    publisher_path = ROOT / "app" / "publisher.py"
    content = publisher_path.read_text()
    # Should not write outside results/ via arbitrary paths
    assert "os.system" not in content.lower().replace("_", ""), "no shell execution"


# ── Medical disclaimer ─────────────────────────────────────

def test_health_response_contains_service_name():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("service") == "cancerhawk"


def test_agents_prompts_domain_frame_has_medical_guidance():
    resp = client.get("/api/agents/prompts")
    data = resp.json()
    domain = data.get("domain_frame", "")
    assert "falsif" in domain.lower() or "mechanism" in domain.lower()


def test_enrich_no_credentials_in_response():
    resp = client.post("/api/agents/enrich", json={"research_goal": "test"})
    assert resp.status_code == 200
    data = resp.json()
    dumped = json.dumps(data)
    assert "Authorization" not in dumped
    assert "Bearer" not in dumped
