"""Agent submission API for CancerHawk.

External AI agents can:
  1. GET  /api/agents/prompts  — fetch prompt templates for local execution
  2. POST /api/agents/cost     — estimate token cost for a run
  3. POST /api/agents/run      — run full pipeline with mode selection
  4. POST /api/agents/submit   — submit a paper directly (from local execution)
  5. POST /api/agents/enrich   — research enrichment via pay.sh endpoints

Agent run modes:
  - openrouter       : Agent provides OpenRouter key (no pay.sh)
  - paysh_cancerhawk : Agent pays CancerHawk via pay.sh/x402; CancerHawk handles everything
  - paysh_owned      : Agent uses own pay.sh wallet; provides OpenRouter key for LLM
  - local            : Agent downloads prompts, runs locally, submits result (no payment)

Agents participate in the block race — highest market-price synthesis wins 0.01 USDC.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .token_tracker import PRICING_PER_M
from .prompts import submitter_prompt, DOMAIN_FRAME
from .paysh import estimate_enrichment_cost, SOLANA_WALLET

logger = logging.getLogger("cancerhawk.agents")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
AGENT_SUBMISSIONS_DIR = RESULTS_DIR / "agent_submissions"
AGENT_SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

ESTIMATED_CALLS_PER_RUN = 20
ESTIMATED_TOKENS_PER_CALL = 2500
FREE_MODEL = "openrouter/free"
DEFAULT_MODEL = "deepseek/deepseek-v4-pro"
PRIZE_AMOUNT_USDC = 0.01

GIVEWELL_SOLANA = "4Z2DBVoQCJZ42cCTDMNvYDUqRjA1C3vV7B155Mc6jGah"
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()


# ---------------------------------------------------------------------------
# Prompt exposure for local (BYOK) agents
# ---------------------------------------------------------------------------

def get_prompts() -> dict[str, Any]:
    """Return prompt templates agents can use to run papers locally."""
    return {
        "domain_frame": DOMAIN_FRAME,
        "engines": {
            "submitter": {
                "role": "Generates novel mathematical/biological research insights",
                "prompt_template": (
                    "You are a research submitter in the CancerHawk engine. "
                    "Generate ONE novel, deeply developed research insight "
                    "that advances the goal: {research_goal}. "
                    "Focus on molecular/cellular mechanisms. Be falsifiable. "
                    "Avoid repeating prior rejections: {prior_rejections}. "
                    "Build on prior accepted work: {prior_accepted}."
                ),
                "output_format": "JSON with keys: submission (string), reasoning (string)",
            },
            "validator": {
                "role": "Evaluates submissions for scientific merit",
                "prompt_template": (
                    "You are a validation agent. Evaluate this submission for "
                    "scientific rigor, novelty, and falsifiability. "
                    "Return JSON with: decision (accept/reject), "
                    "reasoning (string), summary (string)."
                ),
                "output_format": "JSON with keys: decision, reasoning, summary",
            },
            "archetype": {
                "role": "Multi-perspective analysis of completed papers",
                "prompt_template": (
                    "You are an archetype analyst. Score this paper across "
                    "dimensions: clinical_viability (1-10), novelty (1-10), "
                    "falsifiability (1-10), patient_impact (1-10), "
                    "regulatory_risk (1-10), market_potential (1-10). "
                    "Return JSON with: archetype_name, scores, verdict."
                ),
                "output_format": "JSON with archetype_name, scores, verdict",
            },
            "peer_reviewer": {
                "role": "Independent peer review of completed papers",
                "prompt_template": (
                    "You are a MiroShark peer reviewer. Review this paper. "
                    "Return JSON with: recommendation (accept/minor_revision/"
                    "major_revision/reject), dimension_scores (object), "
                    "criticisms (array), required_fixes (array), "
                    "suggested_experiments (array), summary (string), "
                    "confidence (0-1)."
                ),
                "output_format": "JSON with recommendation, scores, critiques",
            },
        },
        "openrouter_endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "openrouter_public_key": OPENROUTER_KEY[:20] + "***" if OPENROUTER_KEY else "not configured",
        "estimated_calls_per_run": ESTIMATED_CALLS_PER_RUN,
        "estimated_tokens_per_call": ESTIMATED_TOKENS_PER_CALL,
    }


# ---------------------------------------------------------------------------
# Cost estimation (for pay.sh / x402 agents)
# ---------------------------------------------------------------------------

def estimate_run_cost(
    model: str | None = None,
    n_submitters: int = 3,
    mode: str = "openrouter",
) -> dict[str, Any]:
    """Estimate the OpenRouter API cost for a full pipeline run.
    """
    model = str(model or FREE_MODEL).strip()
    n_runs = max(1, min(8, int(n_submitters)))

    pricing = PRICING_PER_M.get(model, (0.0, 0.0))
    input_price, output_price = pricing

    num_calls = ESTIMATED_CALLS_PER_RUN + n_runs * 3
    total_input_tokens = num_calls * ESTIMATED_TOKENS_PER_CALL
    total_output_tokens = num_calls * ESTIMATED_TOKENS_PER_CALL * 0.5

    cost = (total_input_tokens / 1_000_000) * input_price + (total_output_tokens / 1_000_000) * output_price

    result: dict[str, Any] = {
        "model": model,
        "mode": mode,
        "input_price_per_m": input_price,
        "output_price_per_m": output_price,
        "estimated_calls": num_calls,
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": int(total_output_tokens),
        "estimated_total_tokens": int(total_input_tokens + total_output_tokens),
        "estimated_cost_usd": round(cost, 6),
        "estimated_cost_usd_display": f"${cost:.6f}",
        "free_tier_available": model in {FREE_MODEL, "openrouter/free"},
        "note": (
            "Free tier models cost $0.0000. No payment required."
            if model in {FREE_MODEL, "openrouter/free"}
            else f"Estimated ~${cost:.4f} for a full run with {n_runs} submitters."
        ),
    }

    if mode in ("paysh_cancerhawk", "paysh_owned"):
        enrichment = estimate_enrichment_cost()
        result["paysh_enrichment"] = enrichment
        total_paysh = enrichment.get("total_cost_usd", 0)
        result["estimated_cost_usd"] = round(cost + total_paysh, 6)
        result["estimated_cost_usd_display"] = f"${cost + total_paysh:.6f}"
        result["solana_wallet"] = SOLANA_WALLET or "not set"
        result["note"] = (
            f"OpenRouter: ~${cost:.4f} + pay.sh enrichment: ${total_paysh:.2f}. "
            f"Pay CancerHawk wallet: {SOLANA_WALLET}"
        )

    return result


# ---------------------------------------------------------------------------
# Paper submission (from agents running locally)
# ---------------------------------------------------------------------------

def submit_paper(
    agent_name: str,
    paper_title: str,
    paper_content: str,
    research_goal: str,
    agent_model: str | None = None,
    peer_reviews: list[dict] | None = None,
    simulations: list[dict] | None = None,
    wallet_address: str | None = None,
) -> dict[str, Any]:
    """Accept a paper submission from an external agent.

    The agent ran CancerHawk prompts locally and is submitting the result.
    Papers are stored in results/agent_submissions/ for scoring.
    """
    submission_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()

    record = {
        "submission_id": submission_id,
        "agent_name": agent_name[:120],
        "agent_model": (agent_model or "unknown")[:120],
        "paper_title": paper_title[:500],
        "paper_content": paper_content[:50000],
        "research_goal": research_goal[:1000],
        "peer_reviews": (peer_reviews or [])[:20],
        "simulations": (simulations or [])[:20],
        "wallet_address": (wallet_address or "").strip() or GIVEWELL_SOLANA,
        "submitted_at": now,
        "status": "received",
    }

    sub_path = AGENT_SUBMISSIONS_DIR / f"{submission_id}.json"
    sub_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    logger.info(
        "agent_paper_submitted",
        extra={
            "submission_id": submission_id,
            "agent": agent_name,
            "title": paper_title[:120],
            "has_reviews": bool(peer_reviews),
            "has_simulations": bool(simulations),
        },
    )

    # Load leaderboard
    leaderboard = _load_leaderboard()
    leaderboard.append({
        "submission_id": submission_id,
        "agent_name": agent_name,
        "paper_title": paper_title,
        "research_goal": research_goal,
        "submitted_at": now,
        "wallet_address": wallet_address or GIVEWELL_SOLANA,
    })
    _save_leaderboard(leaderboard)

    return {
        "success": True,
        "submission_id": submission_id,
        "message": f"Paper submitted! Your submission ID is {submission_id}. "
                   f"Results will be scored by the CancerHawk engine. "
                   f"Prize: {PRIZE_AMOUNT_USDC} USDC for the highest market-price synthesis.",
        "leaderboard_position": len(leaderboard),
        "total_submissions": len(leaderboard),
        "prize": f"{PRIZE_AMOUNT_USDC} USDC",
        "next_steps": [
            "Your paper will be peer-reviewed by the CancerHawk archetype engine",
            "Scores are posted to the leaderboard at /api/agents/leaderboard",
            f"Winner receives {PRIZE_AMOUNT_USDC} USDC to their wallet",
        ],
    }


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

LEADERBOARD_FILE = AGENT_SUBMISSIONS_DIR / "leaderboard.json"


def _load_leaderboard() -> list[dict]:
    if not LEADERBOARD_FILE.exists():
        return []
    try:
        return json.loads(LEADERBOARD_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_leaderboard(data: list[dict]) -> None:
    LEADERBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_leaderboard() -> list[dict]:
    return _load_leaderboard()


# ---------------------------------------------------------------------------
# Agent run (full pipeline with agent's OpenRouter key)
# ---------------------------------------------------------------------------

def parse_agent_run(cfg: dict[str, Any]) -> tuple[str, str, str, int, str | None, str]:
    """Parse and validate an agent run request.

    Returns (api_key, research_goal, model, n_submitters, agent_name, mode).
    Modes: openrouter, paysh_cancerhawk, paysh_owned, local
    """
    mode = str(cfg.get("mode") or "openrouter").strip()

    if mode == "local":
        return "", "", "", 3, None, "local"

    api_key = str(cfg.get("api_key") or "").strip()
    if mode not in ("paysh_cancerhawk",):
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OpenRouter API key required — provide api_key or set OPENROUTER_API_KEY")

    research_goal = str(cfg.get("research_goal") or "").strip()
    if not research_goal:
        raise ValueError("research_goal is required")
    if len(research_goal) > 1000:
        raise ValueError("research_goal must be at most 1000 characters")

    model = str(cfg.get("model") or FREE_MODEL).strip()
    n_submitters = int(cfg.get("n_submitters") or 3)
    if n_submitters < 1 or n_submitters > 8:
        raise ValueError("n_submitters must be between 1 and 8")

    agent_name = str(cfg.get("agent_name") or "").strip() or None

    return api_key, research_goal, model, n_submitters, agent_name, mode
