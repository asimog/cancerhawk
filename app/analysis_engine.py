"""MiroShark-style archetype analysis engine.

Eight archetype agents each evaluate the finished paper across six
dimensions (clinical_viability, regulatory_risk, market_potential,
patient_impact, novelty, falsifiability) on a 1-10 scale, plus a 200-word
verdict in their persona's voice and a one-sentence "what would move the
prediction-market price" catalyst note.

The engine then synthesizes a market signal: a single 0-1 price derived
from a weighted blend of the dimension scores (regulatory_risk inverted
since high risk should push price down).
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .openrouter import CallEmitFn, chat, chat_json
from .prompts import *
from .token_tracker import TokenTracker

EmitFn = Callable[[str, str, dict | None], Awaitable[None]]


PRICE_WEIGHTS = {
    "clinical_viability": 0.25,
    "regulatory_risk": -0.20,  # negated below
    "market_potential": 0.20,
    "patient_impact": 0.15,
    "novelty": 0.10,
    "falsifiability": 0.10,
}


@dataclass
class AnalysisResult:
    archetypes: list[dict]  # [{archetype, scores, verdict, would_move_price}]
    market_price: float  # 0.0 - 1.0
    score_matrix: dict  # archetype_id -> {dim: score}
    consensus_dim: dict  # dim -> mean score
    headline_catalysts: list[str]


async def run_analysis_engine(
    api_key: str,
    paper_text: str,
    archetype_model: str,
    emit: EmitFn,
    tracker: TokenTracker,
    on_call: CallEmitFn,
) -> AnalysisResult:
    await emit(
        "analyze",
        f"Spawning {len(ARCHETYPES)} archetype agents in parallel",
        {"archetype_count": len(ARCHETYPES)},
    )

    results = await asyncio.gather(
        *[
            _evaluate(api_key, archetype, paper_text, archetype_model, emit, tracker, on_call)
            for archetype in ARCHETYPES
        ],
        return_exceptions=True,
    )

    archetype_results: list[dict] = []
    score_matrix: dict[str, dict] = {}
    catalysts: list[str] = []

    for archetype, result in zip(ARCHETYPES, results):
        if isinstance(result, Exception):
            await emit(
                "analyze",
                f"✗ {archetype['name']} failed: {result}",
                {"archetype_id": archetype["id"], "error": str(result)},
            )
            continue
        archetype_results.append(result)
        score_matrix[archetype["id"]] = result.get("scores", {})
        catalyst = (result.get("would_move_price") or "").strip()
        if catalyst:
            catalysts.append(f"[{archetype['name']}] {catalyst}")

    if not archetype_results:
        raise RuntimeError("All archetype evaluations failed")

    consensus = _consensus(score_matrix)
    market_price = _market_price(consensus)

    await emit(
        "analyze",
        f"Synthesis complete · market price = {market_price:.2f} · "
        f"consensus clinical_viability = {consensus.get('clinical_viability', 0):.1f}",
        {"market_price": market_price, "consensus": consensus},
    )

    return AnalysisResult(
        archetypes=archetype_results,
        market_price=market_price,
        score_matrix=score_matrix,
        consensus_dim=consensus,
        headline_catalysts=catalysts,
    )


async def _evaluate(
    api_key: str,
    archetype: dict,
    paper_text: str,
    model: str,
    emit: EmitFn,
    tracker: TokenTracker,
    on_call: CallEmitFn,
) -> dict:
    verdict = await chat_json(
        api_key,
        model,
        archetype_prompt(archetype, paper_text),
        temperature=0.5,
        max_tokens=1200,
        role=f"archetype:{archetype['id']}",
        tracker=tracker,
        on_call=on_call,
    )
    # LLMs sometimes wrap responses in arrays
    if isinstance(verdict, list) and verdict:
        verdict = verdict[0]
    # Defensive: ensure dict
    if not isinstance(verdict, dict):
        if isinstance(verdict, list) and verdict and isinstance(verdict[0], dict):
            verdict = verdict[0]
        else:
            raise RuntimeError(f"Analysis engine got unexpected type: {type(verdict).__name__}")
    verdict["archetype_id"] = archetype["id"]
    verdict["archetype_name"] = archetype["name"]
    avg = _avg_scores(verdict.get("scores", {}))
    await emit(
        "analyze",
        f"✓ {archetype['name']} (avg {avg:.1f})",
        {
            "archetype_id": archetype["id"],
            "scores": verdict.get("scores"),
        },
    )
    return verdict


def _avg_scores(scores: dict) -> float:
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def _consensus(matrix: dict[str, dict]) -> dict[str, float]:
    dims: dict[str, list[float]] = {}
    for scores in matrix.values():
        for dim, val in scores.items():
            if isinstance(val, (int, float)):
                dims.setdefault(dim, []).append(float(val))
    return {dim: sum(vals) / len(vals) for dim, vals in dims.items() if vals}


def _market_price(consensus: dict[str, float]) -> float:
    """Weighted blend → 0.0-1.0 price. regulatory_risk is inverted: high
    risk drags the price down."""
    score = 0.0
    weight_sum = 0.0
    for dim, weight in PRICE_WEIGHTS.items():
        if dim not in consensus:
            continue
        val = consensus[dim] / 10.0
        if dim == "regulatory_risk":
            val = 1.0 - val
        score += abs(weight) * val
        weight_sum += abs(weight)
    if weight_sum == 0:
        return 0.5
    return max(0.0, min(1.0, score / weight_sum))
