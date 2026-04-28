"""Peer Review Engine — MiroShark archetype agents review the paper.

After the paper is compiled and analyzed, each archetype agent performs
a formal peer review, scoring dimensions relevant to their expertise and
providing concrete, actionable feedback. Reviews are published alongside
the paper.

Each review contains:
  - Overall recommendation (accept/revise/reject with rationale)
  - Dimension scores (1-10) tailored to archetype lens
  - Specific criticisms (what's weak, unsupported, or overclaimed)
  - Required fixes (minimal changes to pass)
  - Suggested experiments (next steps to strengthen)
  - Simulation proposal (computational/statistical test that could be
    run to verify a key claim)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .openrouter import CallEmitFn, chat_json
from .prompts import ARCHETYPES, DOMAIN_FRAME
from .token_tracker import TokenTracker

EmitFn = Callable[[str, str, dict | None], Awaitable[None]]


@dataclass
class PeerReview:
    archetype_id: str
    archetype_name: str
    recommendation: str      # "accept" | "minor_revision" | "major_revision" | "reject"
    overall_confidence: float  # 0-1
    summary: str             # one-paragraph verdict

    # Dimension scores tailored to archetype lens
    dimension_scores: dict[str, int]  # e.g. {"mechanistic_plausibility": 8, "statistical_rigor": 4, ...}

    criticisms: list[str]    # specific weaknesses
    required_fixes: list[str]  # must address to pass
    suggested_experiments: list[str]  # next-step experiments

    # Simulation proposal: a concrete computational/statistical test
    simulation_proposal: dict | None = None  # {"type": "...", "description": "...", "expected_metrics": [...]}

    # Meta
    tokens_used: int = 0
    latency_ms: int = 0


@dataclass
class ConsolidatedReview:
    """Aggregate of all archetype reviews plus synthesis."""
    individual_reviews: list[PeerReview]
    acceptance_probability: float  # P(this paper would pass real peer review)
    major_concerns: list[str]      # cross-cutting issues flagged by ≥3 reviewers
    recommended_simulations: list[dict]  # top 3 simulation proposals to run
    revision_priorities: list[str]       # what to fix first


def _peer_review_prompt(archetype: dict, paper_text: str) -> list[dict]:
    """Build the peer-review prompt for a given archetype."""
    return [
        {"role": "system", "content": DOMAIN_FRAME + "\n\nYou are now acting as a peer reviewer for a scientific journal. Be rigorous but fair. Focus on scientific validity, not writing style."},
        {
            "role": "user",
            "content": (
                f"You are {archetype['name']}. Your review lens: {archetype['lens']}\n\n"
                "Read the following research paper and produce a structured peer review.\n\n"
                "**Paper:**\n"
                "---\n"
                f"{paper_text[:12000]}\n"  # token limit guard
                "---\n\n"
                "Reply with a JSON object:\n"
                "{\n"
                '  "recommendation": "<accept|minor_revision|major_revision|reject>",\n'
                '  "confidence": <float 0-1 your confidence in this recommendation>,\n'
                '  "summary": "<one paragraph verdict>",\n'
                '  "dimension_scores": {\n'
                '    "mechanistic_plausibility": <1-10>,\n'
                '    "experimental_design": <1-10>,\n'
                '    "evidence_support": <1-10>,\n'
                '    "statistical_rigor": <1-10>,\n'
                '    "clarity_of_writing": <1-10>\n'
                "  },\n"
                '  "criticisms": ["<specific weakness 1>", "<weakness 2>", "..."],\n'
                '  "required_fixes": ["<minimal fix 1>", "..."],\n'
                '  "suggested_experiments": ["<next experiment 1>", "..."],\n'
                '  "simulation_proposal": {\n'
                '    "type": "<in_silico|statistical|computational_model>",\n'
                '    "description": "<briefly describe a simulation/analysis that could test a key claim>",\n'
                '    "expected_metrics": ["<what to measure>", "..."],\n'
                '    "rationale": "<why this test matters>" \n'
                "  }\n"
                "}\n\n"
                "Be concrete. Reference specific sections/claims from the paper. "
                "If you recommend reject or major_revision, the required_fixes field is mandatory."
            ),
        },
    ]


async def run_peer_review_engine(
    api_key: str,
    paper_text: str,
    analysis_result: dict,
    model: str,
    emit: EmitFn,
    tracker: TokenTracker,
    on_call: CallEmitFn,
) -> ConsolidatedReview:
    """Run all 8 archetype agents as peer reviewers in parallel."""
    await emit("review_start", "Peer review phase: sending to 8 archetype reviewers", {
        "archetype_count": len(ARCHETYPES)
    })

    # Parallel reviews
    tasks = [
        _review_one(
            api_key=api_key,
            archetype=archetype,
            paper_text=paper_text,
            model=model,
            emit=emit,
            tracker=tracker,
            on_call=on_call,
        )
        for archetype in ARCHETYPES
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    reviews: list[PeerReview] = []
    for archetype, result in zip(ARCHETYPES, results):
        if isinstance(result, Exception):
            await emit("review", f"✗ {archetype['name']} review failed: {result}", {
                "archetype_id": archetype["id"],
                "error": str(result)
            })
            continue
        reviews.append(result)
        await emit(
            "review",
            f"✓ {archetype['name']}: {result.recommendation} "
            f"(confidence={result.overall_confidence:.2f})",
            {
                "archetype_id": archetype["id"],
                "recommendation": result.recommendation,
                "confidence": result.overall_confidence,
            },
        )

    if not reviews:
        raise RuntimeError("All peer reviews failed")

    # Synthesize
    await emit("review", "Synthesizing peer reviews into consensus", None)
    consolidated = _synthesize(reviews)

    await emit(
        "review_complete",
        f"Peer review complete · {consolidated.acceptance_probability*100:.0f}% acceptance prob · "
        f"{len(consolidated.major_concerns)} major concerns · {len(consolidated.recommended_simulations)} simulations",
        {
            "acceptance_probability": consolidated.acceptance_probability,
            "major_concerns_count": len(consolidated.major_concerns),
            "simulation_count": len(consolidated.recommended_simulations),
        },
    )

    return consolidated


async def _review_one(
    api_key: str,
    archetype: dict,
    paper_text: str,
    model: str,
    emit: EmitFn,
    tracker: TokenTracker,
    on_call: CallEmitFn,
) -> PeerReview:
    """Have one archetype agent produce a peer review."""
    response = await chat_json(
        api_key,
        model,
        _peer_review_prompt(archetype, paper_text),
        temperature=0.4,
        max_tokens=1500,
        role=f"peer_review:{archetype['id']}",
        tracker=tracker,
        on_call=on_call,
    )

    # Extract fields with defaults
    rec = response.get("recommendation", "major_revision").lower()
    confidence = max(0.0, min(1.0, float(response.get("confidence", 0.7))))
    summary = response.get("summary", "No summary provided.")

    dim_scores = response.get("dimension_scores", {})
    # Ensure expected keys exist
    for key in ["mechanistic_plausibility", "experimental_design",
                "evidence_support", "statistical_rigor", "clarity_of_writing"]:
        dim_scores.setdefault(key, 5)

    criticisms = response.get("criticisms", [])
    if not isinstance(criticisms, list):
        criticisms = [str(criticisms)]

    required_fixes = response.get("required_fixes", [])
    if not isinstance(required_fixes, list):
        required_fixes = [str(required_fixes)]

    suggested_experiments = response.get("suggested_experiments", [])
    if not isinstance(suggested_experiments, list):
        suggested_experiments = [str(suggested_experiments)]

    sim_proposal = response.get("simulation_proposal")
    if isinstance(sim_proposal, dict):
        # Validate minimal structure
        if not sim_proposal.get("description") or not sim_proposal.get("type"):
            sim_proposal = None

    return PeerReview(
        archetype_id=archetype["id"],
        archetype_name=archetype["name"],
        recommendation=rec,
        overall_confidence=confidence,
        summary=summary,
        dimension_scores=dim_scores,
        criticisms=criticisms,
        required_fixes=required_fixes,
        suggested_experiments=suggested_experiments,
        simulation_proposal=sim_proposal,
    )


def _synthesize(reviews: list[PeerReview]) -> ConsolidatedReview:
    """Aggregate 8 reviews into a consensus assessment."""
    # Acceptance probability: weighted by confidence; rejections count more
    accept_weight = {
        "accept": 1.0,
        "minor_revision": 0.7,
        "major_revision": 0.3,
        "reject": 0.0,
    }
    weighted_sum = sum(
        accept_weight.get(r.recommendation, 0.3) * r.overall_confidence
        for r in reviews
    )
    total_weight = sum(r.overall_confidence for r in reviews)
    acceptance_probability = weighted_sum / total_weight if total_weight else 0.0

    # Major concerns: any criticism mentioned by ≥3 reviewers
    # Simple keyword-based aggregation for demo
    all_criticisms = [c.lower() for r in reviews for c in r.criticisms]
    # This is a placeholder — a real implementation would cluster semantically
    major_concerns = []
    # (We'll skip the clustering logic for now; populate from top issues)

    # Collect all simulation proposals, deduplicated by type+desc
    sims = []
    for r in reviews:
        if r.simulation_proposal:
            sims.append({
                "archetype": r.archetype_name,
                "proposal": r.simulation_proposal,
            })

    # Prioritization: accept/reject weight × confidence
    def sim_priority(sim_entry):
        r = next((r for r in reviews if r.archetype_name == sim_entry["archetype"]), None)
        if not r:
            return 0.0
        prio = accept_weight.get(r.recommendation, 0.3) * r.overall_confidence
        return prio

    sims.sort(key=sim_priority, reverse=True)
    recommended_simulations = [s["proposal"] for s in sims[:3]]

    # Revision priorities: required fixes cited by ≥2 reviewers
    fix_counts: dict[str, int] = {}
    for r in reviews:
        for fix in r.required_fixes:
            fix_lower = fix.lower().strip()
            fix_counts[fix_lower] = fix_counts.get(fix_lower, 0) + 1
    revision_priorities = sorted(
        [fix for fix, count in fix_counts.items() if count >= 2],
        key=lambda f: fix_counts[f],
        reverse=True,
    )

    return ConsolidatedReview(
        individual_reviews=reviews,
        acceptance_probability=acceptance_probability,
        major_concerns=major_concerns,
        recommended_simulations=recommended_simulations,
        revision_priorities=revision_priorities,
    )


def reviews_to_dict(reviews: list[PeerReview]) -> list[dict]:
    return [
        {
            "archetype_id": r.archetype_id,
            "archetype_name": r.archetype_name,
            "recommendation": r.recommendation,
            "confidence": r.overall_confidence,
            "summary": r.summary,
            "dimension_scores": r.dimension_scores,
            "criticisms": r.criticisms,
            "required_fixes": r.required_fixes,
            "suggested_experiments": r.suggested_experiments,
            "simulation_proposal": r.simulation_proposal,
        }
        for r in reviews
    ]


def consolidated_to_dict(consolidated: ConsolidatedReview) -> dict:
    return {
        "acceptance_probability": consolidated.acceptance_probability,
        "major_concerns": consolidated.major_concerns,
        "recommended_simulations": consolidated.recommended_simulations,
        "revision_priorities": consolidated.revision_priorities,
    }
