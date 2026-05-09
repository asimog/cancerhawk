"""CancerHawk block race orchestration.

HyperFlow boxes:
1. Precigenetic base layer emits ten TCGA/GDC target-discovery topics.
2. Ten independent MOTO workers research those topics into candidate papers.
3. Ten independent MiroShark reviewers review the candidates.
4. One MOTO validator ranks the papers and emits the next ten topics.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .openrouter import CallEmitFn, chat_json
from .paper_engine import Paper, run_paper_engine
from .precigenetic_base_layer import build_precigenetic_topics, format_base_layer_context
from .subagent_wallets import assign_block_race_wallets
from .token_tracker import APIFailureLimitExceeded, TokenTracker

logger = logging.getLogger("cancerhawk.block_race")

EmitFn = Callable[[str, str, dict | None], Awaitable[None]]

BLOCK_RACE_TOPIC_COUNT = int(os.environ.get("CANCERHAWK_BLOCK_RACE_TOPIC_COUNT", "10"))
MOTO_WORKER_CONCURRENCY = int(os.environ.get("CANCERHAWK_MOTO_WORKER_CONCURRENCY", "10"))
MIROSHARK_REVIEW_CONCURRENCY = int(os.environ.get("CANCERHAWK_MIROSHARK_REVIEW_CONCURRENCY", "10"))


MIROSHARK_REVIEWERS = [
    {"id": "oncologist", "name": "MiroShark Oncologist", "lens": "clinical mechanism, patient burden, standard-of-care fit"},
    {"id": "biostatistician", "name": "MiroShark Biostatistician", "lens": "endpoint validity, power, confounding, falsifiability"},
    {"id": "fda", "name": "MiroShark FDA Strategist", "lens": "IND path, safety package, regulatory failure modes"},
    {"id": "investor", "name": "MiroShark Biotech Investor", "lens": "market pressure, moat, capital intensity, translational velocity"},
    {"id": "kol", "name": "MiroShark Academic KOL", "lens": "field novelty, publishability, scientific controversy"},
    {"id": "patient", "name": "MiroShark Patient Advocate", "lens": "access, tolerability, lived treatment burden"},
    {"id": "payer", "name": "MiroShark Payer", "lens": "coverage, ICER, budget impact, diagnostic burden"},
    {"id": "short_seller", "name": "MiroShark Short-Seller", "lens": "fatal flaw, overclaim, unpriced risk"},
    {"id": "computational", "name": "MiroShark Computational Auditor", "lens": "data provenance, reproducibility, model misspecification"},
    {"id": "translational", "name": "MiroShark Translational Biologist", "lens": "assay design, tissue realism, target validation"},
]


@dataclass
class RaceCandidate:
    index: int
    topic: dict[str, Any]
    worker_id: str
    worker_wallet: dict[str, Any]
    paper: Paper

    def summary(self) -> dict[str, Any]:
        text = self.paper.full_text()
        return {
            "candidate_index": self.index,
            "worker_id": self.worker_id,
            "worker_wallet": self.worker_wallet,
            "topic": self.topic,
            "title": self.paper.title,
            "word_count": len(text.split()),
            "section_count": len(self.paper.sections),
            "accepted_submissions": len(self.paper.accepted_submissions),
            "excerpt": text[:1200],
        }


@dataclass
class BlockRaceResult:
    topics: list[dict[str, Any]]
    base_metadata: dict[str, Any]
    candidates: list[RaceCandidate]
    reviews: list[dict[str, Any]]
    validator_decision: dict[str, Any]
    winner: RaceCandidate
    next_topics: list[dict[str, Any]]
    subagent_wallets: dict[str, dict[str, Any]]

    def race_metadata(self) -> dict[str, Any]:
        return {
            "base_layer": self.base_metadata,
            "topics": self.topics,
            "candidate_count": len(self.candidates),
            "review_count": len(self.reviews),
            "winner_index": self.winner.index,
            "winner_worker_id": self.winner.worker_id,
            "validator_decision": self.validator_decision,
            "candidate_summaries": [candidate.summary() for candidate in self.candidates],
            "subagent_wallets": self.subagent_wallets,
        }


async def run_block_race(
    *,
    api_key: str,
    research_goal: str,
    models: dict[str, str],
    emit: EmitFn,
    tracker: TokenTracker,
    on_call: CallEmitFn,
    previous_block_context: str,
    run_id: str | None = None,
) -> BlockRaceResult:
    """Run the full 10-topic CancerHawk block race."""
    topic_count = max(1, BLOCK_RACE_TOPIC_COUNT)
    run_key = run_id or str(int(time.time()))
    wallets = assign_block_race_wallets(run_key, topic_count=topic_count)

    topics, base_metadata = build_precigenetic_topics(research_goal, topic_count=topic_count)
    base_context = format_base_layer_context(topics, base_metadata)
    merged_context = "\n\n".join(part for part in [base_context, previous_block_context] if part.strip())

    await emit(
        "base_layer",
        f"Precigenetic base layer produced {len(topics)} TCGA/GDC research topics",
        {
            "source_repo": base_metadata["source_repo"],
            "gdc_ok": bool(base_metadata.get("gdc_summary", {}).get("ok")),
            "topics": topics,
            "subagent_wallets": wallets,
        },
    )

    candidates = await _run_moto_workers(
        api_key=api_key,
        topics=topics,
        models=models,
        emit=emit,
        tracker=tracker,
        on_call=on_call,
        previous_block_context=merged_context,
        wallets=wallets,
    )
    if not candidates:
        raise RuntimeError("No MOTO workers produced candidate papers")

    reviews = await _run_miroshark_reviews(
        api_key=api_key,
        candidates=candidates,
        model=models["archetype"],
        emit=emit,
        tracker=tracker,
        on_call=on_call,
        wallets=wallets,
    )

    decision = await _rank_with_moto_validator(
        api_key=api_key,
        candidates=candidates,
        reviews=reviews,
        model=models["validator"],
        emit=emit,
        tracker=tracker,
        on_call=on_call,
        wallets=wallets,
        research_goal=research_goal,
    )
    winner_index = _coerce_winner_index(decision, candidates)
    winner = next(candidate for candidate in candidates if candidate.index == winner_index)
    next_topics = _normalize_next_topics(decision.get("next_topics") or decision.get("topics"), winner.topic)
    decision["winner_index"] = winner.index
    decision["winner_worker_id"] = winner.worker_id
    decision["validator_wallet"] = wallets["moto_validator_01"]

    await emit(
        "block_validator",
        f"MOTO validator selected candidate {winner.index}: {winner.paper.title}",
        {
            "winner_index": winner.index,
            "winner_title": winner.paper.title,
            "next_topics": next_topics,
            "rankings": decision.get("rankings", []),
            "validator_wallet": wallets["moto_validator_01"],
        },
    )

    return BlockRaceResult(
        topics=topics,
        base_metadata=base_metadata,
        candidates=candidates,
        reviews=reviews,
        validator_decision=decision,
        winner=winner,
        next_topics=next_topics,
        subagent_wallets=wallets,
    )


async def _run_moto_workers(
    *,
    api_key: str,
    topics: list[dict[str, Any]],
    models: dict[str, str],
    emit: EmitFn,
    tracker: TokenTracker,
    on_call: CallEmitFn,
    previous_block_context: str,
    wallets: dict[str, dict[str, Any]],
) -> list[RaceCandidate]:
    semaphore = asyncio.Semaphore(max(1, MOTO_WORKER_CONCURRENCY))

    async def run_one(topic: dict[str, Any]) -> RaceCandidate | None:
        index = int(topic["id"])
        worker_id = f"moto_worker_{index:02d}"
        async with semaphore:
            await emit(
                "moto_worker",
                f"{worker_id} researching topic {index}/10: {topic['title']}",
                {"topic": topic, "worker_id": worker_id, "wallet": wallets[worker_id]},
            )
            try:
                paper = await run_paper_engine(
                    api_key=api_key,
                    research_goal=topic["research_goal"],
                    models=models,
                    n_submitters=1,
                    emit=emit,
                    tracker=tracker,
                    on_call=on_call,
                    previous_block_context=previous_block_context,
                )
            except APIFailureLimitExceeded:
                raise
            except Exception as exc:
                logger.warning("moto_worker_failed", extra={"worker_id": worker_id, "error": str(exc)})
                await emit(
                    "moto_worker",
                    f"{worker_id} failed: {type(exc).__name__}: {exc}",
                    {"worker_id": worker_id, "error": str(exc), "topic": topic},
                )
                return None
            await emit(
                "moto_worker_done",
                f"{worker_id} produced candidate paper: {paper.title}",
                {
                    "candidate_index": index,
                    "worker_id": worker_id,
                    "wallet": wallets[worker_id],
                    "title": paper.title,
                    "sections": len(paper.sections),
                },
            )
            return RaceCandidate(
                index=index,
                topic=topic,
                worker_id=worker_id,
                worker_wallet=wallets[worker_id],
                paper=paper,
            )

    results = await asyncio.gather(*(run_one(topic) for topic in topics), return_exceptions=True)
    candidates: list[RaceCandidate] = []
    for result in results:
        if isinstance(result, APIFailureLimitExceeded):
            raise result
        if isinstance(result, Exception):
            await emit("moto_worker", f"MOTO worker task failed: {result}", {"error": str(result)})
        elif result is not None:
            candidates.append(result)
    candidates.sort(key=lambda candidate: candidate.index)
    return candidates


async def _run_miroshark_reviews(
    *,
    api_key: str,
    candidates: list[RaceCandidate],
    model: str,
    emit: EmitFn,
    tracker: TokenTracker,
    on_call: CallEmitFn,
    wallets: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, MIROSHARK_REVIEW_CONCURRENCY))

    async def review_one(slot: int, candidate: RaceCandidate) -> dict[str, Any] | None:
        reviewer = MIROSHARK_REVIEWERS[(slot - 1) % len(MIROSHARK_REVIEWERS)]
        reviewer_id = f"miroshark_reviewer_{slot:02d}"
        async with semaphore:
            await emit(
                "review",
                f"{reviewer_id} reviewing candidate {candidate.index}: {candidate.paper.title}",
                {
                    "reviewer": reviewer,
                    "reviewer_id": reviewer_id,
                    "reviewer_wallet": wallets[reviewer_id],
                    "candidate_index": candidate.index,
                },
            )
            try:
                response = await chat_json(
                    api_key,
                    model,
                    _miroshark_race_review_prompt(reviewer, candidate),
                    temperature=0.35,
                    max_tokens=1500,
                    role=f"miroshark_review:{reviewer['id']}:{candidate.index}",
                    tracker=tracker,
                    on_call=on_call,
                )
            except APIFailureLimitExceeded:
                raise
            except Exception as exc:
                logger.warning("miroshark_review_failed", extra={"reviewer_id": reviewer_id, "error": str(exc)})
                await emit(
                    "review",
                    f"{reviewer_id} review failed: {type(exc).__name__}: {exc}",
                    {"reviewer_id": reviewer_id, "candidate_index": candidate.index, "error": str(exc)},
                )
                return None

            review = _normalize_review_response(response, reviewer, reviewer_id, candidate, wallets[reviewer_id])
            await emit(
                "review",
                f"{reviewer_id}: {review['recommendation']} candidate {candidate.index} "
                f"(score={review['overall_score']:.1f})",
                {
                    "reviewer_id": reviewer_id,
                    "candidate_index": candidate.index,
                    "overall_score": review["overall_score"],
                    "recommendation": review["recommendation"],
                    "wallet": wallets[reviewer_id],
                },
            )
            return review

    slots = list(enumerate(candidates, start=1))
    results = await asyncio.gather(*(review_one(slot, candidate) for slot, candidate in slots), return_exceptions=True)
    reviews: list[dict[str, Any]] = []
    for result in results:
        if isinstance(result, APIFailureLimitExceeded):
            raise result
        if isinstance(result, Exception):
            await emit("review", f"MiroShark review task failed: {result}", {"error": str(result)})
        elif result is not None:
            reviews.append(result)
    return reviews


async def _rank_with_moto_validator(
    *,
    api_key: str,
    candidates: list[RaceCandidate],
    reviews: list[dict[str, Any]],
    model: str,
    emit: EmitFn,
    tracker: TokenTracker,
    on_call: CallEmitFn,
    wallets: dict[str, dict[str, Any]],
    research_goal: str,
) -> dict[str, Any]:
    await emit(
        "block_validator",
        "MOTO validator ranking candidate papers and deriving the next ten topics",
        {"candidate_count": len(candidates), "review_count": len(reviews), "wallet": wallets["moto_validator_01"]},
    )
    try:
        decision = await chat_json(
            api_key,
            model,
            _moto_validator_prompt(research_goal, candidates, reviews),
            temperature=0.25,
            max_tokens=2500,
            role="moto_block_validator",
            tracker=tracker,
            on_call=on_call,
        )
    except APIFailureLimitExceeded:
        raise
    except Exception as exc:
        logger.warning("moto_validator_failed", extra={"error": str(exc)})
        await emit("block_validator", f"MOTO validator failed; using score fallback: {exc}", {"error": str(exc)})
        return _fallback_validator_decision(candidates, reviews)
    return decision if isinstance(decision, dict) else _fallback_validator_decision(candidates, reviews)


def _miroshark_race_review_prompt(reviewer: dict[str, str], candidate: RaceCandidate) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a MiroShark peer-review agent in CancerHawk. Be rigorous, "
                "specific, and skeptical. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"REVIEWER: {reviewer['name']}\n"
                f"LENS: {reviewer['lens']}\n"
                f"CANDIDATE INDEX: {candidate.index}\n"
                f"BASE TOPIC: {candidate.topic['title']}\n\n"
                "Review this candidate paper for the block race. Return JSON with:\n"
                "{\n"
                '  "recommendation": "accept|minor_revision|major_revision|reject",\n'
                '  "overall_score": 1-10,\n'
                '  "confidence": 0.0-1.0,\n'
                '  "summary": "one paragraph",\n'
                '  "dimension_scores": {"mechanistic_plausibility":1-10,"falsifiability":1-10,"evidence_support":1-10,"clinical_viability":1-10,"novelty":1-10},\n'
                '  "criticisms": ["..."],\n'
                '  "required_fixes": ["..."],\n'
                '  "suggested_experiments": ["..."],\n'
                '  "simulation_proposal": {"type":"computational_model|statistical|in_silico","description":"...","expected_metrics":["..."],"rationale":"..."}\n'
                "}\n\n"
                "PAPER:\n---\n"
                f"{candidate.paper.full_text()[:12000]}\n---"
            ),
        },
    ]


def _moto_validator_prompt(
    research_goal: str,
    candidates: list[RaceCandidate],
    reviews: list[dict[str, Any]],
) -> list[dict[str, str]]:
    candidate_summaries = json.dumps([candidate.summary() for candidate in candidates], indent=2)[:22000]
    review_summaries = json.dumps(reviews, indent=2)[:14000]
    return [
        {
            "role": "system",
            "content": (
                "You are the single MOTO validator for a CancerHawk block race. "
                "Rank the candidate papers using the MiroShark reviews, choose one "
                "winner for publication, and derive the next ten topics. Return only JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"RESEARCH GOAL: {research_goal}\n\n"
                "Return JSON with this shape:\n"
                "{\n"
                '  "winner_index": <candidate_index>,\n'
                '  "rankings": [{"rank":1,"candidate_index":1,"score":0.0-1.0,"reason":"..."}],\n'
                '  "selection_rationale": "why this winner is best",\n'
                '  "next_topics": [{"id":1,"title":"...","probability":0.0-1.0,"impact":1-10,"token_cost":4000,"rationale":"..."}]\n'
                "}\n"
                "The next_topics array must contain exactly 10 topics for the next block.\n\n"
                f"CANDIDATES:\n{candidate_summaries}\n\n"
                f"MIROSHARK REVIEWS:\n{review_summaries}"
            ),
        },
    ]


def _normalize_review_response(
    response: Any,
    reviewer: dict[str, str],
    reviewer_id: str,
    candidate: RaceCandidate,
    wallet: dict[str, Any],
) -> dict[str, Any]:
    data = response[0] if isinstance(response, list) and response and isinstance(response[0], dict) else response
    if not isinstance(data, dict):
        data = {}
    scores = data.get("dimension_scores")
    if not isinstance(scores, dict):
        scores = {}
    numeric_scores = [float(v) for v in scores.values() if isinstance(v, (int, float))]
    overall = data.get("overall_score")
    if not isinstance(overall, (int, float)):
        overall = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 5.0
    criticisms = data.get("criticisms") if isinstance(data.get("criticisms"), list) else []
    fixes = data.get("required_fixes") if isinstance(data.get("required_fixes"), list) else []
    experiments = data.get("suggested_experiments") if isinstance(data.get("suggested_experiments"), list) else []
    simulation = data.get("simulation_proposal") if isinstance(data.get("simulation_proposal"), dict) else None
    return {
        "archetype_id": reviewer["id"],
        "archetype_name": reviewer["name"],
        "reviewer_id": reviewer_id,
        "reviewer_wallet": wallet,
        "candidate_index": candidate.index,
        "candidate_title": candidate.paper.title,
        "recommendation": str(data.get("recommendation") or "major_revision").lower(),
        "confidence": max(0.0, min(1.0, float(data.get("confidence") or 0.65))),
        "overall_score": max(1.0, min(10.0, float(overall))),
        "summary": str(data.get("summary") or "No summary provided."),
        "dimension_scores": scores,
        "criticisms": [str(item) for item in criticisms],
        "required_fixes": [str(item) for item in fixes],
        "suggested_experiments": [str(item) for item in experiments],
        "simulation_proposal": simulation,
    }


def _coerce_winner_index(decision: dict[str, Any], candidates: list[RaceCandidate]) -> int:
    valid = {candidate.index for candidate in candidates}
    raw = decision.get("winner_index") or decision.get("best_candidate_index") or decision.get("winner")
    try:
        winner = int(raw)
    except (TypeError, ValueError):
        winner = -1
    if winner in valid:
        return winner
    fallback = _fallback_validator_decision(candidates, [])
    return int(fallback["winner_index"])


def _fallback_validator_decision(candidates: list[RaceCandidate], reviews: list[dict[str, Any]]) -> dict[str, Any]:
    scores: dict[int, list[float]] = {}
    for review in reviews:
        try:
            scores.setdefault(int(review.get("candidate_index")), []).append(float(review.get("overall_score") or 0.0))
        except (TypeError, ValueError):
            continue
    ranked: list[tuple[int, float]] = []
    for candidate in candidates:
        vals = scores.get(candidate.index, [])
        score = sum(vals) / len(vals) if vals else float(candidate.index == candidates[0].index)
        ranked.append((candidate.index, score))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return {
        "winner_index": ranked[0][0],
        "rankings": [
            {"rank": rank, "candidate_index": index, "score": round(score / 10 if score > 1 else score, 3), "reason": "Fallback score from available MiroShark reviews."}
            for rank, (index, score) in enumerate(ranked, start=1)
        ],
        "selection_rationale": "Fallback ranking used because validator output was unavailable.",
        "next_topics": [],
    }


def _normalize_next_topics(raw: Any, winner_topic: dict[str, Any]) -> list[dict[str, Any]]:
    topics = raw if isinstance(raw, list) else []
    normalized: list[dict[str, Any]] = []
    for index, topic in enumerate(topics[:10], start=1):
        if isinstance(topic, dict):
            normalized.append({
                "id": int(topic.get("id") or index),
                "title": str(topic.get("title") or f"Next topic {index}")[:140],
                "probability": _float_between(topic.get("probability"), 0.0, 1.0, 0.6),
                "impact": int(_float_between(topic.get("impact"), 1, 10, 7)),
                "token_cost": int(_float_between(topic.get("token_cost"), 1000, 100000, 4000)),
                "rationale": str(topic.get("rationale") or "Extends the winning block.")[:500],
            })
        elif isinstance(topic, str):
            normalized.append({
                "id": index,
                "title": topic[:140],
                "probability": 0.6,
                "impact": 7,
                "token_cost": 4000,
                "rationale": "Validator supplied a string topic.",
            })
    while len(normalized) < 10:
        index = len(normalized) + 1
        normalized.append({
            "id": index,
            "title": f"Extend {winner_topic.get('title', 'winning topic')} #{index}",
            "probability": 0.55,
            "impact": 7,
            "token_cost": 4000,
            "rationale": "Fallback next topic derived from the winning Precigenetic base-layer lane.",
        })
    return normalized[:10]


def _float_between(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))
