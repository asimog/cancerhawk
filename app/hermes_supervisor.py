"""Hermes supervisor for CancerHawk runs.

This module is the Railway-side orchestrator: it owns the run lifecycle,
executes the full MOTO -> analysis -> peer review -> simulation -> publish
pipeline, and hands completed artifacts to the GitHub publisher so Vercel can
rebuild the website.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .analysis_engine import run_analysis_engine
from .block_race_engine import run_block_race
from .paysh import PAY_SH_SANDBOX, research_enrichment, estimate_enrichment_cost
from .publisher import hydrate_results_from_github, load_previous_block_context, publish_block, try_git_publish, stage_block
from .simulation_engine import generate_html5_simulations
from .moltbook import post_block_race_invite, post_research_result
from .token_tracker import APICall, TokenTracker

logger = logging.getLogger("cancerhawk.hermes")

EmitFn = Callable[[str, str, dict | None], Awaitable[None]]
CallEmitFn = Callable[[APICall], Awaitable[None]]


@dataclass
class HermesRunConfig:
    api_key: str
    research_goal: str
    n_submitters: int
    auto_publish: bool
    git_push: bool
    models: dict[str, str]
    job_id: str | None = None
    stage: bool = False
    enable_paysh: bool = False
    publication_batch_id: str | None = None
    candidate_index: int | None = None
    batch_size: int | None = None
    wallet_address: str | None = None


@dataclass
class HermesRunResult:
    title: str
    market_price: float
    block: int | None
    result_url: str | None
    stats: dict[str, Any]
    calls: list[dict[str, Any]]
    git_status: str | None = None


class HermesSupervisor:
    """Top-level CancerHawk agent that supervises the research run."""

    def __init__(self, *, emit: EmitFn, on_call: CallEmitFn, tracker: TokenTracker | None = None) -> None:
        self.emit = emit
        self.on_call = on_call
        self.tracker = tracker or TokenTracker()

    async def run(self, cfg: HermesRunConfig) -> HermesRunResult:
        tracker = self.tracker
        publish_meta: dict[str, Any] | None = None
        git_status: str | None = None

        await self.emit(
            "hermes",
            "Hermes supervisor started: overseeing MOTO, peer review, simulations, repo publish",
            {
                "auto_publish": cfg.auto_publish,
                "git_push": cfg.git_push,
                "models": cfg.models,
            },
        )

        hydration_status = hydrate_results_from_github()
        await self.emit("hermes", hydration_status, {"hydration_status": hydration_status})

        previous_block_context = load_previous_block_context()
        if previous_block_context:
            await self.emit(
                "prior_blocks",
                "Hermes loaded prior CancerHawk blocks for continuity",
                {"chars": len(previous_block_context)},
            )

        enrichment_context = ""
        if cfg.enable_paysh:
            await self.emit(
                "paysh",
                f"Pay.sh enrichment enabled (sandbox={PAY_SH_SANDBOX}). Searching for research context...",
                {"sandbox": PAY_SH_SANDBOX},
            )
            enrichment_context = research_enrichment(cfg.research_goal)
            if enrichment_context:
                await self.emit(
                    "paysh",
                    "Pay.sh enrichment complete.",
                    {"chars": len(enrichment_context)},
                )
                previous_block_context = enrichment_context + "\n\n" + previous_block_context
            else:
                await self.emit(
                    "paysh",
                    "Pay.sh enrichment returned no results or is in sandbox mode.",
                    {"sandbox": PAY_SH_SANDBOX},
                )

        logger.info("stage_start", extra={"stage": "block_race", "supervisor": "hermes"})
        race = await run_block_race(
            api_key=cfg.api_key,
            research_goal=cfg.research_goal,
            models=cfg.models,
            emit=self.emit,
            tracker=tracker,
            on_call=self.on_call,
            previous_block_context=previous_block_context,
            run_id=cfg.job_id or cfg.publication_batch_id,
        )
        logger.info("stage_end", extra={"stage": "block_race", "supervisor": "hermes"})
        paper = race.winner.paper
        paper_text = paper.full_text()
        await self.emit(
            "paper_done",
            f"Block-race winner compiled: candidate {race.winner.index} · '{paper.title}' · "
            f"{len(paper.sections)} sections · {len(paper_text.split())} words",
            {
                "title": paper.title,
                "winner_index": race.winner.index,
                "worker_id": race.winner.worker_id,
                "worker_wallet": race.winner.worker_wallet,
                "section_count": len(paper.sections),
                "accepted_count": len(paper.accepted_submissions),
                "rounds_run": getattr(paper, "rounds_run", 0),
                "convergence_reason": getattr(paper, "convergence_reason", ""),
                "candidate_count": len(race.candidates),
            },
        )

        logger.info("stage_start", extra={"stage": "analysis_engine", "supervisor": "hermes"})
        analysis = await run_analysis_engine(
            api_key=cfg.api_key,
            paper_text=paper_text,
            archetype_model=cfg.models["archetype"],
            emit=self.emit,
            tracker=tracker,
            on_call=self.on_call,
        )
        logger.info("stage_end", extra={"stage": "analysis_engine", "supervisor": "hermes"})

        logger.info("stage_start", extra={"stage": "peer_review", "supervisor": "hermes"})
        peer_reviews_dict = race.reviews
        recommended_simulations = _recommended_simulations_from_reviews(peer_reviews_dict)
        await self.emit(
            "review_complete",
            f"MiroShark race review complete · {len(peer_reviews_dict)} independent reviews · "
            f"{len(recommended_simulations)} simulation proposals",
            {
                "review_count": len(peer_reviews_dict),
                "simulation_count": len(recommended_simulations),
                "winner_index": race.winner.index,
            },
        )
        logger.info("stage_end", extra={"stage": "peer_review", "supervisor": "hermes"})

        logger.info("stage_start", extra={"stage": "simulation_generation", "supervisor": "hermes"})
        await self.emit("simulate", "Hermes generating browser-native simulations from peer review", None)
        simulations = generate_html5_simulations(
            paper_text=paper_text,
            analysis_result=analysis,
            peer_reviews=peer_reviews_dict,
            recommended_simulations=recommended_simulations,
        )
        await self.emit(
            "simulate_done",
            f"Hermes generated {len(simulations)} runnable simulations",
            {"simulation_count": len(simulations)},
        )
        logger.info("stage_end", extra={"stage": "simulation_generation", "supervisor": "hermes"})

        derived_topics = race.next_topics
        await self.emit(
            "derive",
            "MOTO validator emitted next-block topics",
            {"topics": derived_topics, "topic_count": len(derived_topics)},
        )
        race_metadata = race.race_metadata()

        publish_meta = None
        if getattr(cfg, 'stage', False) and cfg.job_id:
            logger.info("stage_start", extra={"stage": "stage", "supervisor": "hermes"})
            await self.emit("stage", "Hermes staging block artifacts for later publication", {"job_id": cfg.job_id})
            try:
                publish_meta = stage_block(
                    paper=paper,
                    analysis=analysis,
                    derived_topics=derived_topics,
                    research_goal=cfg.research_goal,
                    models=cfg.models,
                    peer_reviews=peer_reviews_dict,
                    simulations=simulations,
                    race_metadata=race_metadata,
                    job_id=cfg.job_id,
                    git_push=cfg.git_push,
                    publication_batch_id=cfg.publication_batch_id,
                    candidate_index=cfg.candidate_index,
                    batch_size=cfg.batch_size,
                    wallet_address=cfg.wallet_address,
                )
            except Exception as exc:
                logger.error("staging_failed", extra={"job_id": cfg.job_id, "error": str(exc)})
                raise
            await self.emit("stage_done", f"Hermes staged artifacts for job {cfg.job_id}", publish_meta)
            logger.info("stage_end", extra={"stage": "stage", "supervisor": "hermes"})
        elif cfg.auto_publish:
            logger.info("stage_start", extra={"stage": "publish", "supervisor": "hermes"})
            await self.emit("publish", "Hermes writing block bundle to results/", None)
            publish_meta = publish_block(
                paper=paper,
                analysis=analysis,
                derived_topics=derived_topics,
                research_goal=cfg.research_goal,
                models=cfg.models,
                peer_reviews=peer_reviews_dict,
                simulations=simulations,
                race_metadata=race_metadata,
            )
            await self.emit(
                "publish_done",
                f"Hermes wrote block {publish_meta['block']} -> {publish_meta['path']}",
                publish_meta,
            )
            if cfg.git_push:
                await self.emit("git", "Hermes checking out GitHub repo and preparing commit", None)
                git_status = try_git_publish(publish_meta["block"])
                await self.emit("git", git_status, {"status": git_status})
            logger.info("stage_end", extra={"stage": "publish", "supervisor": "hermes"})

        # Post block results to Moltbook
        try:
            block_n = publish_meta.get("block") if publish_meta else None
            result_url = f"/results/block-{block_n}/paper.html" if block_n else None
            await self.emit("moltbook", "Hermes posting to Moltbook...", None)
            await post_research_result(
                title=f"CancerHawk Block {block_n or '?'}: {paper.title}",
                paper_title=paper.title,
                block=block_n or 0,
                market_price=analysis.market_price,
                research_goal=cfg.research_goal,
                result_url=result_url,
            )
            await post_block_race_invite(
                block=block_n or 0,
                paper_title=paper.title,
                market_price=analysis.market_price,
                research_goal=cfg.research_goal,
            )
            await self.emit("moltbook", "Hermes posted block results to Moltbook", None)
        except Exception as exc:
            logger.error("moltbook_post_error", extra={"error": str(exc)})
            await self.emit("moltbook", f"Moltbook post failed: {exc}", None)

        stats = tracker.stats()
        block_n = publish_meta.get("block") if publish_meta else None
        return HermesRunResult(
            title=paper.title,
            market_price=analysis.market_price,
            block=block_n,
            result_url=(f"/results/block-{block_n}/paper.html" if block_n else None),
            stats=stats,
            calls=[c.to_dict() for c in tracker.calls],
            git_status=git_status,
        )


def _recommended_simulations_from_reviews(peer_reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for review in peer_reviews:
        proposal = review.get("simulation_proposal")
        if not isinstance(proposal, dict):
            continue
        if not proposal.get("description") or not proposal.get("type"):
            continue
        proposals.append(proposal)
    return proposals[:3]
