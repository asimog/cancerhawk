"""Full MOTO paper engine — adaptive aggregation + CancerHawk-style compiler.

This module integrates MOTO aggregator prompts (submitter/validator) with the
existing CancerHawk compiler (outline + section writing). The aggregator
runs adaptive loops until convergence, then the compiler produces the final
paper.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .openrouter import CallEmitFn, chat, chat_json
from .token_tracker import TokenTracker

# MOTO aggregator prompts (backend)
from backend.aggregator.prompts.submitter_prompts import build_submitter_prompt
from backend.aggregator.prompts.validator_prompts import (
    build_validator_prompt,
    build_validator_dual_prompt,
    build_validator_triple_prompt,
)

# CancerHawk compiler prompts (local)
from .prompts import compiler_outline_prompt, compiler_section_prompt, DOMAIN_FRAME

EmitFn = Callable[[str, str, dict | None], Awaitable[None]]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


# Convergence thresholds
MIN_ACCEPTED_FLOOR = _env_int("CANCERHAWK_MIN_ACCEPTED", 3)
SATURATION_ROUNDS = _env_int("CANCERHAWK_SATURATION_ROUNDS", 2)
PLATEAU_ROUNDS = _env_int("CANCERHAWK_PLATEAU_ROUNDS", 3)
MAX_API_CALLS_SOFT = _env_int("CANCERHAWK_MAX_CALLS", 400)
MAX_WALL_CLOCK_SECONDS = _env_int("CANCERHAWK_MAX_WALL_CLOCK", 3600)


@dataclass
class Paper:
    title: str
    sections: list[dict]  # [{"heading": str, "content": str}]
    accepted_submissions: list[str]
    rejections: list[dict] = field(default_factory=list)
    rounds_run: int = 0
    convergence_reason: str = ""

    def full_text(self) -> str:
        body = "\n\n".join(f"## {s['heading']}\n\n{s['content']}" for s in self.sections)
        return f"# {self.title}\n\n{body}"


def _check_convergence(
    *,
    accepted_count: int,
    rounds_run: int,
    accepts_per_round: list[int],
    novelty_per_round: list[float],
    api_calls: int,
    elapsed_s: float,
) -> tuple[bool, str]:
    if MAX_API_CALLS_SOFT and api_calls >= MAX_API_CALLS_SOFT:
        return True, f"safety_guard:api_calls>={MAX_API_CALLS_SOFT}"
    if MAX_WALL_CLOCK_SECONDS and elapsed_s >= MAX_WALL_CLOCK_SECONDS:
        return True, f"safety_guard:wall_clock>={MAX_WALL_CLOCK_SECONDS}s"
    if accepted_count < MIN_ACCEPTED_FLOOR:
        return False, ""
    if rounds_run >= SATURATION_ROUNDS:
        recent = accepts_per_round[-SATURATION_ROUNDS:]
        if all(c == 0 for c in recent):
            return True, f"saturation:{SATURATION_ROUNDS}_rounds_no_accepts"
    if len(novelty_per_round) >= PLATEAU_ROUNDS:
        window = novelty_per_round[-PLATEAU_ROUNDS:]
        non_increasing = all(window[i] <= window[i-1] for i in range(1, len(window)))
        if non_increasing:
            return True, f"plateau:novelty_flat_{PLATEAU_ROUNDS}_rounds"
    return False, ""


def _normalize_section_specs(raw_sections: object) -> list[dict[str, str]]:
    """Coerce outline sections into the shape compiler_section_prompt needs."""
    if not isinstance(raw_sections, list):
        return []
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(raw_sections, start=1):
        if isinstance(item, dict):
            heading = str(item.get("heading") or item.get("title") or f"Section {index}").strip()
            summary = str(item.get("summary") or item.get("intent") or item.get("description") or "").strip()
        elif isinstance(item, str):
            heading = item.strip() or f"Section {index}"
            summary = heading
        else:
            continue
        normalized.append({
            "heading": heading or f"Section {index}",
            "summary": summary or "Develop this section from the accepted research aggregate.",
        })
    return normalized


async def _generate_submission(
    api_key: str,
    model: str,
    research_goal: str,
    prior_accepted: list[str],
    prior_rejections: list[str],
    previous_block_context: str,
    tracker: TokenTracker,
    on_call: CallEmitFn,
) -> str:
    """Generate a single research direction via MOTO submitter prompt."""
    parts = []
    if prior_accepted:
        snippets = []
        for i, sub in enumerate(prior_accepted, start=1):
            stripped = (sub or "").strip()
            lines = stripped.splitlines() if stripped else []
            head = lines[0][:200] if lines else "(empty)"
            snippets.append(f"  [{i}] {head}")
        parts.append(
            "\nALREADY-ACCEPTED RESEARCH DIRECTIONS (extend frontier, do NOT duplicate):\n" + "\n".join(snippets)
        )
    if previous_block_context.strip():
        parts.append(
            "\nPREVIOUS CANCERHAWK BLOCKS (cite as `CancerHawk Block N` when relevant):\n" + previous_block_context[:6000]
        )
    if prior_rejections:
        parts.append("\nPRIOR REJECTIONS (avoid these failure modes):\n- " + "\n- ".join(prior_rejections[-5:]))

    context = "\n\n".join(parts) if parts else ""
    prompt_str = build_submitter_prompt(
        user_prompt=research_goal,
        context=context,
        rag_evidence="",
    )
    # MOTO submitter prompt is a single user message
    response = await chat(
        api_key,
        model,
        [{"role": "user", "content": prompt_str}],
        temperature=0.85,
        role="submitter",
        tracker=tracker,
        on_call=on_call,
        max_tokens=25000,
    )
    try:
        parsed = json.loads(response)
        return parsed.get("submission", response)
    except json.JSONDecodeError:
        return response


async def _validate_batch(
    api_key: str,
    model: str,
    research_goal: str,
    shared_context: str,
    submissions: list[str],
    tracker: TokenTracker,
    on_call: CallEmitFn,
) -> list[dict]:
    """Validate a batch of submissions using MOTO validator prompts."""
    batch_len = len(submissions)
    if batch_len == 1:
        prompt_str = build_validator_prompt(research_goal, submissions[0], shared_context, "")
        resp = await chat_json(
            api_key,
            model,
            [{"role": "user", "content": prompt_str}],
            temperature=0.3,
            role="validator",
            tracker=tracker,
            on_call=on_call,
            max_tokens=2000,
        )
        return [resp] if isinstance(resp, dict) else []
    elif batch_len == 2:
        prompt_str = build_validator_dual_prompt(research_goal, submissions, shared_context, "")
        resp = await chat_json(
            api_key,
            model,
            [{"role": "user", "content": prompt_str}],
            temperature=0.3,
            role="validator",
            tracker=tracker,
            on_call=on_call,
            max_tokens=3000,
        )
        if isinstance(resp, dict) and "decisions" in resp:
            return resp["decisions"]
        return []
    elif batch_len == 3:
        prompt_str = build_validator_triple_prompt(research_goal, submissions, shared_context, "")
        resp = await chat_json(
            api_key,
            model,
            [{"role": "user", "content": prompt_str}],
            temperature=0.3,
            role="validator",
            tracker=tracker,
            on_call=on_call,
            max_tokens=4000,
        )
        if isinstance(resp, dict) and "decisions" in resp:
            return resp["decisions"]
        return []
    else:
        raise ValueError("Validator batch size must be 1, 2, or 3")


async def run_paper_engine(
    api_key: str,
    research_goal: str,
    models: dict,
    n_submitters: int,
    emit: EmitFn,
    tracker: TokenTracker,
    on_call: CallEmitFn,
    previous_block_context: str = "",
) -> Paper:
    """Full MOTO pipeline: adaptive aggregation → outline → section writing → abstract."""
    accepted_submissions: list[str] = []
    rejection_feedback: list[str] = []  # global list of recent rejection summaries
    accepts_per_round: list[int] = []
    novelty_per_round: list[float] = []
    round_num = 0
    started_at = time.time()
    convergence_reason = ""

    await emit(
        "brainstorm",
        "MOTO aggregator: starting adaptive brainstorming (batch validation, empirical provenance)",
        {"n_submitters": n_submitters, "min_accepted": MIN_ACCEPTED_FLOOR},
    )

    # ── Phase 1: Adaptive aggregation ────────────────────────────────────────
    while True:
        round_num += 1
        await emit(
            "brainstorm",
            f"Round {round_num}: spawning {n_submitters} parallel submitters "
            f"· aggregate size {len(accepted_submissions)}",
            {
                "round": round_num,
                "accepted_so_far": len(accepted_submissions),
                "elapsed_s": round(time.time() - started_at, 1),
            },
        )

        # Parallel submissions
        generation_tasks = [
            _generate_submission(
                api_key=api_key,
                model=models["submitter"],
                research_goal=research_goal,
                prior_accepted=accepted_submissions,
                prior_rejections=rejection_feedback[-5:],
                previous_block_context=previous_block_context,
                tracker=tracker,
                on_call=on_call,
            )
            for _ in range(n_submitters)
        ]
        raw_submissions = await asyncio.gather(*generation_tasks, return_exceptions=True)

        valid_submissions: list[str] = []
        for i, sub in enumerate(raw_submissions):
            if isinstance(sub, Exception):
                await emit("validate", f"Submitter {i+1} failed: {sub}", {"error": str(sub)})
            else:
                valid_submissions.append(sub or "")

        shared_context = "\n\n".join(accepted_submissions)

        # Batched validator calls
        batch_validations: list[dict] = []
        batch_size = len(valid_submissions)
        if batch_size == 1:
            decisions = await _validate_batch(
                api_key=api_key,
                model=models["validator"],
                research_goal=research_goal,
                shared_context=shared_context,
                submissions=[valid_submissions[0]],
                tracker=tracker,
                on_call=on_call,
            )
            if decisions:
                batch_validations = decisions
        elif batch_size == 2:
            decisions = await _validate_batch(
                api_key=api_key,
                model=models["validator"],
                research_goal=research_goal,
                shared_context=shared_context,
                submissions=valid_submissions,
                tracker=tracker,
                on_call=on_call,
            )
            if decisions:
                batch_validations = decisions
        elif batch_size == 3:
            decisions = await _validate_batch(
                api_key=api_key,
                model=models["validator"],
                research_goal=research_goal,
                shared_context=shared_context,
                submissions=valid_submissions,
                tracker=tracker,
                on_call=on_call,
            )
            if decisions:
                batch_validations = decisions
        else:
            for sub in valid_submissions:
                v = await _validate_batch(
                    api_key=api_key,
                    model=models["validator"],
                    research_goal=research_goal,
                    shared_context=shared_context,
                    submissions=[sub],
                    tracker=tracker,
                    on_call=on_call,
                )
                if v:
                    batch_validations.extend(v)

        round_accepts = 0
        round_novelty_scores: list[float] = []

        for dec_idx, decision in enumerate(batch_validations):
            if not isinstance(decision, dict):
                continue
            # MOTO validator returns {"decision": "accept"|"reject", "reasoning": "...", "summary": "...", "scores": {...} optional}
            scores = decision.get("scores") or {}
            nov = scores.get("novelty") if isinstance(scores, dict) else None
            if isinstance(nov, (int, float)):
                round_novelty_scores.append(float(nov))
            else:
                round_novelty_scores.append(0.0)

            if decision.get("decision") == "accept":
                accepted_submissions.append(valid_submissions[dec_idx])
                round_accepts += 1
                await emit(
                    "validate",
                    f"✓ accepted submission {len(accepted_submissions)} "
                    f"(round {round_num}): {decision.get('reasoning', '')[:120]}",
                    {"scores": scores, "accepted_total": len(accepted_submissions)},
                )
            else:
                steering = decision.get("summary") or decision.get("reasoning") or ""
                rejection_feedback.append(steering[:200])
                await emit(
                    "validate",
                    f"✗ rejected — {steering[:120]}",
                    {"scores": scores},
                )

        accepts_per_round.append(round_accepts)
        round_avg_novelty = (
            sum(round_novelty_scores) / len(round_novelty_scores)
            if round_novelty_scores else 0.0
        )
        novelty_per_round.append(round_avg_novelty)

        api_calls = len(tracker.calls) if hasattr(tracker, "calls") else 0
        elapsed_s = time.time() - started_at

        await emit(
            "brainstorm",
            f"Round {round_num} closed · +{round_accepts} accepted · "
            f"avg_novelty={round_avg_novelty:.1f} · total_accepted={len(accepted_submissions)}",
            {
                "round": round_num,
                "round_accepts": round_accepts,
                "round_avg_novelty": round_avg_novelty,
                "total_accepted": len(accepted_submissions),
                "api_calls": api_calls,
                "elapsed_s": round(elapsed_s, 1),
            },
        )

        should_stop, reason = _check_convergence(
            accepted_count=len(accepted_submissions),
            rounds_run=round_num,
            accepts_per_round=accepts_per_round,
            novelty_per_round=novelty_per_round,
            api_calls=api_calls,
            elapsed_s=elapsed_s,
        )
        if should_stop:
            convergence_reason = reason
            await emit(
                "brainstorm",
                f"Converged after {round_num} rounds · {len(accepted_submissions)} accepted · "
                f"reason={reason}",
                {
                    "rounds": round_num,
                    "accepted": len(accepted_submissions),
                    "reason": reason,
                    "accepts_per_round": accepts_per_round,
                    "novelty_per_round": novelty_per_round,
                },
            )
            break

    if not accepted_submissions:
        raise RuntimeError("No submissions were accepted across the adaptive run")

    # ── Phase 2: Compile outline ─────────────────────────────────────────────
    await emit("compile", "Compiling paper outline from full research aggregate", None)
    outline_messages = compiler_outline_prompt(accepted_submissions, research_goal, previous_block_context)
    outline = await chat_json(
        api_key=api_key,
        model=models["compiler"],
        messages=outline_messages,
        temperature=0.4,
        role="compiler_outline",
        tracker=tracker,
        on_call=on_call,
    )
    if isinstance(outline, list) and outline:
        outline = outline[0]
    if not isinstance(outline, dict):
        raise RuntimeError(f"Compiler outline returned unexpected type: {type(outline).__name__}")
    title = outline.get("title", "Untitled CancerHawk Paper")
    section_specs = _normalize_section_specs(outline.get("sections"))
    if not section_specs:
        raise RuntimeError("Compiler outline returned no sections")

    await emit(
        "compile",
        f"Outline ready: '{title}' · {len(section_specs)} sections",
        {"title": title, "section_count": len(section_specs)},
    )

    # ── Phase 3: Write sections sequentially ─────────────────────────────────
    written: list[dict] = []
    for i, spec in enumerate(section_specs):
        await emit(
            "compile",
            f"Writing section {i + 1}/{len(section_specs)}: {spec.get('heading', '')}",
            {"section_index": i, "heading": spec.get("heading", "")},
        )
        section_messages = compiler_section_prompt(
            title=title,
            section=spec,
            prior_sections=written,
            research_goal=research_goal,
            previous_block_context=previous_block_context,
        )
        content = await chat(
            api_key=api_key,
            model=models["compiler"],
            messages=section_messages,
            temperature=0.55,
            role="compiler_section",
            tracker=tracker,
            on_call=on_call,
            max_tokens=2200,
        )
        written.append({
            "heading": spec.get("heading", f"Section {i + 1}"),
            "content": content.strip(),
        })

    # ── Phase 4: Generate abstract ───────────────────────────────────────────
    await emit("compile", "Generating paper abstract", None)
    # Build simple abstract prompt using DOMAIN_FRAME
    sections_summary = "\n\n".join(f"## {s['heading']}\n{s['content'][:1500]}" for s in written)
    abstract_user = (
        f"PAPER TITLE: {title}\n"
        f"RESEARCH GOAL: {research_goal}\n\n"
        f"PAPER CONTENT (summarized):\n{sections_summary}\n\n"
        "Write an abstract (150-250 words) that summarizes the paper's key findings and implications."
    )
    abstract_messages = [
        {"role": "system", "content": DOMAIN_FRAME},
        {"role": "user", "content": abstract_user},
    ]
    abstract_text = await chat(
        api_key=api_key,
        model=models["compiler"],
        messages=abstract_messages,
        temperature=0.5,
        role="compiler_abstract",
        tracker=tracker,
        on_call=on_call,
        max_tokens=600,
    )
    abstract_text = abstract_text.strip()
    # Prepend abstract as first section
    written.insert(0, {"heading": "Abstract", "content": abstract_text})

    return Paper(
        title=title,
        sections=written,
        accepted_submissions=accepted_submissions,
        rejections=[],
        rounds_run=round_num,
        convergence_reason=convergence_reason,
    )
