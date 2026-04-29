"""Native HTML5 canvas simulation generation for CancerHawk blocks.

The peer-review engine may return prose simulation proposals. This module turns
the completed paper and review context into runnable browser-native scenes that
can be published alongside each block without a separate build step.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


def generate_html5_simulations(
    *,
    paper_text: str,
    analysis_result: Any,
    peer_reviews: list[dict],
    recommended_simulations: list[dict] | None = None,
) -> list[dict]:
    """Return runnable HTML5 canvas simulation specs for the published page.

    Peer reviews are still the source of truth for scientific pressure-testing,
    but they are often conservative and may not emit a full runnable proposal.
    In that case we synthesize a small, deterministic set of visual experiments
    from the current paper so the Simulations tab is never empty.
    """

    base_specs = _normalise_peer_review_simulations(recommended_simulations or [])
    title = _extract_title(paper_text)
    seed = _seed_from_text(paper_text)
    critique = _top_review_critique(peer_reviews)
    catalysts = list(getattr(analysis_result, "headline_catalysts", []) or [])
    market_price = float(getattr(analysis_result, "market_price", 0.5) or 0.5)

    generated = [
        {
            "id": "trajectory-manifold",
            "title": "Cell Trajectory Manifold",
            "type": "html5_canvas",
            "description": (
                "Animates live-cell state trajectories as glowing paths through a "
                "latent morphology manifold, mirroring the paper's cell-cinema thesis."
            ),
            "rationale": (
                "Tests whether the claimed video-derived state space produces separable "
                "trajectory fingerprints rather than a visually plausible but mixed cloud."
            ),
            "expected_metrics": [
                "trajectory separation",
                "state-transition smoothness",
                "phenotype-cluster stability",
            ],
            "scene": "trajectory_manifold",
            "seed": seed,
            "parameters": {
                "title": title,
                "confidence": market_price,
                "critique": critique,
                "catalyst": catalysts[0] if catalysts else "",
            },
        },
        {
            "id": "counterfactual-perturbation",
            "title": "Counterfactual Drug Perturbation",
            "type": "html5_canvas",
            "description": (
                "Shows paired control and treated organoid trajectories diverging under "
                "a simulated perturbation pulse."
            ),
            "rationale": (
                "Makes the central falsifier visible: a useful simulator must predict "
                "post-treatment divergence before it is observed experimentally."
            ),
            "expected_metrics": [
                "predicted-vs-observed divergence",
                "response latency",
                "escape-trajectory frequency",
            ],
            "scene": "counterfactual_perturbation",
            "seed": seed + 17,
            "parameters": {
                "title": title,
                "confidence": market_price,
                "critique": critique,
                "catalyst": catalysts[1] if len(catalysts) > 1 else "",
            },
        },
        {
            "id": "microenvironment-gradient",
            "title": "Tumor Microenvironment Gradient",
            "type": "html5_canvas",
            "description": (
                "Maps tumor cells moving through oxygen, immune, and stromal gradients "
                "to expose where video-only predictions may lose clinical context."
            ),
            "rationale": (
                "Addresses the strongest peer-review concern: patient response depends "
                "on microenvironment and immune context, not only single-cell video."
            ),
            "expected_metrics": [
                "gradient sensitivity",
                "immune-contact escape",
                "context-dependent phenotype switch rate",
            ],
            "scene": "microenvironment_gradient",
            "seed": seed + 31,
            "parameters": {
                "title": title,
                "confidence": market_price,
                "critique": critique,
                "catalyst": catalysts[2] if len(catalysts) > 2 else "",
            },
        },
    ]

    merged = base_specs + [s for s in generated if s["id"] not in {b.get("id") for b in base_specs}]
    return merged[:3]


def _normalise_peer_review_simulations(simulations: list[dict]) -> list[dict]:
    normalised = []
    for idx, sim in enumerate(simulations, start=1):
        if not isinstance(sim, dict):
            continue
        description = str(sim.get("description") or "").strip()
        if not description:
            continue
        sim_id = _slugify(str(sim.get("id") or sim.get("type") or f"review-simulation-{idx}"))
        normalised.append(
            {
                "id": sim_id,
                "title": str(sim.get("title") or sim.get("type") or f"Peer Review Simulation {idx}"),
                "type": "html5_canvas",
                "description": description,
                "rationale": str(sim.get("rationale") or "Recommended by the peer-review panel."),
                "expected_metrics": [
                    str(m) for m in sim.get("expected_metrics", []) if str(m).strip()
                ][:5],
                "scene": str(sim.get("scene") or "trajectory_manifold"),
                "seed": idx * 101,
                "parameters": dict(sim.get("parameters") or {}),
            }
        )
    return normalised


def _extract_title(paper_text: str) -> str:
    for line in paper_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "CancerHawk Simulation"


def _seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10000


def _top_review_critique(peer_reviews: list[dict]) -> str:
    for review in peer_reviews:
        criticisms = review.get("criticisms") if isinstance(review, dict) else None
        if criticisms:
            return str(criticisms[0])[:220]
        summary = review.get("summary") if isinstance(review, dict) else None
        if summary:
            return str(summary)[:220]
    return "Validate that simulated trajectories predict unseen experimental outcomes."


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "simulation"
