"""Precigenetic CancerHawk base layer.

The source project at https://github.com/Precigenetic/CancerHawk starts from
TCGA/GDC target discovery, with an exploratory notebook querying the GDC cases
endpoint. This module turns that seed into the first interface box for a
CancerHawk block race: exactly ten research topics.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from collections import Counter
from typing import Any

logger = logging.getLogger("cancerhawk.precigenetic_base_layer")

SOURCE_REPO = "https://github.com/Precigenetic/CancerHawk"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
TOPIC_COUNT = 10

CORE_LANES = [
    ("Brain", "glioma immune exclusion and scRNA-seq state topology"),
    ("Blood", "mature B-cell lymphoma antigen escape and target ranking"),
    ("Breast", "breast carcinoma stromal barriers and endocrine resistance"),
    ("Lung", "NSCLC T-cell exhaustion and checkpoint-resistance circuits"),
]

FALLBACK_TOPICS = [
    ("Brain glioma topology", "Use TCGA/GDC brain glioma cases plus scRNA-seq previews to rank immune-excluded niches and druggable stromal targets."),
    ("Blood lymphoma target atlas", "Build a B-cell lymphoma target map from GDC case structure and prioritize antigen-loss-resistant mechanisms."),
    ("Breast stromal resistance", "Connect breast carcinoma transcriptomic states to collagen, CAF, and hypoxia barriers that blunt immune entry."),
    ("Lung checkpoint resistance", "Rank lung cancer mechanisms that stabilize exhausted CD8 T-cell attractors after PD-1 blockade."),
    ("Pan-cancer scRNA barcode bridge", "Use cellular-barcode-level scRNA updates as a bridge between TCGA cohorts and single-cell target discovery."),
    ("Viral-oncology mechanisms", "Search for tumor-virus interactions that create exploitable antigen-presentation or innate-immune vulnerabilities."),
    ("Metabolic immune suppression", "Prioritize lactate, adenosine, and hypoxia pathways that convert tumor metabolism into immune paralysis."),
    ("Early detection targetability", "Identify biomarkers that are both detectable in liquid biopsy and mechanistically tied to targetable cancer states."),
    ("CRISPR perturbation validation", "Design perturb-seq screens that falsify top TCGA-derived targets across tumor and immune compartments."),
    ("Therapy-delivery bottlenecks", "Find target-delivery pairs where nanoparticle, mRNA, or antibody formats solve an otherwise blocked mechanism."),
]


def build_precigenetic_topics(
    research_goal: str,
    *,
    topic_count: int = TOPIC_COUNT,
    fetch_live_gdc: bool | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return exactly ``topic_count`` base topics plus provenance metadata."""
    live_enabled = (
        os.environ.get("CANCERHAWK_GDC_BASE_LAYER_ENABLED", "true").strip().lower()
        in {"1", "true", "yes", "on"}
        if fetch_live_gdc is None
        else fetch_live_gdc
    )
    gdc_summary = _fetch_gdc_summary() if live_enabled else {"ok": False, "reason": "disabled"}
    topics = _topics_from_gdc_summary(research_goal, gdc_summary, topic_count)
    metadata = {
        "source": "precigenetic_cancerhawk",
        "source_repo": SOURCE_REPO,
        "gdc_cases_endpoint": GDC_CASES_ENDPOINT,
        "gdc_summary": gdc_summary,
        "topic_count": len(topics),
    }
    return topics, metadata


def format_base_layer_context(topics: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    rows = [
        "PRECIGENETIC / TCGA BASE LAYER",
        f"Source repo: {metadata.get('source_repo', SOURCE_REPO)}",
        f"GDC endpoint: {metadata.get('gdc_cases_endpoint', GDC_CASES_ENDPOINT)}",
        "The block must begin from these ten target-discovery lanes:",
    ]
    for topic in topics:
        rows.append(
            f"{topic['id']}. {topic['title']} - {topic['research_goal']} "
            f"(lane={topic.get('lane', 'unknown')})"
        )
    return "\n".join(rows)


def _fetch_gdc_summary() -> dict[str, Any]:
    params = {
        "fields": "case_id,primary_site,disease_type,submitter_id",
        "format": "JSON",
        "size": "100",
    }
    url = f"{GDC_CASES_ENDPOINT}?{urllib.parse.urlencode(params)}"
    timeout = float(os.environ.get("CANCERHAWK_GDC_TIMEOUT_SECONDS", "2.5"))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        logger.info("gdc_base_layer_unavailable", extra={"error": str(exc)})
        return {"ok": False, "url": url, "reason": type(exc).__name__, "error": str(exc)[:300]}

    hits = payload.get("data", {}).get("hits", [])
    primary_sites = Counter(str(hit.get("primary_site") or "Unknown") for hit in hits)
    disease_types = Counter(str(hit.get("disease_type") or "Unknown") for hit in hits)
    return {
        "ok": True,
        "url": url,
        "case_count": len(hits),
        "primary_sites": dict(primary_sites.most_common(8)),
        "disease_types": dict(disease_types.most_common(8)),
    }


def _topics_from_gdc_summary(
    research_goal: str,
    gdc_summary: dict[str, Any],
    topic_count: int,
) -> list[dict[str, Any]]:
    goal = research_goal.strip()
    topics: list[dict[str, Any]] = []
    for index, (lane, focus) in enumerate(CORE_LANES, start=1):
        topics.append({
            "id": index,
            "title": f"{lane}: {focus}",
            "research_goal": f"{goal}. Base-layer lane: {lane}. Research {focus} using TCGA/GDC-style target discovery.",
            "lane": lane,
            "source": "precigenetic_tcga_core_lane",
            "probability": 0.75,
            "impact": 8,
            "token_cost": 4500,
            "rationale": "Core lane from the Precigenetic CancerHawk README: start with Brain and Blood, with Breast and Lung as strong starting points.",
        })

    live_sites = list((gdc_summary.get("primary_sites") or {}).keys()) if gdc_summary.get("ok") else []
    live_diseases = list((gdc_summary.get("disease_types") or {}).keys()) if gdc_summary.get("ok") else []
    for site in live_sites:
        if len(topics) >= topic_count:
            break
        disease = live_diseases[(len(topics) - len(CORE_LANES)) % max(1, len(live_diseases))] if live_diseases else "pan-cancer"
        topics.append({
            "id": len(topics) + 1,
            "title": f"GDC-derived {site} target-discovery lane",
            "research_goal": f"{goal}. Analyze {site} / {disease} as a GDC-derived oncology target-discovery lane.",
            "lane": site,
            "source": "live_gdc_cases_endpoint",
            "probability": 0.68,
            "impact": 7,
            "token_cost": 4200,
            "rationale": "Generated from the live GDC cases endpoint exposed by the Precigenetic exploratory notebook.",
        })

    for title, description in FALLBACK_TOPICS:
        if len(topics) >= topic_count:
            break
        topics.append({
            "id": len(topics) + 1,
            "title": title,
            "research_goal": f"{goal}. {description}",
            "lane": "fallback_tcga",
            "source": "precigenetic_fallback_topic",
            "probability": 0.62,
            "impact": 7,
            "token_cost": 4000,
            "rationale": "Fallback topic preserving the Precigenetic TCGA/GDC starting map when live GDC metadata is unavailable.",
        })

    return topics[:topic_count]
