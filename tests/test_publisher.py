"""Unit tests for app.publisher HTML rendering."""

import html
import json

from app.publisher import (
    _render_peer_reviews,
    _render_simulations,
    _archetype_table,
    _topics_table,
)


def test_render_peer_reviews_empty():
    html_out = _render_peer_reviews([])
    assert "No peer reviews available" in html_out


def test_render_peer_reviews_with_one_review():
    review = {
        "archetype_name": "Oncogene Hunter",
        "recommendation": "accept",
        "confidence": 0.95,
        "summary": "Solid paper with strong evidence.",
        "dimension_scores": {
            "mechanistic_plausibility": 9,
            "experimental_design": 8,
            "evidence_support": 9,
            "statistical_rigor": 8,
            "clarity_of_writing": 7,
        },
        "criticisms": ["Minor typo in section 2."],
        "required_fixes": ["Fix typo."],
        "suggested_experiments": ["Validate in another cell line."],
        "simulation_proposal": {
            "type": "in_silico",
            "description": "Virtual trial with 1000 patients.",
            "rationale": "Confirm efficacy prediction.",
            "expected_metrics": ["response_rate"],
        },
    }
    html_out = _render_peer_reviews([review])

    # Should contain acceptance banner
    assert "Peer review acceptance probability:" in html_out
    assert "100%" in html_out  # accept weight 1.0 * confidence 0.95 / 0.95 = 1

    # Should contain archetype name escaped
    assert "Oncogene Hunter" in html_out

    # Should contain dimension scores (numbers)
    assert "9" in html_out
    assert "8" in html_out

    # Should contain simulation card
    assert "in_silico" in html_out
    assert "Virtual trial" in html_out

    # Ensure no raw < or > from any field
    assert "<script>" not in html_out
    assert "&lt;" not in html_out  # we didn't escape the archetype name? It should be escaped via html.escape
    # The function uses html.escape on archetype_name and summaries etc.
    # So any dangerous characters should be escaped.


def test_render_peer_reviews_xss_protection():
    malicious_review = {
        "archetype_name": "<script>alert('xss')</script>",
        "recommendation": "accept",
        "confidence": 0.5,
        "summary": "<img src=x onerror=alert(1)>",
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
        "simulation_proposal": None,
    }
    html_out = _render_peer_reviews([malicious_review])
    # The script tag should be escaped
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out
    assert "<img" not in html_out
    assert "&lt;img" in html_out


def test_render_simulations_empty():
    html_out = _render_simulations([])
    assert "No simulation proposals" in html_out


def test_render_simulations_with_entry():
    sims = [
        {
            "type": "statistical",
            "description": "Bootstrap resampling of the dataset.",
            "rationale": "Assess robustness to sample variance.",
            "expected_metrics": ["p_value", "confidence_interval"],
        }
    ]
    html_out = _render_simulations(sims)
    assert "statistical" in html_out
    assert "Bootstrap resampling" in html_out
    assert "p_value" in html_out


def test_archetype_table_renders():
    archetypes = [
        {
            "archetype_name": "Test Archetype",
            "scores": {
                "clinical_viability": 8,
                "regulatory_risk": 3,
                "market_potential": 7,
                "patient_impact": 9,
                "novelty": 6,
                "falsifiability": 5,
            },
            "verdict": "Promising but needs more validation.",
        }
    ]
    html_out = _archetype_table(archetypes)
    assert "Test Archetype" in html_out
    assert "8" in html_out  # clinical viability
    assert "Promising but needs more validation" in html_out


def test_topics_table_renders():
    topics = [
        {
            "id": "T1",
            "title": "Next block topic",
            "probability": 0.75,
            "impact": 9,
            "token_cost": 5000,
            "rationale": "Because it follows.",
        }
    ]
    html_out = _topics_table(topics)
    assert "T1" in html_out
    assert "Next block topic" in html_out
    assert "0.75" in html_out
