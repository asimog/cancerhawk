"""Unit tests for app.peer_review_engine."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.peer_review_engine import (
    PeerReview,
    _synthesize,
    _review_one,
    reviews_to_dict,
    consolidated_to_dict,
)
from app.prompts import ARCHETYPES


@pytest.fixture
def mock_chat_json():
    """Return a mock chat_json that yields a deterministic review."""
    async def _mock(api_key, model, messages, **kwargs):
        return {
            "recommendation": "accept",
            "confidence": 0.92,
            "summary": "The paper is solid and well-supported.",
            "dimension_scores": {
                "mechanistic_plausibility": 8,
                "experimental_design": 7,
                "evidence_support": 9,
                "statistical_rigor": 8,
                "clarity_of_writing": 7,
            },
            "criticisms": ["Limited sample size."],
            "required_fixes": ["Add more statistical power analysis."],
            "suggested_experiments": ["Run validation on independent cohort."],
            "simulation_proposal": {
                "type": "in_silico",
                "description": "Simulate treatment response using a virtual trial model.",
                "rationale": "To verify the predicted efficacy claim.",
                "expected_metrics": ["response_rate", "survival_improvement"],
            },
        }
    return _mock


@pytest.mark.asyncio
async def test_review_one_parses_response(mock_chat_json):
    archetype = ARCHETYPES[0]  # Oncogene Hunter
    tracker = None  # not needed
    # We'll patch chat_json in openrouter module
    with patch("app.openrouter.chat_json", new=mock_chat_json):
        review = await _review_one(
            api_key="sk-test",
            archetype=archetype,
            paper_text="Sample paper text...",
            model="anthropic/claude-haiku-4.5",
            emit=lambda *a, **k: None,
            tracker=tracker,
            on_call=lambda *a, **k: None,
        )

    assert isinstance(review, PeerReview)
    assert review.archetype_id == archetype["id"]
    assert review.recommendation == "accept"
    assert review.overall_confidence == 0.92
    assert len(review.dimension_scores) == 5
    assert review.simulation_proposal["type"] == "in_silico"


def test_synthesize_acceptance_probability():
    # Create mock reviews
    r1 = PeerReview(
        archetype_id="a1", archetype_name="A", recommendation="accept",
        overall_confidence=0.9, summary="", dimension_scores={},
        criticisms=[], required_fixes=[], suggested_experiments=[],
        simulation_proposal=None,
    )
    r2 = PeerReview(
        archetype_id="a2", archetype_name="B", recommendation="minor_revision",
        overall_confidence=0.8, summary="", dimension_scores={},
        criticisms=[], required_fixes=[], suggested_experiments=[],
        simulation_proposal=None,
    )
    r3 = PeerReview(
        archetype_id="a3", archetype_name="C", recommendation="reject",
        overall_confidence=0.9, summary="", dimension_scores={},
        criticisms=[], required_fixes=[], suggested_experiments=[],
        simulation_proposal=None,
    )
    consolidated = _synthesize([r1, r2, r3])
    # acceptance = (1*0.9 + 0.7*0.8 + 0*0.9) / (0.9+0.8+0.9) = (0.9 + 0.56 + 0) / 2.6 = 1.46/2.6 ≈ 0.5615
    expected = (0.9 + 0.56) / 2.6
    assert abs(consolidated.acceptance_probability - expected) < 1e-4


def test_synthesize_simulation_priority():
    r_accept = PeerReview(
        archetype_id="a1", archetype_name="High", recommendation="accept",
        overall_confidence=0.9, summary="", dimension_scores={},
        criticisms=[], required_fixes=[], suggested_experiments=[],
        simulation_proposal={"type": "sim", "description": "good", "rationale": "", "expected_metrics": []},
    )
    r_reject = PeerReview(
        archetype_id="a2", archetype_name="Low", recommendation="reject",
        overall_confidence=0.5, summary="", dimension_scores={},
        criticisms=[], required_fixes=[], suggested_experiments=[],
        simulation_proposal={"type": "sim", "description": "bad", "rationale": "", "expected_metrics": []},
    )
    consolidated = _synthesize([r_accept, r_reject])
    # The accept one should be prioritized
    assert len(consolidated.recommended_simulations) == 2
    assert consolidated.recommended_simulations[0]["archetype"] == "High"


def test_reviews_to_dict():
    review = PeerReview(
        archetype_id="test", archetype_name="Tester", recommendation="minor_revision",
        overall_confidence=0.75, summary="Test summary", dimension_scores={"a": 5},
        criticisms=["c1"], required_fixes=["f1"], suggested_experiments=["e1"],
        simulation_proposal={"type": "stat", "description": "desc", "rationale": "why", "expected_metrics": ["m1"]},
    )
    d = reviews_to_dict([review])[0]
    assert d["archetype_id"] == "test"
    assert d["recommendation"] == "minor_revision"
    assert d["simulation_proposal"]["type"] == "stat"


def test_consolidated_to_dict():
    # Not testing much beyond structure
    cr = type('ConsolidatedReview', (), {
        'individual_reviews': [],
        'acceptance_probability': 0.8,
        'major_concerns': [],
        'recommended_simulations': [{"type": "sim"}],
        'revision_priorities': ["fix A"],
    })()
    d = consolidated_to_dict(cr)
    assert d["acceptance_probability"] == 0.8
    assert d["recommended_simulations"][0]["type"] == "sim"
