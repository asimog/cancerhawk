"""Tests for native HTML5 canvas simulation generation."""

from app.simulation_engine import generate_html5_simulations


class FakeAnalysis:
    market_price = 0.42
    headline_catalysts = ["independent organoid validation", "drug-response AUC"]


def test_generate_html5_simulations_fills_empty_peer_review_output():
    sims = generate_html5_simulations(
        paper_text="# Cell Cinema\n\n## Mechanism\n\nVideo-derived trajectories.",
        analysis_result=FakeAnalysis(),
        peer_reviews=[{"summary": "Needs independent validation."}],
        recommended_simulations=[],
    )

    assert len(sims) == 3
    assert all(sim["type"] == "html5_canvas" for sim in sims)
    assert {sim["scene"] for sim in sims} == {
        "trajectory_manifold",
        "counterfactual_perturbation",
        "microenvironment_gradient",
    }
    assert sims[0]["parameters"]["title"] == "Cell Cinema"


def test_generate_html5_simulations_keeps_peer_review_proposal_first():
    sims = generate_html5_simulations(
        paper_text="# Paper\n\nbody",
        analysis_result=FakeAnalysis(),
        peer_reviews=[],
        recommended_simulations=[
            {
                "type": "statistical",
                "description": "Reviewer-requested survival bootstrap.",
                "expected_metrics": ["hazard_ratio"],
            }
        ],
    )

    assert len(sims) == 3
    assert sims[0]["description"] == "Reviewer-requested survival bootstrap."
    assert sims[0]["type"] == "html5_canvas"
