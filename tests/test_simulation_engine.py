"""Tests for native browser simulation generation (2D canvas + Three.js)."""

from app.simulation_engine import generate_html5_simulations


class FakeAnalysis:
    market_price = 0.42
    headline_catalysts = ["independent organoid validation", "drug-response AUC"]


def test_generate_html5_simulations_fills_empty_peer_review_output():
    """With no peer-review simulation proposals, the engine fills both
    visualization tracks from the paper itself: 3 × 2D canvas + 3 × Three.js.
    """
    sims = generate_html5_simulations(
        paper_text="# Cell Cinema\n\n## Mechanism\n\nVideo-derived trajectories.",
        analysis_result=FakeAnalysis(),
        peer_reviews=[{"summary": "Needs independent validation."}],
        recommended_simulations=[],
    )

    assert len(sims) == 6
    canvas_sims = [s for s in sims if s["type"] == "html5_canvas"]
    three_sims = [s for s in sims if s["type"] == "threejs"]
    assert len(canvas_sims) == 3
    assert len(three_sims) == 3

    assert {s["scene"] for s in canvas_sims} == {
        "trajectory_manifold",
        "counterfactual_perturbation",
        "microenvironment_gradient",
    }
    assert {s["three_scene"] for s in three_sims} == {
        "tumor_volume_3d",
        "mitotic_lattice_3d",
        "perturbation_cone_3d",
    }
    assert canvas_sims[0]["parameters"]["title"] == "Cell Cinema"
    assert three_sims[0]["parameters"]["title"] == "Cell Cinema"


def test_generate_html5_simulations_keeps_peer_review_proposal_first():
    """A reviewer-supplied proposal still claims the first 2D canvas slot;
    Three.js specs are appended after the 2D track."""
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

    assert len(sims) == 6
    # First spec is the canvas track (peer-review proposal slotted here).
    assert sims[0]["type"] == "html5_canvas"
    assert sims[0]["description"] == "Reviewer-requested survival bootstrap."
    # Three.js track lives in the back half of the list.
    assert all(s["type"] == "threejs" for s in sims[3:])


def test_threejs_specs_have_required_renderer_fields():
    """Renderer contract: every threejs spec must carry a known `three_scene`,
    a numeric `seed`, and a parameters dict so the publisher's importmap
    script can boot it deterministically."""
    sims = generate_html5_simulations(
        paper_text="# Paper\n\nbody",
        analysis_result=FakeAnalysis(),
        peer_reviews=[],
    )
    three_sims = [s for s in sims if s["type"] == "threejs"]
    valid_scenes = {"tumor_volume_3d", "mitotic_lattice_3d", "perturbation_cone_3d"}
    for s in three_sims:
        assert s["three_scene"] in valid_scenes
        assert isinstance(s["seed"], int)
        assert isinstance(s["parameters"], dict)
