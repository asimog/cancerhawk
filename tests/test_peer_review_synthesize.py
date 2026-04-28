"""Test that _synthesize correctly calculates acceptance probability."""

from app.peer_review_engine import _synthesize, PeerReview


def test_synthesize_all_reject():
    reviews = [
        PeerReview(
            archetype_id=f"a{i}",
            archetype_name=f"R{i}",
            recommendation="reject",
            overall_confidence=0.9,
            summary="",
            dimension_scores={},
            criticisms=[],
            required_fixes=[],
            suggested_experiments=[],
            simulation_proposal=None,
        )
        for i in range(3)
    ]
    cons = _synthesize(reviews)
    assert cons.acceptance_probability == 0.0


def test_synthesize_all_accept():
    reviews = [
        PeerReview(
            archetype_id=f"a{i}",
            archetype_name=f"R{i}",
            recommendation="accept",
            overall_confidence=1.0,
            summary="",
            dimension_scores={},
            criticisms=[],
            required_fixes=[],
            suggested_experiments=[],
            simulation_proposal=None,
        )
        for i in range(3)
    ]
    cons = _synthesize(reviews)
    assert cons.acceptance_probability == 1.0


def test_synthesize_mixed_recommendations():
    # 1 accept (conf=1), 1 minor (conf=1), 1 major (conf=1), 1 reject (conf=1)
    reviews = [
        PeerReview("a1","R1","accept",1.0,{},"",[],[],[],None),
        PeerReview("a2","R2","minor_revision",1.0,{},"",[],[],[],None),
        PeerReview("a3","R3","major_revision",1.0,{},"",[],[],[],None),
        PeerReview("a4","R4","reject",1.0,{},"",[],[],[],None),
    ]
    cons = _synthesize(reviews)
    # weighted = 1 + 0.7 + 0.3 + 0 = 2.0; total_weight = 4; acceptance = 0.5
    assert abs(cons.acceptance_probability - 0.5) < 1e-6
