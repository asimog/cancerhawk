"""Tests for the Precigenetic base layer and subagent wallet identities."""

from app.precigenetic_base_layer import build_precigenetic_topics
from app.subagent_wallets import assign_block_race_wallets


def test_precigenetic_base_layer_returns_ten_topics_without_network():
    topics, metadata = build_precigenetic_topics(
        "PD-1 resistance in melanoma",
        fetch_live_gdc=False,
    )

    assert len(topics) == 10
    assert metadata["source"] == "precigenetic_cancerhawk"
    assert metadata["topic_count"] == 10
    assert metadata["gdc_summary"]["reason"] == "disabled"
    assert [topic["id"] for topic in topics] == list(range(1, 11))
    assert topics[0]["source"] == "precigenetic_tcga_core_lane"
    assert all("PD-1 resistance in melanoma" in topic["research_goal"] for topic in topics)


def test_block_race_subagent_wallets_are_deterministic_public_identities():
    wallets = assign_block_race_wallets("job-123", topic_count=10)
    same_wallets = assign_block_race_wallets("job-123", topic_count=10)

    assert wallets == same_wallets
    assert len(wallets) == 21
    assert "moto_worker_10" in wallets
    assert "miroshark_reviewer_10" in wallets
    assert "moto_validator_01" in wallets
    assert wallets["moto_worker_01"]["custody"] == "public-attribution-only"
    assert all("private" not in wallet for wallet in wallets.values())
