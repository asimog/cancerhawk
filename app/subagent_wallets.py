"""Deterministic subagent wallet identities for CancerHawk runs.

These are public, per-run ledger identities used to attribute work and points
inside a block race. They intentionally do not create or store private keys.
"""

from __future__ import annotations

import hashlib
from typing import Any

BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _base58_encode(data: bytes) -> str:
    value = int.from_bytes(data, "big")
    chars: list[str] = []
    while value:
        value, remainder = divmod(value, 58)
        chars.append(BASE58_ALPHABET[remainder])
    encoded = "".join(reversed(chars)) or "1"
    leading_zeroes = len(data) - len(data.lstrip(b"\0"))
    return "1" * leading_zeroes + encoded


def subagent_wallet(run_id: str, agent_id: str, role: str) -> dict[str, Any]:
    """Return a stable Solana-shaped public wallet for a run-scoped subagent."""
    seed = f"cancerhawk:{run_id}:{role}:{agent_id}".encode("utf-8")
    digest = hashlib.sha256(seed).digest()
    return {
        "agent_id": agent_id,
        "role": role,
        "wallet_address": _base58_encode(digest),
        "wallet_type": "subagent-ledger",
        "custody": "public-attribution-only",
    }


def assign_block_race_wallets(run_id: str, topic_count: int = 10) -> dict[str, dict[str, Any]]:
    """Create wallets for MOTO workers, MiroShark reviewers, and the validator."""
    wallets: dict[str, dict[str, Any]] = {}
    for index in range(1, topic_count + 1):
        worker_id = f"moto_worker_{index:02d}"
        reviewer_id = f"miroshark_reviewer_{index:02d}"
        wallets[worker_id] = subagent_wallet(run_id, worker_id, "moto_worker")
        wallets[reviewer_id] = subagent_wallet(run_id, reviewer_id, "miroshark_reviewer")
    wallets["moto_validator_01"] = subagent_wallet(run_id, "moto_validator_01", "moto_validator")
    return wallets
