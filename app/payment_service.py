"""Pay winners via Bankr.bot — cross-chain SOL, USDC (Solana), USDC (Base)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

import httpx

logger = logging.getLogger("cancerhawk.payment")

BANKR_BASE_URL = "https://api.bankr.bot"
BANKR_API_KEY = os.environ.get("BANKR_API_KEY", "").strip()
USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

SOLANA_BASE58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")
ETH_HEX_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


def _detect_chain(wallet: str) -> str:
    if ETH_HEX_RE.match(wallet):
        return "base"
    if SOLANA_BASE58_RE.match(wallet):
        return "solana"
    return "unknown"


async def _wallet_transfer_base(
    client: httpx.AsyncClient,
    recipient: str,
    amount_usdc: float,
) -> dict[str, Any]:
    amount_str = f"{amount_usdc:.6f}".rstrip("0").rstrip(".")
    payload = {
        "tokenAddress": USDC_BASE,
        "recipientAddress": recipient,
        "amount": amount_str,
        "isNativeToken": False,
    }
    resp = await client.post(
        f"{BANKR_BASE_URL}/wallet/transfer",
        json=payload,
        timeout=30,
    )
    data = resp.json()
    if resp.status_code != 200 or not data.get("success"):
        raise RuntimeError(f"Bankr Base transfer failed [{resp.status_code}]: {data}")
    return data


async def _agent_transfer_solana(
    client: httpx.AsyncClient,
    recipient: str,
    amount_usdc: float,
) -> dict[str, Any]:
    prompt = f"send {amount_usdc} USDC on solana to {recipient}"
    resp = await client.post(
        f"{BANKR_BASE_URL}/agent/prompt",
        json={"prompt": prompt},
        timeout=30,
    )
    data = resp.json()
    if resp.status_code not in (200, 202):
        raise RuntimeError(f"Bankr agent prompt failed [{resp.status_code}]: {data}")
    job_id = data.get("jobId")
    if not job_id:
        raise RuntimeError(f"Bankr agent prompt missing jobId: {data}")
    return await _poll_agent_job(client, job_id)


async def _poll_agent_job(
    client: httpx.AsyncClient,
    job_id: str,
    max_wait: float = 120,
    interval: float = 3,
) -> dict[str, Any]:
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        resp = await client.get(
            f"{BANKR_BASE_URL}/agent/job/{job_id}",
            timeout=15,
        )
        data = resp.json()
        status = data.get("status", "")
        if status == "completed":
            return data
        if status == "failed":
            raise RuntimeError(f"Bankr agent job failed: {data.get('error', data)}")
        await asyncio.sleep(interval)
    raise TimeoutError(f"Bankr agent job {job_id} did not complete in {max_wait}s")


async def pay_winner(
    wallet: str,
    amount_usdc: float = 0.01,
) -> dict[str, Any]:
    if not BANKR_API_KEY:
        raise RuntimeError("BANKR_API_KEY not configured")

    chain = _detect_chain(wallet)
    if chain == "unknown":
        raise ValueError(f"Cannot determine chain for wallet: {wallet[:12]}...")

    headers = {"X-API-Key": BANKR_API_KEY, "Content-Type": "application/json"}
    async with httpx.AsyncClient(headers=headers) as client:
        if chain == "base":
            result = await _wallet_transfer_base(client, wallet, amount_usdc)
        else:
            result = await _agent_transfer_solana(client, wallet, amount_usdc)

    logger.info(
        "prize_paid",
        extra={
            "wallet": wallet[:12] + "...",
            "chain": chain,
            "amount_usdc": amount_usdc,
            "tx_hash": result.get("txHash", ""),
            "job_id": result.get("jobId", ""),
        },
    )
    return result


def payment_enabled() -> bool:
    configured = os.environ.get("BANKR_PAYMENT_ENABLED", "").strip().lower()
    if configured:
        return configured in {"1", "true", "yes", "on"}
    return False
