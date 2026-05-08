"""Pay.sh integration — outbound calls to pay.sh endpoints and local x402 gateway.

CancerHawk uses two sides of pay.sh:

1. **Outbound** — calls pay.sh endpoints (Perplexity, stableneirch) via the `pay` CLI
   for richer research data. CancerHawk's wallet pays these calls.

2. **Inbound** — a `pay server` gateway sits in front of the agent API so external
   agents can pay CancerHawk for runs. The gateway handles x402 payment verification
   and proxies paid requests to the FastAPI app.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("paysh")

SOLANA_WALLET = os.environ.get("SOLANA_WALLET_ADDRESS", "").strip()
PAY_SH_SANDBOX = os.environ.get("PAY_SH_SANDBOX", "true").strip().lower() in ("true", "1", "yes")

PAYSH_ENDPOINTS: dict[str, dict[str, Any]] = {
    "perplexity_sonar": {
        "url": "https://pplx.x402.paysponge.com/v1/sonar",
        "price_usd": 0.00,
        "method": "POST",
        "description": "Grounded AI answer with citations",
    },
    "perplexity_search": {
        "url": "https://pplx.x402.paysponge.com/v1/search",
        "price_usd": 0.01,
        "method": "POST",
        "description": "Web search with inline citations",
    },
    "stableenrich_exa_search": {
        "url": "https://stableenrich.x402.paysponge.com/v1/exa/search",
        "price_usd": 0.01,
        "method": "POST",
        "description": "Neural web search — paper discovery, clinical trial lookup",
    },
    "stableenrich_exa_answer": {
        "url": "https://stableenrich.x402.paysponge.com/v1/exa/answer",
        "price_usd": 0.01,
        "method": "POST",
        "description": "AI-generated answer with source citations",
    },
}


@dataclass
class PayshResult:
    ok: bool
    endpoint: str
    data: dict[str, Any] | None = None
    error: str | None = None
    cost_usd: float = 0.0
    raw_output: str = ""


def get_pay_command() -> list[str]:
    """Return the `pay` CLI invocation prefix, with sandbox flag if configured."""
    cmd = ["pay"]
    if PAY_SH_SANDBOX:
        cmd.append("--sandbox")
    return cmd


def call_paysh(
    endpoint_key: str,
    body: dict[str, Any],
    timeout: int = 30,
) -> PayshResult:
    """Call a pay.sh endpoint through the `pay` CLI.

    The CLI handles the full x402 challenge-response flow: it sends the
    request, receives a 402 with payment details, pays from the configured
    wallet, and retries with proof.
    """
    if endpoint_key not in PAYSH_ENDPOINTS:
        return PayshResult(ok=False, endpoint=endpoint_key, error=f"Unknown pay.sh endpoint: {endpoint_key}")

    spec = PAYSH_ENDPOINTS[endpoint_key]
    url = spec["url"]
    price = spec["price_usd"]

    cmd = get_pay_command() + ["curl", "-s", "-X", spec["method"], url]
    cmd.extend(["-H", "Content-Type: application/json"])
    cmd.extend(["-d", json.dumps(body)])

    logger.info("paysh_call_start", extra={"endpoint": endpoint_key, "sandbox": PAY_SH_SANDBOX, "price_usd": price})

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - start

        if proc.returncode != 0:
            stderr = proc.stderr.strip()[:500]
            logger.error("paysh_call_failed", extra={"endpoint": endpoint_key, "rc": proc.returncode, "stderr": stderr})
            return PayshResult(ok=False, endpoint=endpoint_key, error=f"pay CLI exited {proc.returncode}: {stderr}")

        raw = proc.stdout.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"raw_text": raw[:2000]}

        logger.info("paysh_call_ok", extra={"endpoint": endpoint_key, "elapsed_s": round(elapsed, 2), "cost_usd": price})
        return PayshResult(ok=True, endpoint=endpoint_key, data=data, cost_usd=price, raw_output=raw[:2000])

    except subprocess.TimeoutExpired:
        logger.error("paysh_call_timeout", extra={"endpoint": endpoint_key})
        return PayshResult(ok=False, endpoint=endpoint_key, error="pay.sh call timed out")
    except FileNotFoundError:
        return PayshResult(ok=False, endpoint=endpoint_key, error="`pay` CLI not installed. Install from https://pay.sh")
    except Exception as exc:
        logger.error("paysh_call_exception", extra={"endpoint": endpoint_key, "error": str(exc)})
        return PayshResult(ok=False, endpoint=endpoint_key, error=str(exc))


def research_enrichment(research_goal: str) -> str:
    """Use pay.sh endpoints to enrich a research goal with web context.

    Returns concatenated enrichment text that can be prepended to prompts.
    Calls are sequential to respect pay.sh wallet flow.
    """
    if PAY_SH_SANDBOX:
        logger.info("research_enrichment_skipped_sandbox", extra={"research_goal": research_goal[:80]})
        return ""

    parts: list[str] = []
    total_cost = 0.0

    # 1. Perplexity Sonar — free grounded answer
    result = call_paysh("perplexity_sonar", {"query": f"Recent scientific research on: {research_goal}. Provide mechanism-level detail with citations."})
    if result.ok and result.data:
        text = _extract_text(result.data, "answer")
        if text:
            parts.append(f"## Perplexity Sonar\n{text[:1500]}")

    # 2. Exa neural search — paper and clinical trial discovery
    result = call_paysh("stableenrich_exa_search", {"query": f"{research_goal} clinical trials OR molecular mechanism OR immunotherapy", "num_results": 5})
    total_cost += result.cost_usd
    if result.ok and result.data:
        text = _extract_text(result.data, "results")
        if text:
            parts.append(f"## Web search\n{text[:1500]}")

    if parts:
        enrichment = "\n\n---\n\n".join(parts)
        cost_str = f"${total_cost:.2f}" if total_cost > 0 else "$0.00"
        logger.info("research_enrichment_done", extra={"cost_usd": total_cost, "chars": len(enrichment)})
        return f"[Pay.sh enrichment — {cost_str}]\n\n{enrichment}"

    return ""


def _extract_text(data: dict[str, Any], key: str) -> str:
    """Extract text content from various pay.sh response shapes."""
    if isinstance(data, dict):
        val = data.get(key)
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            return "\n".join(
                item.get("text") or item.get("content") or item.get("snippet") or str(item)[:500]
                for item in val[:5]
            )
        if isinstance(val, dict):
            return val.get("text") or val.get("content") or json.dumps(val)[:1000]
        return str(val)[:2000] if val is not None else ""
    if isinstance(data, list) and data:
        return _extract_text(data[0], key)
    return ""


def estimate_enrichment_cost() -> dict[str, Any]:
    """Return the estimated cost breakdown for pay.sh enrichment."""
    endpoints_used = ["perplexity_sonar", "stableenrich_exa_search"]
    total = sum(PAYSH_ENDPOINTS[k]["price_usd"] for k in endpoints_used if k in PAYSH_ENDPOINTS)
    return {
        "endpoints": [{"key": k, "price_usd": PAYSH_ENDPOINTS[k]["price_usd"], "desc": PAYSH_ENDPOINTS[k]["description"]} for k in endpoints_used],
        "total_cost_usd": total,
        "pay_sh_sandbox": PAY_SH_SANDBOX,
        "solana_wallet": SOLANA_WALLET or "not set",
    }
