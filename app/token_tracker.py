"""Token tracking + per-call API log for one CancerHawk run.

Mirrors MOTO's TokenTracker pattern but is per-session (instantiated per
WebSocket connection) so concurrent runs don't bleed counters into each
other. Also keeps a rolling log of every API call so the UI can show
exactly which model was called when, with prompt/completion token counts
and round-trip latency.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

# Approximate OpenRouter pricing in USD per 1M tokens (input, output).
# Used for a rough cumulative cost estimate. Update freely — these numbers
# drift. Unknown models fall back to (0, 0).
PRICING_PER_M = {
    "anthropic/claude-opus-4.7": (15.00, 75.00),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-oss-120b": (0.50, 1.50),
    "openai/gpt-oss-20b": (0.10, 0.30),
    "google/gemini-2.0-flash-001": (0.10, 0.40),
    "google/gemini-2.0-flash-lite-001": (0.075, 0.30),
    "meta-llama/llama-3.1-70b-instruct": (0.40, 0.40),
    "mistralai/mistral-large": (2.00, 6.00),
    "deepseek/deepseek-r1": (0.55, 2.19),
    "deepseek/deepseek-chat": (0.27, 1.10),
    "x-ai/grok-2-1212": (2.00, 10.00),
}


@dataclass
class APICall:
    seq: int
    timestamp: float
    role: str          # which engine role made the call: submitter/validator/...
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    cost_usd: float
    ok: bool
    error: str | None = None
    prompt_messages: list[dict] | None = None  # full messages array
    response_text: str | None = None           # full response text

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "role": self.role,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": round(self.cost_usd, 6),
            "ok": self.ok,
            "error": self.error,
            "prompt": self._format_prompt(),
            "response": self.response_text or "",
        }

    def _format_prompt(self) -> str:
        if self.prompt_messages:
            return "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in self.prompt_messages
            )
        return ""


@dataclass
class TokenTracker:
    started_at: float = field(default_factory=time.time)
    total_input: int = 0
    total_output: int = 0
    total_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: int = 0
    total_cost_usd: float = 0.0
    by_model: dict[str, dict] = field(default_factory=lambda: defaultdict(lambda: {"input": 0, "output": 0, "calls": 0, "latency_ms": 0, "cost_usd": 0.0}))
    by_role: dict[str, dict] = field(default_factory=lambda: defaultdict(lambda: {"input": 0, "output": 0, "calls": 0, "latency_ms": 0, "cost_usd": 0.0}))
    calls: list[APICall] = field(default_factory=list)

    def record(
        self,
        role: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        ok: bool = True,
        error: str | None = None,
        prompt_messages: list[dict] | None = None,
        response_text: str | None = None,
    ) -> APICall:
        cost = _estimate_cost(model, prompt_tokens, completion_tokens)
        self.total_calls += 1
        if not ok:
            self.failed_calls += 1
        self.total_input += prompt_tokens
        self.total_output += completion_tokens
        self.total_latency_ms += latency_ms
        self.total_cost_usd += cost

        m = self.by_model[model]
        m["input"] += prompt_tokens
        m["output"] += completion_tokens
        m["calls"] += 1
        m["latency_ms"] += latency_ms
        m["cost_usd"] = round(m["cost_usd"] + cost, 6)

        r = self.by_role[role]
        r["input"] += prompt_tokens
        r["output"] += completion_tokens
        r["calls"] += 1
        r["latency_ms"] += latency_ms
        r["cost_usd"] = round(r["cost_usd"] + cost, 6)

        call = APICall(
            seq=self.total_calls,
            timestamp=time.time(),
            role=role,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            ok=ok,
            error=error,
            prompt_messages=prompt_messages,
            response_text=response_text,
        )
        self.calls.append(call)
        return call

    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at

    def stats(self) -> dict:
        avg_latency = round(self.total_latency_ms / self.total_calls) if self.total_calls else 0
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "total_input": self.total_input,
            "total_output": self.total_output,
            "total_tokens": self.total_input + self.total_output,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": avg_latency,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "elapsed_seconds": round(self.elapsed_seconds(), 1),
            "by_model": {k: dict(v) for k, v in self.by_model.items()},
            "by_role": {k: dict(v) for k, v in self.by_role.items()},
        }


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_rate, out_rate = PRICING_PER_M.get(model, (0.0, 0.0))
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0
