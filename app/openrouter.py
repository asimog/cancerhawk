"""Async OpenRouter chat-completions client.

API key is passed per-call (sourced from the browser session, never
persisted server-side). Per-call usage is recorded into a TokenTracker
when one is supplied via ``ctx``.
"""

from __future__ import annotations

import json
import time
from itertools import permutations
from typing import Any, Awaitable, Callable

import httpx

from .token_tracker import APICall, TokenTracker

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
APP_REFERER = "http://localhost:8765"
APP_TITLE = "CancerHawk"

# Optional async hook the engines can install to push every API call to
# the WebSocket as it happens. Signature: async (call: APICall) -> None.
CallEmitFn = Callable[[APICall], Awaitable[None]]


class OpenRouterError(RuntimeError):
    pass


_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0))
    return _client


async def chat(
    api_key: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    response_format: dict | None = None,
    role: str = "unknown",
    tracker: TokenTracker | None = None,
    on_call: CallEmitFn | None = None,
) -> str:
    if not api_key:
        raise OpenRouterError("OpenRouter API key missing")
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if response_format:
        payload["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": APP_REFERER,
        "X-Title": APP_TITLE,
    }

    client = _get_client()
    started = time.perf_counter()
    err: Exception | None = None
    response_data: dict | None = None
    text: str | None = None

    try:
        r = await client.post(OPENROUTER_URL, headers=headers, json=payload)
        if r.status_code >= 400:
            raise OpenRouterError(f"HTTP {r.status_code}: {r.text[:500]}")
        response_data = r.json()
        text = response_data["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, OpenRouterError) as exc:
        err = exc
    except Exception as exc:
        err = exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    usage = (response_data or {}).get("usage") or {}
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)

    call: APICall | None = None
    if tracker is not None:
        call = tracker.record(
            role=role,
            model=model,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=latency_ms,
            ok=err is None,
            error=str(err) if err is not None else None,
            prompt_messages=messages,
            response_text=text,
        )
        if on_call is not None:
            try:
                await on_call(call)
            except Exception:
                pass

    if err is not None:
        if isinstance(err, OpenRouterError):
            raise err
        raise OpenRouterError(f"{type(err).__name__}: {err}") from err

    return text or ""


async def chat_json(
    api_key: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.4,
    max_tokens: int | None = None,
    role: str = "unknown",
    tracker: TokenTracker | None = None,
    on_call: CallEmitFn | None = None,
) -> dict:
    try:
        text = await chat(
            api_key,
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            role=role,
            tracker=tracker,
            on_call=on_call,
        )
    except OpenRouterError:
        text = await chat(
            api_key,
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            role=role,
            tracker=tracker,
            on_call=on_call,
        )
    return _extract_json(text)


def _extract_json(text: str) -> dict:
    """Extract a JSON object from an LLM response.

    Handles:
      - Markdown code fences (```json ... ```)
      - Leading/trailing explanatory text
      - Truncated JSON (missing closing braces/brackets) by trying
        all permutations of needed closing characters.
    """
    text = text.strip()

    # Strip code fences
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Fast path: whole text valid JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first {
    start = text.find("{")
    if start == -1:
        raise OpenRouterError(f"no JSON object found in response: {text[:200]!r}")

    # Walk to compute unclosed { and [
    # Walk to compute depth and optionally find a balanced candidate
    candidate = None
    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not in_string:
            in_string = True
        elif ch == '"' and in_string:
            in_string = False
        elif not in_string:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    break  # found complete object span
            elif ch == '[':
                bracket_depth += 1
            elif ch == ']':
                bracket_depth -= 1

    if candidate is None:
        candidate = text[start:]

    # If we have unclosed containers, generate closing permutations
    if depth < 0:
        depth = 0
    if bracket_depth < 0:
        bracket_depth = 0

    if depth == 0 and bracket_depth == 0:
        # No unclosed; but earlier fast parse failed, so candidate itself might be malformed
        # We'll still try a few repairs just in case minor truncation occurred within structures
        closing_variants = {''}
    else:
        needed = '}' * depth + ']' * bracket_depth
        # Generate unique permutations (multiset). If N is large, limit attempts.
        closing_variants = set([''.join(p) for p in permutations(needed)])
        closing_variants.add('')  # also try as-is

    # Try parsing candidate with each repair suffix
    for repair in closing_variants:
        try:
            return json.loads(candidate + repair)
        except json.JSONDecodeError:
            continue

    raise OpenRouterError(
        f"unbalanced JSON (depth={depth}, brackets={bracket_depth}): {text[:200]!r}"
    )


async def close() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
