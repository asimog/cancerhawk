"""Async OpenRouter chat-completions client.

API key is passed per-call (sourced from the browser session, never
persisted server-side). Per-call usage is recorded into a TokenTracker
when one is supplied via ``ctx``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Awaitable, Callable

import httpx

from .token_tracker import APICall, APIFailureLimitExceeded, MAX_FAILED_API_CALLS, TokenTracker

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
APP_REFERER = "http://localhost:8765"
APP_TITLE = "CancerHawk"
TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
TRANSIENT_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
)

# Optional async hook the engines can install to push every API call to
# the WebSocket as it happens. Signature: async (call: APICall) -> None.
CallEmitFn = Callable[[APICall], Awaitable[None]]


class OpenRouterError(RuntimeError):
    pass


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, str(default))))
    except ValueError:
        return default


OPENROUTER_MAX_RETRIES = _env_int("CANCERHAWK_OPENROUTER_MAX_RETRIES", 8)
OPENROUTER_RETRY_BASE_SECONDS = _env_float("CANCERHAWK_OPENROUTER_RETRY_BASE_SECONDS", 2.0)
OPENROUTER_RETRY_MAX_SECONDS = _env_float("CANCERHAWK_OPENROUTER_RETRY_MAX_SECONDS", 60.0)


_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0))
    return _client


def _retry_after_seconds(response: httpx.Response | None) -> float | None:
    if response is None:
        return None
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _retry_delay(attempt: int, response: httpx.Response | None = None) -> float:
    header_delay = _retry_after_seconds(response)
    if header_delay is not None:
        return min(header_delay, OPENROUTER_RETRY_MAX_SECONDS)
    delay = OPENROUTER_RETRY_BASE_SECONDS * (2 ** max(0, attempt - 1))
    return min(delay, OPENROUTER_RETRY_MAX_SECONDS)


def _should_retry(err: Exception | None, status_code: int | None) -> bool:
    if status_code in TRANSIENT_STATUS_CODES:
        return True
    if isinstance(err, TRANSIENT_EXCEPTIONS):
        return True
    if isinstance(err, (KeyError, IndexError, json.JSONDecodeError)):
        return True
    return False


async def _record_call(
    *,
    tracker: TokenTracker | None,
    on_call: CallEmitFn | None,
    role: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    ok: bool,
    error: str | None,
    messages: list[dict],
    response_text: str | None,
) -> None:
    if tracker is None:
        return
    call = tracker.record(
        role=role,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        ok=ok,
        error=error,
        prompt_messages=messages,
        response_text=response_text,
    )
    if on_call is not None:
        try:
            await on_call(call)
        except Exception:
            pass


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
    last_err: Exception | None = None

    for attempt in range(OPENROUTER_MAX_RETRIES + 1):
        if tracker is not None and MAX_FAILED_API_CALLS and tracker.failed_calls >= MAX_FAILED_API_CALLS:
            raise APIFailureLimitExceeded(tracker.failed_calls, MAX_FAILED_API_CALLS)

        started = time.perf_counter()
        err: Exception | None = None
        response_data: dict | None = None
        response: httpx.Response | None = None
        status_code: int | None = None
        text: str | None = None

        try:
            response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
            status_code = response.status_code
            if response.status_code >= 400:
                raise OpenRouterError(f"HTTP {response.status_code}: {response.text[:500]}")
            response_data = response.json()
            text = response_data["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError, OpenRouterError) as exc:
            err = exc
        except Exception as exc:
            err = exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = (response_data or {}).get("usage") or {}
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)

        await _record_call(
            tracker=tracker,
            on_call=on_call,
            role=role,
            model=model,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=latency_ms,
            ok=err is None,
            error=str(err) if err is not None else None,
            messages=messages,
            response_text=text,
        )

        if err is None:
            return text or ""

        last_err = err
        should_retry = attempt < OPENROUTER_MAX_RETRIES and _should_retry(err, status_code)
        if not should_retry:
            break

        await asyncio.sleep(_retry_delay(attempt + 1, response))

    if isinstance(last_err, OpenRouterError):
        raise last_err
    raise OpenRouterError(f"{type(last_err).__name__}: {last_err}") from last_err


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
    first_error: OpenRouterError | None = None

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
        try:
            return _extract_json(text)
        except OpenRouterError as exc:
            first_error = exc
    except OpenRouterError as exc:
        first_error = exc

    retry_messages = [
        *messages,
        {
            "role": "user",
            "content": (
                "Your previous answer could not be parsed as a JSON object. "
                "Return only one valid JSON object. Do not include markdown, "
                "commentary, arrays, or code fences."
            ),
        },
    ]
    try:
        retry_text = await chat(
            api_key,
            model,
            retry_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            role=role,
            tracker=tracker,
            on_call=on_call,
        )
        return _extract_json(retry_text)
    except OpenRouterError as exc:
        detail = f"retry failed: {exc}"
        if first_error is not None:
            detail = f"initial JSON response failed: {first_error}; {detail}"
        raise OpenRouterError(detail) from exc


def _parse_and_unwrap(json_str: str) -> dict:
    """Parse JSON string and unwrap arrays containing a single dict."""
    parsed = json.loads(json_str)
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        return parsed[0]
    if isinstance(parsed, dict):
        return parsed
    raise OpenRouterError(f"expected JSON object, got {type(parsed).__name__}: {json_str[:200]!r}")


def _extract_json(text: str) -> dict:
    """Extract a JSON object from an LLM response.

    Handles:
      - Markdown code fences (```json ... ```)
      - Leading/trailing explanatory text
      - Truncated JSON (response cut off by max_tokens) — repaired by
        closing the open string (if any), then trimming back to the last
        complete element and closing all unclosed containers.
      - Arrays wrapping a single object: [{...}] is unwrapped to {...}.
    """
    text = text.strip()

    # Strip code fences.
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Fast path: whole text valid JSON.
    try:
        return _parse_and_unwrap(text)
    except json.JSONDecodeError:
        pass
    except OpenRouterError:
        raise

    start = text.find("{")
    if start == -1:
        raise OpenRouterError(f"no JSON object found in response: {text[:200]!r}")

    body = text[start:]

    # Walk the body tracking string/escape state and the container stack.
    # Record the index of the last complete element boundary (a `,` or
    # an opening `{`/`[` at the current top level) so we can trim back to
    # it if the response was truncated mid-value.
    in_string = False
    escape_next = False
    stack: list[str] = []
    last_safe_trim: int | None = None
    balanced_end: int | None = None

    for i, ch in enumerate(body):
        if escape_next:
            escape_next = False
            continue
        if in_string:
            if ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append(ch)
            # After an opener, a safe trim is just before this point — i.e.
            # the empty container "{}" / "[]" is always a valid fallback.
            last_safe_trim = i + 1
            continue
        if ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                balanced_end = i + 1
                break
            continue
        if ch == "," and stack:
            # End of a complete element at the current container level.
            last_safe_trim = i

    if balanced_end is not None:
        candidate = body[:balanced_end]
        try:
            return _parse_and_unwrap(candidate)
        except json.JSONDecodeError:
            pass

    # Truncated. Build a repaired candidate.
    repaired = body
    if in_string:
        repaired += '"'  # close the open string
    # Trim trailing whitespace and dangling separators that follow the
    # last completed element.
    if last_safe_trim is not None and last_safe_trim < len(repaired):
        # Trim back to last complete element; this drops the partial
        # (truncated) element entirely.
        repaired = repaired[:last_safe_trim].rstrip().rstrip(",:")
    else:
        repaired = repaired.rstrip().rstrip(",:=")

    # Close remaining open containers in reverse order.
    closers = {"{": "}", "[": "]"}
    repaired += "".join(closers[c] for c in reversed(stack))

    try:
        return _parse_and_unwrap(repaired)
    except json.JSONDecodeError as exc:
        raise OpenRouterError(
            f"could not repair truncated JSON ({exc}): {text[:200]!r}"
        ) from exc


async def close() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
