"""Unit tests for app.openrouter."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from app.openrouter import chat, chat_json, OpenRouterError, _extract_json


@pytest.mark.asyncio
async def test_chat_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello world"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }

    with patch("app.openrouter._get_client") as get_client:
        client = AsyncMock()
        client.post.return_value = mock_response
        get_client.return_value = client

        result = await chat(
            api_key="sk-test",
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
        )

        assert result == "Hello world"
        # Verify tracker not called (not passed)
        # verify on_call not called


@pytest.mark.asyncio
async def test_chat_http_error():
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = "Rate limited"

    with patch("app.openrouter._get_client") as get_client:
        client = AsyncMock()
        client.post.return_value = mock_response
        get_client.return_value = client

        with pytest.raises(OpenRouterError) as exc:
            await chat("sk-test", "m", [{"role": "user", "content": "x"}])
        assert "429" in str(exc.value)


@pytest.mark.asyncio
async def test_chat_network_error():
    with patch("app.openrouter._get_client") as get_client:
        client = AsyncMock()
        client.post.side_effect = httpx.ConnectError("Network down")
        get_client.return_value = client

        with pytest.raises(OpenRouterError):
            await chat("sk-test", "m", [{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_chat_records_to_tracker():
    from app.token_tracker import TokenTracker

    tracker = TokenTracker()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Response"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
    }

    with patch("app.openrouter._get_client") as get_client:
        client = AsyncMock()
        client.post.return_value = mock_response
        get_client.return_value = client

        result = await chat(
            api_key="sk-test",
            model="m",
            messages=[],
            tracker=tracker,
            role="test_role",
        )

    assert tracker.total_calls == 1
    assert tracker.total_input == 5
    assert tracker.total_output == 10
    stats = tracker.stats()
    assert stats["by_role"]["test_role"]["calls"] == 1
    assert stats["by_model"]["m"]["calls"] == 1


@pytest.mark.asyncio
async def test_chat_calls_on_call_hook():
    from app.token_tracker import TokenTracker

    tracker = TokenTracker()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }

    on_call = AsyncMock()

    with patch("app.openrouter._get_client") as get_client:
        client = AsyncMock()
        client.post.return_value = mock_response
        get_client.return_value = client
        await chat("k", "m", [], tracker=tracker, on_call=on_call)

    on_call.assert_awaited_once()
    call_arg = on_call.await_args[0][0]
    assert call_arg.seq == 1
    assert call_arg.model == "m"


def test_extract_json_simple():
    obj = {"a": 1, "b": "text"}
    s = json.dumps(obj)
    assert _extract_json(s) == obj


def test_extract_json_with_code_fences():
    text = "```json\n{\"a\": 1}\n```"
    assert _extract_json(text) == {"a": 1}


def test_extract_json_nested():
    text = 'Some preamble {"inner": {"x": 5}} suffix'
    result = _extract_json(text)
    assert result == {"inner": {"x": 5}}


def test_extract_json_truncated_with_braces():
    """Truncated JSON missing closing braces should be repaired."""
    # Simple object with one unclosed brace
    text = '{"title": "Test", "sections": [{"heading": "Intro"'
    result = _extract_json(text)
    assert result["title"] == "Test"
    assert isinstance(result["sections"], list)
    assert result["sections"][0]["heading"] == "Intro"


def test_extract_json_truncated_nested():
    """Nested object + array with multiple unclosed containers."""
    text = '{"a": [1, 2, {"b": {"c": 3'
    result = _extract_json(text)
    assert result["a"][2]["b"]["c"] == 3


def test_extract_json_code_fence_and_truncation():
    text = "```json\n{\"a\": 1, \"b\": [2, 3"
    result = _extract_json(text)
    assert result["a"] == 1
    assert result["b"] == [2, 3]


@pytest.mark.asyncio
async def test_chat_json_retry_on_non_json_response():
    """If first response is not valid JSON, chat_json retries without response_format."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    # First call returns non-JSON starting content, second call returns valid JSON
    call_count = 0

    async def post_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Return a response with content that doesn't parse as JSON
            mock = MagicMock()
            mock.json.side_effect = json.JSONDecodeError("boom", "doc", 0)
            mock.status_code = 200
            mock.text = "Not JSON"
            return mock
        else:
            mock = MagicMock()
            mock.status_code = 200
            mock.json.return_value = {
                "choices": [{"message": {"content": "{\"parsed\":true}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
            return mock

    with patch("app.openrouter._get_client") as get_client:
        client = AsyncMock()
        client.post.side_effect = post_side_effect
        get_client.return_value = client

        result = await chat_json("key", "m", [{"role": "user", "content": "x"}])

    assert result == {"parsed": True}
    assert call_count == 2
