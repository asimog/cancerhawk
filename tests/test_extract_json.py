"""Comprehensive coverage for `app.openrouter._extract_json`.

This is a regression suite for the rewrite of the JSON extractor. The
previous implementation referenced uninitialized locals (`escape_next`,
`in_string`, `depth`, `bracket_depth`, `candidate`) and crashed on any
input that didn't take the fast-path. The new implementation parses
truncated LLM output by walking the body with a string/escape state
machine and a container stack, then trims back to the last completed
element and closes any open containers.

Each test below documents what input class it covers; the *_REGRESSION
tests reproduce specific failures observed in production.
"""

from __future__ import annotations

import json

import pytest

from app.openrouter import OpenRouterError, _extract_json


# --- happy-path / fast-path ------------------------------------------------

class TestComplete:
    def test_simple_object(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_nested(self):
        assert _extract_json('{"a": {"b": [1, 2, 3]}, "c": null}') == \
            {"a": {"b": [1, 2, 3]}, "c": None}

    def test_unicode(self):
        assert _extract_json('{"name": "Иван", "emoji": "🧬"}') == \
            {"name": "Иван", "emoji": "🧬"}

    def test_whitespace_around(self):
        assert _extract_json('   \n {"a": 1}\n  ') == {"a": 1}

    def test_escaped_quote_inside_string(self):
        assert _extract_json(r'{"a": "he said \"hi\""}') == {"a": 'he said "hi"'}

    def test_escaped_backslash_inside_string(self):
        # Raw text on the wire is: {"path": "C:\\\\users"} → JSON value C:\\users
        assert _extract_json(r'{"path": "C:\\users"}') == {"path": r"C:\users"}


# --- markdown fences -------------------------------------------------------

class TestFences:
    def test_json_fence(self):
        raw = "```json\n{\"a\": 1}\n```"
        assert _extract_json(raw) == {"a": 1}

    def test_bare_fence(self):
        raw = "```\n{\"a\": 1}\n```"
        assert _extract_json(raw) == {"a": 1}

    def test_fence_with_trailing_text_inside(self):
        # Fenced block with newline before the closing fence.
        raw = "```\n{\"a\": 1}\n```\n"
        assert _extract_json(raw) == {"a": 1}


# --- prefix / suffix garbage -----------------------------------------------

class TestGarbage:
    def test_explanatory_prefix(self):
        assert _extract_json('Here is the JSON: {"a": 1}') == {"a": 1}

    def test_explanatory_suffix(self):
        assert _extract_json('{"a": 1}\n\nHope that helps!') == {"a": 1}

    def test_both(self):
        assert _extract_json('Sure! {"a": 1} done') == {"a": 1}

    def test_no_object_raises(self):
        with pytest.raises(OpenRouterError, match="no JSON object found"):
            _extract_json("just plain text without braces")

    def test_empty_string_raises(self):
        with pytest.raises(OpenRouterError):
            _extract_json("")


# --- truncation repair (max_tokens cut-off) --------------------------------

class TestTruncationRepair:
    def test_user_reported_failure_REGRESSION(self):
        """The exact input that crashed the previous extractor:
        outline JSON cut off mid-string in the second section's `summary`.
        Repair should keep the title and the first complete section.
        """
        raw = (
            '{ "title": "Cell Cinema: Latent-Space Trajectory Inversion in '
            'Video Diffusion Models as a Causal Discovery Engine for Oncology", '
            '"sections": [ { "heading": "1. Introduction", "summ'
        )
        out = _extract_json(raw)
        assert out["title"].startswith("Cell Cinema")
        assert isinstance(out["sections"], list)
        # The partial second section (only "summ...") must be dropped, but
        # if the first section was complete it should survive. In this
        # input the first section's `summary` was never closed, so the
        # repaired output will trim the whole `sections` list back to
        # `[]` — that's the safe behavior.
        assert all(isinstance(s, dict) for s in out["sections"])

    def test_truncated_mid_string_inside_object(self):
        # `{"a": 1, "b": "trunc` → keep the first complete pair.
        out = _extract_json('{"a": 1, "b": "trunc')
        assert out == {"a": 1}

    def test_truncated_mid_array_keeps_complete_elements(self):
        out = _extract_json('{"x": [1, 2, 3, 4')
        assert out == {"x": [1, 2, 3]}

    def test_truncated_mid_nested_object(self):
        out = _extract_json('{"x": 1, "y": {"a": "hello')
        # Inner object's only key is incomplete (string never closed) →
        # inner becomes empty.
        assert out == {"x": 1, "y": {}}

    def test_truncated_after_escape_does_not_crash(self):
        # Body ends mid-escape sequence. Should not raise.
        out = _extract_json(r'{"a": "foo\\')
        assert isinstance(out, dict)

    def test_truncated_with_complete_first_then_partial_second(self):
        out = _extract_json(
            '{"sections":[{"heading":"a","summary":"hi"},{"heading":"b","summa'
        )
        assert out["sections"][0] == {"heading": "a", "summary": "hi"}
        # Second is partial — at minimum its key was complete.
        assert out["sections"][1].get("heading") == "b"

    def test_truncated_just_after_opening_brace(self):
        out = _extract_json("{")
        assert out == {}

    def test_truncated_with_dangling_comma(self):
        out = _extract_json('{"a": 1,')
        assert out == {"a": 1}

    def test_truncated_with_dangling_colon(self):
        out = _extract_json('{"a": 1, "b":')
        assert out == {"a": 1}


# --- cannot-repair cases ---------------------------------------------------

class TestUnrepairable:
    def test_garbage_inside_braces_raises(self):
        with pytest.raises(OpenRouterError):
            _extract_json('{this is not json}')


# --- general invariants ----------------------------------------------------

class TestInvariants:
    @pytest.mark.parametrize("payload", [
        {"a": 1},
        {"nested": {"deep": {"x": [1, 2, {"y": True}]}}},
        {"strs": ["a", "b\"c", "d\\e"]},
        {"empty_list": [], "empty_obj": {}, "null": None},
    ])
    def test_roundtrip_complete_payloads(self, payload):
        raw = json.dumps(payload)
        assert _extract_json(raw) == payload

    @pytest.mark.parametrize("prefix", ["", "Here: ", "```json\n", "```\n"])
    @pytest.mark.parametrize("suffix", ["", "\n", "\n```", " (done)"])
    def test_roundtrip_with_wrappers(self, prefix, suffix):
        payload = {"a": 1, "b": [2, 3]}
        raw = prefix + json.dumps(payload) + suffix
        # If suffix re-introduces a fence on a payload that wasn't fenced
        # to begin with, the parser still finds the first { and walks to
        # the matching close — that's the contract.
        assert _extract_json(raw) == payload

    def test_does_not_consume_past_first_balanced_object(self):
        # Two objects on the wire — extractor returns only the first.
        out = _extract_json('{"a": 1} {"b": 2}')
        assert out == {"a": 1}
