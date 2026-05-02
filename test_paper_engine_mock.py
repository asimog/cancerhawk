"""End-to-end smoke test for paper_engine with mocked LLM calls."""

import asyncio
import json
import time
import app.paper_engine as pe
from app.token_tracker import TokenTracker
from app.openrouter import chat as real_chat, chat_json as real_chat_json

# We'll monkeypatch chat and chat_json in the paper_engine module
# to avoid real API calls.

call_log = []

async def mock_chat(api_key, model, messages, *, temperature=0.7, role="unknown", tracker=None, on_call=None, max_tokens=None, response_format=None):
    """Mock chat that simulates LLM responses for various roles."""
    call_log.append(('chat', role, model))
    # Simulate token tracking if tracker provided
    if tracker:
        tracker.record(
            role=role,
            model=model,
            prompt_tokens=200,
            completion_tokens=100,
            latency_ms=50,
            ok=True,
            prompt_messages=messages,
            response_text="(mocked response)",
        )
    if role == "submitter":
        return json.dumps({"submission": f"Novel research direction #{len(call_log)} regarding the topic."})
    elif role == "compiler_section":
        # Return a plausible section body
        idx = len([c for c in call_log if c[1] == "compiler_section"])
        return f"This is the content of section {idx}. It contains detailed, rigorous exposition relevant to the research goal."
    elif role == "compiler_abstract":
        return "This abstract summarizes the key findings and implications of the paper in 150-250 words."
    else:
        return "{}"

async def mock_chat_json(api_key, model, messages, *, temperature=0.4, role="unknown", tracker=None, on_call=None, max_tokens=None):
    """Mock chat_json for structured outputs."""
    call_log.append(('chat_json', role, model))
    if tracker:
        tracker.record(
            role=role,
            model=model,
            prompt_tokens=300,
            completion_tokens=150,
            latency_ms=80,
            ok=True,
            prompt_messages=messages,
            response_text=json.dumps({}),
        )
    if role == "validator":
        # Accept every submission, with novelty score for convergence tracking
        return {
            "decision": "accept",
            "reasoning": "Mock acceptance: adds novel insight.",
            "summary": "",
            "scores": {"novelty": 9}
        }
    elif role == "compiler_outline":
        return {
            "title": "Mock Generated Paper Title",
            "sections": [
                {"heading": "Introduction", "summary": "Introduce the problem and objectives."},
                {"heading": "Background", "summary": "Establish necessary background and definitions."},
                {"heading": "Main Results", "summary": "Present the core theorems and findings."},
                {"heading": "Discussion", "summary": "Discuss implications and connections."},
                {"heading": "Conclusion", "summary": "Summarize and suggest future directions."},
            ]
        }
    else:
        return {}

async def mock_emit(event, message, data=None):
    # Log events quietly to avoid Unicode printing issues on Windows console
    mock_emit.log.append((event, message, data))
mock_emit.log = []

class SimpleTracker(TokenTracker):
    def __init__(self):
        super().__init__()
        self.calls = []  # override to hold APICall objects
    def record(self, *, role, model, prompt_tokens, completion_tokens, latency_ms, ok=True, error=None, prompt_messages=None, response_text=None):
        from app.token_tracker import APICall
        # Very rough cost calculation (0 for test models)
        cost_usd = 0.0
        call = APICall(
            seq=len(self.calls)+1,
            timestamp=time.time(),
            role=role,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            ok=ok,
            error=error,
            prompt_messages=prompt_messages,
            response_text=response_text,
        )
        self.calls.append(call)
        self.total_input += prompt_tokens
        self.total_output += completion_tokens
        self.total_calls += 1
        self.total_latency_ms += latency_ms
        if ok:
            self.total_cost_usd += cost_usd

    @staticmethod
    def _cost_for(model, pt, ct):
        # Simplified cost lookup (use same as PRICING_PER_M)
        pricing = {
            "test/submitter": (0, 0),
            "test/validator": (0, 0),
            "test/compiler": (0, 0),
        }
        ip, op = pricing.get(model, (0,0))
        return (ip * pt + op * ct) / 1e6

async def main():
    # Patch paper_engine's chat and chat_json
    pe.chat = mock_chat
    pe.chat_json = mock_chat_json

    tracker = SimpleTracker()
    models = {
        "submitter": "test/submitter",
        "validator": "test/validator",
        "compiler": "test/compiler",
    }

    paper = await pe.run_paper_engine(
        api_key="fake-key",
        research_goal="Investigate the role of mitochondrial metabolism in cancer progression.",
        models=models,
        n_submitters=1,  # keep it simple
        emit=mock_emit,
        tracker=tracker,
        on_call=lambda c: None,
        previous_block_context="",
    )

    print("\n=== PAPER GENERATED ===")
    print(f"Title: {paper.title}")
    print(f"Sections ({len(paper.sections)}):")
    for s in paper.sections:
        print(f"  - {s['heading']} ({len(s['content'])} chars)")
    print(f"Accepted submissions: {len(paper.accepted_submissions)}")
    print(f"Rounds run: {paper.rounds_run}")
    print(f"Convergence reason: {paper.convergence_reason}")
    print("\nFull text preview:\n")
    print(paper.full_text()[:1000])

    # Basic sanity assertions
    assert paper.title, "Title should not be empty"
    assert len(paper.sections) >= 2, "Should have at least abstract and introduction"
    assert any(s['heading'].lower() == 'abstract' for s in paper.sections), "Should include abstract"
    assert any(s['heading'].lower() == 'introduction' for s in paper.sections), "Should include introduction"
    assert any(s['heading'].lower() == 'conclusion' for s in paper.sections), "Should include conclusion"
    print("\nAll assertions passed. Smoke test successful.")

if __name__ == "__main__":
    asyncio.run(main())
