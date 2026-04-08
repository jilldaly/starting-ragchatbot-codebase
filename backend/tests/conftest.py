"""
Shared fixtures for all test modules.

All tests are pure unit tests — no real API calls, no real ChromaDB.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

# Ensure backend/ is on sys.path so imports like `from search_tools import ...` resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Anthropic SDK object factories
# These produce real anthropic.types instances (not raw MagicMocks) so that
# attribute access on .type, .text, .id, .name, .input all works correctly.
# ---------------------------------------------------------------------------

def make_text_message(text: str, stop_reason: str = "end_turn") -> Message:
    """Return an Anthropic Message with a single TextBlock."""
    return Message(
        id="msg_text_001",
        content=[TextBlock(type="text", text=text)],
        model="claude-sonnet-4-6",
        role="assistant",
        stop_reason=stop_reason,
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=20, output_tokens=10),
    )


def make_tool_use_message(
    tool_name: str,
    tool_input: dict,
    tool_use_id: str = "tu_001",
) -> Message:
    """Return an Anthropic Message with a ToolUseBlock (stop_reason='tool_use')."""
    return Message(
        id="msg_tool_001",
        content=[
            ToolUseBlock(
                type="tool_use",
                id=tool_use_id,
                name=tool_name,
                input=tool_input,
            )
        ],
        model="claude-sonnet-4-6",
        role="assistant",
        stop_reason="tool_use",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=25, output_tokens=15),
    )


@pytest.fixture
def mock_vector_store():
    """A fully mocked VectorStore — no ChromaDB on disk."""
    return MagicMock()


@pytest.fixture
def sample_search_results():
    """
    Returns a factory callable so tests can customise documents/metadata.
    Usage: sample_search_results(docs=[...], metas=[...])
    """
    from vector_store import SearchResults

    def _factory(docs=None, metas=None, distances=None, error=None):
        if error:
            return SearchResults.empty(error)
        docs = docs or ["Content about Python loops."]
        metas = metas or [
            {
                "course_title": "Intro to Python",
                "lesson_number": 1,
                "lesson_link": "https://example.com/lesson1",
            }
        ]
        distances = distances or [0.1] * len(docs)
        return SearchResults(documents=docs, metadata=metas, distances=distances)

    return _factory
