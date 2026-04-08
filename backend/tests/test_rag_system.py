"""
Unit tests for RAGSystem.query() orchestration.

AIGenerator, VectorStore, and DocumentProcessor are patched at the module level
inside rag_system's namespace. SessionManager is real (pure Python, no I/O).
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def rag_system_under_test():
    """
    Return a RAGSystem with VectorStore, AIGenerator, and DocumentProcessor
    patched out. SessionManager is real.
    """
    with (
        patch("rag_system.VectorStore") as mock_vs_cls,
        patch("rag_system.AIGenerator") as mock_ai_cls,
        patch("rag_system.DocumentProcessor") as mock_dp_cls,
    ):
        mock_vs = MagicMock()
        mock_ai = MagicMock()
        mock_dp = MagicMock()

        mock_vs_cls.return_value = mock_vs
        mock_ai_cls.return_value = mock_ai
        mock_dp_cls.return_value = mock_dp

        from config import Config
        from rag_system import RAGSystem

        cfg = Config(ANTHROPIC_API_KEY="sk-test", ANTHROPIC_MODEL="claude-sonnet-4-6")
        system = RAGSystem(cfg)

        # Expose mocks for assertions
        system._mock_ai = mock_ai
        system._mock_vs = mock_vs

        yield system


# ---------------------------------------------------------------------------
# Core query() orchestration
# ---------------------------------------------------------------------------

class TestRAGSystemQuery:

    def test_query_calls_generate_response_once(self, rag_system_under_test):
        """query() calls ai_generator.generate_response() exactly once."""
        sys = rag_system_under_test
        sys._mock_ai.generate_response.return_value = "Great answer."
        sys.query("What is Python?")
        sys._mock_ai.generate_response.assert_called_once()

    def test_query_passes_non_empty_tools_list(self, rag_system_under_test):
        """query() passes a non-empty tools list to generate_response."""
        sys = rag_system_under_test
        sys._mock_ai.generate_response.return_value = "answer"
        sys.query("What is Python?")
        _, kwargs = sys._mock_ai.generate_response.call_args
        assert "tools" in kwargs
        assert isinstance(kwargs["tools"], list)
        assert len(kwargs["tools"]) > 0

    def test_query_passes_tool_manager(self, rag_system_under_test):
        """query() passes tool_manager= to generate_response."""
        sys = rag_system_under_test
        sys._mock_ai.generate_response.return_value = "answer"
        sys.query("anything")
        _, kwargs = sys._mock_ai.generate_response.call_args
        assert "tool_manager" in kwargs
        assert kwargs["tool_manager"] is sys.tool_manager

    def test_query_returns_response_and_sources_tuple(self, rag_system_under_test):
        """query() returns a (str, list) tuple."""
        sys = rag_system_under_test
        sys._mock_ai.generate_response.return_value = "The answer."
        result = sys.query("anything")
        assert isinstance(result, tuple)
        assert len(result) == 2
        response, sources = result
        assert isinstance(response, str)
        assert isinstance(sources, list)

    def test_query_sources_come_from_tool_manager(self, rag_system_under_test):
        """Sources in the return value are from tool_manager.get_last_sources()."""
        sys = rag_system_under_test
        sys._mock_ai.generate_response.return_value = "answer"
        sys.search_tool.last_sources = [
            "[Python Basics - Lesson 1](https://example.com/lesson1)"
        ]
        _, sources = sys.query("question")
        assert len(sources) == 1
        assert "Python Basics" in sources[0]

    def test_query_resets_sources_after_retrieval(self, rag_system_under_test):
        """After query(), search_tool.last_sources is empty."""
        sys = rag_system_under_test
        sys._mock_ai.generate_response.return_value = "answer"
        sys.search_tool.last_sources = ["[Course A - Lesson 1](https://url)"]
        sys.query("first question")
        assert sys.search_tool.last_sources == []

    def test_query_sources_empty_when_no_tool_used(self, rag_system_under_test):
        """Sources is empty list when AI gave a direct answer (no tool called)."""
        sys = rag_system_under_test
        sys._mock_ai.generate_response.return_value = "Direct knowledge answer."
        _, sources = sys.query("general question")
        assert sources == []


# ---------------------------------------------------------------------------
# Session handling
# ---------------------------------------------------------------------------

class TestRAGSystemSessionHandling:

    def test_no_session_id_passes_none_history(self, rag_system_under_test):
        """Without session_id, conversation_history kwarg is None."""
        sys = rag_system_under_test
        sys._mock_ai.generate_response.return_value = "answer"
        sys.query("no session query")
        _, kwargs = sys._mock_ai.generate_response.call_args
        assert kwargs.get("conversation_history") is None

    def test_session_id_with_history_passes_history_string(self, rag_system_under_test):
        """When session has history, conversation_history is a non-None string."""
        sys = rag_system_under_test
        sys._mock_ai.generate_response.return_value = "answer"

        session_id = sys.session_manager.create_session()
        sys.session_manager.add_exchange(session_id, "prior question", "prior answer")

        sys.query("follow-up", session_id=session_id)
        _, kwargs = sys._mock_ai.generate_response.call_args
        history = kwargs.get("conversation_history")
        assert history is not None
        assert "prior question" in history

    def test_query_updates_session_history(self, rag_system_under_test):
        """After query(), session history contains the new user query and AI response."""
        sys = rag_system_under_test
        sys._mock_ai.generate_response.return_value = "Updated answer."

        session_id = sys.session_manager.create_session()
        sys.query("my new question", session_id=session_id)

        history = sys.session_manager.get_conversation_history(session_id)
        assert history is not None
        assert "my new question" in history
        assert "Updated answer." in history

    def test_no_session_update_without_session_id(self, rag_system_under_test):
        """When no session_id is provided, session_manager.add_exchange is not called."""
        sys = rag_system_under_test
        sys._mock_ai.generate_response.return_value = "answer"

        mock_sm = MagicMock()
        mock_sm.get_conversation_history.return_value = None
        sys.session_manager = mock_sm

        sys.query("ephemeral question")
        mock_sm.add_exchange.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling — RAGSystem.query() catches AI exceptions gracefully
# ---------------------------------------------------------------------------

class TestRAGSystemErrorHandling:
    """
    RAGSystem.query() catches exceptions from AIGenerator and returns a graceful
    error message tuple instead of propagating to FastAPI as HTTP 500.
    """

    def test_ai_generator_exception_returns_graceful_error_message(
        self, rag_system_under_test
    ):
        """
        If generate_response() raises (e.g. NotFoundError for invalid model),
        query() catches it and returns a user-friendly error string.
        The fix: config.py model changed from "claude-sonnet-4-20250514" to
        "claude-sonnet-4-6", and rag_system.query() now has try/except.
        """
        import anthropic as anthropic_module
        sys = rag_system_under_test

        mock_response = MagicMock()
        mock_response.status_code = 404
        sys._mock_ai.generate_response.side_effect = anthropic_module.NotFoundError(
            message="model: claude-sonnet-4-20250514 not found",
            response=mock_response,
            body={"type": "error", "error": {"type": "not_found_error"}},
        )

        response, sources = sys.query("will this work?")

        assert isinstance(response, str)
        assert "error" in response.lower()
        assert sources == []

    def test_any_exception_from_ai_generator_returns_error_tuple(
        self, rag_system_under_test
    ):
        """Generic exceptions are caught — query() returns (error_str, []) not raises."""
        sys = rag_system_under_test
        sys._mock_ai.generate_response.side_effect = RuntimeError("unexpected failure")

        response, sources = sys.query("test query")
        assert isinstance(response, str)
        assert "error" in response.lower()
        assert sources == []
