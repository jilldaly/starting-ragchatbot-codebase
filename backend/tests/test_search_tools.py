"""
Unit tests for CourseSearchTool.execute() and ToolManager.

VectorStore is fully mocked — no ChromaDB involved.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_results(docs, metas, distances=None):
    distances = distances or [0.1] * len(docs)
    return SearchResults(documents=docs, metadata=metas, distances=distances)


def _make_error_results(msg):
    return SearchResults.empty(msg)


# ---------------------------------------------------------------------------
# CourseSearchTool.execute()
# ---------------------------------------------------------------------------

class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute()"""

    def setup_method(self):
        self.mock_store = MagicMock()
        self.tool = CourseSearchTool(self.mock_store)

    def test_execute_returns_formatted_string_on_results(self):
        """execute() with results returns a non-empty formatted string."""
        self.mock_store.search.return_value = _make_results(
            docs=["Python uses indentation."],
            metas=[{"course_title": "Intro Python", "lesson_number": 1, "lesson_link": ""}],
        )
        result = self.tool.execute(query="what is indentation")
        assert "Intro Python" in result
        assert "Python uses indentation." in result

    def test_execute_calls_store_search_with_correct_args(self):
        """execute() passes query, course_name, lesson_number to store.search."""
        self.mock_store.search.return_value = _make_results(
            docs=["content"],
            metas=[{"course_title": "X", "lesson_number": 2, "lesson_link": ""}],
        )
        self.tool.execute(query="loops", course_name="Python", lesson_number=2)
        self.mock_store.search.assert_called_once_with(
            query="loops", course_name="Python", lesson_number=2
        )

    def test_execute_returns_no_results_message_when_empty(self):
        """execute() with empty results returns 'No relevant content found'."""
        self.mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        result = self.tool.execute(query="obscure topic")
        assert "No relevant content found" in result

    def test_execute_no_results_includes_course_filter_info(self):
        """'No relevant content found' includes the course name when provided."""
        self.mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        result = self.tool.execute(query="something", course_name="Advanced ML")
        assert "Advanced ML" in result

    def test_execute_no_results_includes_lesson_filter_info(self):
        """'No relevant content found' includes the lesson number when provided."""
        self.mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        result = self.tool.execute(query="something", lesson_number=3)
        assert "lesson 3" in result.lower()

    def test_execute_returns_error_string_on_search_error(self):
        """execute() propagates the error string from SearchResults.error."""
        self.mock_store.search.return_value = _make_error_results(
            "Search error: collection is empty"
        )
        result = self.tool.execute(query="anything")
        assert "Search error" in result

    def test_execute_passes_course_name_filter_to_store(self):
        """course_name kwarg is forwarded to store.search()."""
        self.mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        self.tool.execute(query="q", course_name="MCP Course")
        _, kwargs = self.mock_store.search.call_args
        assert kwargs["course_name"] == "MCP Course"

    def test_execute_passes_lesson_number_filter_to_store(self):
        """lesson_number kwarg is forwarded to store.search()."""
        self.mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        self.tool.execute(query="q", lesson_number=5)
        _, kwargs = self.mock_store.search.call_args
        assert kwargs["lesson_number"] == 5


# ---------------------------------------------------------------------------
# CourseSearchTool._format_results — source tracking
# ---------------------------------------------------------------------------

class TestFormatResultsSourceTracking:
    """Tests for _format_results and last_sources attribute."""

    def setup_method(self):
        self.mock_store = MagicMock()
        self.tool = CourseSearchTool(self.mock_store)

    def test_last_sources_populated_after_results(self):
        """last_sources is set to non-empty list after successful search."""
        self.mock_store.search.return_value = _make_results(
            docs=["doc1"],
            metas=[{"course_title": "Course A", "lesson_number": 1, "lesson_link": ""}],
        )
        self.tool.execute(query="test")
        assert len(self.tool.last_sources) == 1
        assert "Course A" in self.tool.last_sources[0]
        assert "Lesson 1" in self.tool.last_sources[0]

    def test_last_sources_includes_markdown_link_when_lesson_link_present(self):
        """When lesson_link is set, source entry is a markdown hyperlink."""
        self.mock_store.search.return_value = _make_results(
            docs=["doc1"],
            metas=[{
                "course_title": "Course B",
                "lesson_number": 2,
                "lesson_link": "https://example.com/lesson2",
            }],
        )
        self.tool.execute(query="test")
        src = self.tool.last_sources[0]
        assert "https://example.com/lesson2" in src
        assert src.startswith("[")
        assert "](" in src

    def test_last_sources_plain_label_when_no_lesson_link(self):
        """When lesson_link is empty, source entry is a plain string."""
        self.mock_store.search.return_value = _make_results(
            docs=["doc1"],
            metas=[{"course_title": "Course C", "lesson_number": 3, "lesson_link": ""}],
        )
        self.tool.execute(query="test")
        src = self.tool.last_sources[0]
        assert "Course C" in src
        assert not src.startswith("[")

    def test_last_sources_empty_on_no_results(self):
        """last_sources stays empty when results are empty."""
        self.mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        self.tool.execute(query="nothing")
        assert self.tool.last_sources == []

    def test_last_sources_multiple_documents(self):
        """last_sources contains one entry per returned document."""
        self.mock_store.search.return_value = _make_results(
            docs=["doc1", "doc2"],
            metas=[
                {"course_title": "CourseX", "lesson_number": 1, "lesson_link": ""},
                {"course_title": "CourseY", "lesson_number": 2, "lesson_link": ""},
            ],
        )
        self.tool.execute(query="multi")
        assert len(self.tool.last_sources) == 2


# ---------------------------------------------------------------------------
# ToolManager
# ---------------------------------------------------------------------------

class TestToolManager:
    """Tests for ToolManager orchestration."""

    def _make_manager_with_tool(self):
        mock_store = MagicMock()
        mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
        tool = CourseSearchTool(mock_store)
        mgr = ToolManager()
        mgr.register_tool(tool)
        return mgr, tool

    def test_execute_tool_routes_to_correct_tool(self):
        """execute_tool('search_course_content', ...) runs without error."""
        mgr, _ = self._make_manager_with_tool()
        result = mgr.execute_tool("search_course_content", query="loops")
        assert isinstance(result, str)

    def test_execute_tool_unknown_name_returns_error_string(self):
        """execute_tool with unrecognised name returns a 'not found' string, not an exception."""
        mgr = ToolManager()
        result = mgr.execute_tool("nonexistent_tool", query="x")
        assert "not found" in result.lower()

    def test_get_last_sources_returns_tool_last_sources(self):
        """get_last_sources() collects last_sources from registered tools."""
        mock_store = MagicMock()
        mock_store.search.return_value = _make_results(
            docs=["doc"],
            metas=[{"course_title": "CS101", "lesson_number": 1, "lesson_link": ""}],
        )
        tool = CourseSearchTool(mock_store)
        mgr = ToolManager()
        mgr.register_tool(tool)
        mgr.execute_tool("search_course_content", query="test")
        sources = mgr.get_last_sources()
        assert len(sources) >= 1

    def test_reset_sources_clears_all_tool_sources(self):
        """reset_sources() sets last_sources to [] on all tools."""
        mock_store = MagicMock()
        mock_store.search.return_value = _make_results(
            docs=["doc"],
            metas=[{"course_title": "CS101", "lesson_number": 1, "lesson_link": ""}],
        )
        tool = CourseSearchTool(mock_store)
        mgr = ToolManager()
        mgr.register_tool(tool)
        mgr.execute_tool("search_course_content", query="test")
        assert len(tool.last_sources) > 0
        mgr.reset_sources()
        assert tool.last_sources == []

    def test_get_tool_definitions_returns_list_with_correct_name(self):
        """get_tool_definitions() returns a list containing the search tool schema."""
        mgr, _ = self._make_manager_with_tool()
        defs = mgr.get_tool_definitions()
        assert isinstance(defs, list)
        assert len(defs) == 1
        assert defs[0]["name"] == "search_course_content"
