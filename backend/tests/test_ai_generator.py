"""
Unit tests for AIGenerator.generate_response() and _run_agentic_loop().

The anthropic.Anthropic client is mocked entirely — no real API calls.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Helpers — produce real anthropic.types objects so attribute access is exact
# ---------------------------------------------------------------------------

def _text_message(text: str, stop_reason: str = "end_turn") -> Message:
    return Message(
        id="msg_001",
        content=[TextBlock(type="text", text=text)],
        model="claude-sonnet-4-6",
        role="assistant",
        stop_reason=stop_reason,
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def _tool_use_message(tool_name: str, tool_input: dict, tool_id: str = "tu_001") -> Message:
    return Message(
        id="msg_002",
        content=[
            ToolUseBlock(
                type="tool_use",
                id=tool_id,
                name=tool_name,
                input=tool_input,
            )
        ],
        model="claude-sonnet-4-6",
        role="assistant",
        stop_reason="tool_use",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=15, output_tokens=8),
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def generator_and_mock_client():
    """Return an AIGenerator whose internal client.messages.create is mocked."""
    with patch("ai_generator.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        yield gen, mock_client


# ---------------------------------------------------------------------------
# Direct text response (no tool use)
# ---------------------------------------------------------------------------

class TestDirectTextResponse:
    """generate_response() when Claude returns stop_reason='end_turn'."""

    def test_returns_text_content_directly(self, generator_and_mock_client):
        """Simple text response: result equals content[0].text."""
        gen, mock_client = generator_and_mock_client
        mock_client.messages.create.return_value = _text_message("42 is the answer.")
        result = gen.generate_response(query="what is 42?")
        assert result == "42 is the answer."

    def test_no_tools_provided_omits_tools_from_api_call(self, generator_and_mock_client):
        """When tools=None, the API call omits 'tools' and 'tool_choice' keys."""
        gen, mock_client = generator_and_mock_client
        mock_client.messages.create.return_value = _text_message("answer")
        gen.generate_response(query="hello")
        _, kwargs = mock_client.messages.create.call_args
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    def test_query_sent_as_user_message(self, generator_and_mock_client):
        """The query string is placed in messages as role='user'."""
        gen, mock_client = generator_and_mock_client
        mock_client.messages.create.return_value = _text_message("resp")
        gen.generate_response(query="Tell me about Python")
        _, kwargs = mock_client.messages.create.call_args
        messages = kwargs["messages"]
        assert messages[0]["role"] == "user"
        assert "Tell me about Python" in messages[0]["content"]

    def test_conversation_history_included_in_system_prompt(self, generator_and_mock_client):
        """conversation_history is appended to the system prompt when provided."""
        gen, mock_client = generator_and_mock_client
        mock_client.messages.create.return_value = _text_message("resp")
        gen.generate_response(
            query="follow-up",
            conversation_history="User: hi\nAssistant: hello",
        )
        _, kwargs = mock_client.messages.create.call_args
        assert "hi" in kwargs["system"]
        assert "hello" in kwargs["system"]

    def test_only_one_api_call_when_no_tool_use(self, generator_and_mock_client):
        """When stop_reason != 'tool_use', only a single messages.create call is made."""
        gen, mock_client = generator_and_mock_client
        mock_client.messages.create.return_value = _text_message("answer")
        gen.generate_response(query="simple question")
        assert mock_client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# Single tool round (tool use → text within budget)
# ---------------------------------------------------------------------------

class TestToolUseFlow:
    """generate_response() with one round of tool use."""

    def test_one_tool_round_makes_two_api_calls(self, generator_and_mock_client):
        """One tool call followed by a text response = 2 total API calls."""
        gen, mock_client = generator_and_mock_client
        mock_client.messages.create.side_effect = [
            _tool_use_message("search_course_content", {"query": "Python loops"}),
            _text_message("Loops repeat code blocks."),
        ]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Tool result: info about loops."

        result = gen.generate_response(
            query="What are Python loops?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )
        assert mock_client.messages.create.call_count == 2
        assert result == "Loops repeat code blocks."

    def test_tool_manager_execute_tool_called_with_correct_args(
        self, generator_and_mock_client
    ):
        """tool_manager.execute_tool is called with tool_name and unpacked input kwargs."""
        gen, mock_client = generator_and_mock_client
        mock_client.messages.create.side_effect = [
            _tool_use_message(
                "search_course_content",
                {"query": "decorators", "course_name": "Advanced Python"},
                tool_id="tu_xyz",
            ),
            _text_message("Decorators wrap functions."),
        ]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "some tool result"

        gen.generate_response(
            query="explain decorators",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="decorators",
            course_name="Advanced Python",
        )

    def test_tool_result_sent_back_with_correct_tool_use_id(
        self, generator_and_mock_client
    ):
        """Second API call includes a tool_result block with the matching tool_use_id."""
        gen, mock_client = generator_and_mock_client
        tool_id = "tu_result_check"
        mock_client.messages.create.side_effect = [
            _tool_use_message("search_course_content", {"query": "classes"}, tool_id=tool_id),
            _text_message("Classes are blueprints."),
        ]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Classes info here."

        gen.generate_response(
            query="classes?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        tool_result_block = None
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for block in msg["content"]:
                    if block.get("type") == "tool_result":
                        tool_result_block = block
        assert tool_result_block is not None
        assert tool_result_block["tool_use_id"] == tool_id
        assert tool_result_block["content"] == "Classes info here."

    def test_second_api_call_includes_tools_while_within_budget(
        self, generator_and_mock_client
    ):
        """
        After round 1, the second API call still includes tools because
        tool_rounds_used (1) < max_rounds (2). Claude can make a second tool call.
        """
        gen, mock_client = generator_and_mock_client
        mock_client.messages.create.side_effect = [
            _tool_use_message("search_course_content", {"query": "async"}),
            _text_message("Async allows concurrent code."),
        ]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "async content"

        gen.generate_response(
            query="async?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs
        assert "tool_choice" in second_call_kwargs

    def test_final_text_returned_from_second_response(self, generator_and_mock_client):
        """Return value is content[0].text of the second (follow-up) response."""
        gen, mock_client = generator_and_mock_client
        mock_client.messages.create.side_effect = [
            _tool_use_message("search_course_content", {"query": "test"}),
            _text_message("The final synthesized answer."),
        ]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "intermediate result"

        result = gen.generate_response(
            query="test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )
        assert result == "The final synthesized answer."

    def test_tools_included_in_first_call(self, generator_and_mock_client):
        """When tools list is non-empty, first API call includes 'tools' and 'tool_choice'."""
        gen, mock_client = generator_and_mock_client
        tool_def = {"name": "search_course_content", "description": "search"}
        mock_client.messages.create.return_value = _text_message("direct answer")

        gen.generate_response(
            query="q",
            tools=[tool_def],
            tool_manager=MagicMock(),
        )
        _, kwargs = mock_client.messages.create.call_args
        assert "tools" in kwargs
        assert kwargs["tools"] == [tool_def]
        assert kwargs["tool_choice"] == {"type": "auto"}


# ---------------------------------------------------------------------------
# Sequential tool calling — two rounds
# ---------------------------------------------------------------------------

class TestSequentialToolCalling:
    """
    Tests for the two-round agentic loop introduced in _run_agentic_loop.
    Scenarios: tool_use → tool_use → text (3 API calls total).
    """

    def _two_round_side_effect(self, tool1_name="get_course_outline",
                                tool1_input=None, tool1_id="tu_r1",
                                tool2_name="search_course_content",
                                tool2_input=None, tool2_id="tu_r2",
                                final_text="Here is your answer."):
        return [
            _tool_use_message(tool1_name, tool1_input or {"course_name": "Python"}, tool1_id),
            _tool_use_message(tool2_name, tool2_input or {"query": "lesson 4 topic"}, tool2_id),
            _text_message(final_text),
        ]

    def test_two_tool_rounds_makes_three_api_calls(self, generator_and_mock_client):
        """Two rounds of tool use → 3 total API calls (2 tool rounds + 1 synthesis)."""
        gen, mock_client = generator_and_mock_client
        mock_client.messages.create.side_effect = self._two_round_side_effect()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "some result"

        result = gen.generate_response(
            query="Search for a course about the same topic as lesson 4 of Python",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )
        assert mock_client.messages.create.call_count == 3
        assert result == "Here is your answer."

    def test_both_tools_executed_across_two_rounds(self, generator_and_mock_client):
        """execute_tool is called exactly twice — once per tool round."""
        gen, mock_client = generator_and_mock_client
        mock_client.messages.create.side_effect = self._two_round_side_effect()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "result"

        gen.generate_response(
            query="multi-step query",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )
        assert mock_tool_manager.execute_tool.call_count == 2

    def test_different_tools_called_in_each_round(self, generator_and_mock_client):
        """Round 1 calls get_course_outline; round 2 calls search_course_content."""
        gen, mock_client = generator_and_mock_client
        mock_client.messages.create.side_effect = self._two_round_side_effect()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "result"

        gen.generate_response(
            query="multi-step query",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )
        calls = mock_tool_manager.execute_tool.call_args_list
        assert calls[0][0][0] == "get_course_outline"
        assert calls[1][0][0] == "search_course_content"

    def test_round_two_api_call_includes_tools(self, generator_and_mock_client):
        """
        The second API call (round 2 in the loop) includes tools because
        tool_rounds_used (1) is still < max_rounds (2).
        """
        gen, mock_client = generator_and_mock_client
        mock_client.messages.create.side_effect = self._two_round_side_effect()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "result"

        gen.generate_response(
            query="multi-step query",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs
        assert "tool_choice" in second_call_kwargs

    def test_synthesis_call_after_two_rounds_omits_tools(self, generator_and_mock_client):
        """
        The third API call (synthesis after 2 tool rounds) omits tools — the budget
        is exhausted, so Claude is forced to produce a text response.
        """
        gen, mock_client = generator_and_mock_client
        mock_client.messages.create.side_effect = self._two_round_side_effect()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "result"

        gen.generate_response(
            query="multi-step query",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )
        third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
        assert "tools" not in third_call_kwargs
        assert "tool_choice" not in third_call_kwargs

    def test_messages_grow_correctly_across_two_rounds(self, generator_and_mock_client):
        """
        The synthesis (3rd) call receives 5 messages:
        [user query, assistant r1, tool_results r1, assistant r2, tool_results r2]
        """
        gen, mock_client = generator_and_mock_client
        mock_client.messages.create.side_effect = self._two_round_side_effect()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "result"

        gen.generate_response(
            query="multi-step query",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )
        third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
        messages = third_call_kwargs["messages"]
        assert len(messages) == 5
        assert messages[0]["role"] == "user"       # original query
        assert messages[1]["role"] == "assistant"  # round 1 tool use
        assert messages[2]["role"] == "user"       # round 1 tool results
        assert messages[3]["role"] == "assistant"  # round 2 tool use
        assert messages[4]["role"] == "user"       # round 2 tool results

    def test_round_one_tool_result_present_in_round_two_messages(
        self, generator_and_mock_client
    ):
        """Round 2 API call messages include the tool result from round 1."""
        gen, mock_client = generator_and_mock_client
        mock_client.messages.create.side_effect = self._two_round_side_effect(
            tool1_id="tu_r1"
        )
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "round-1-result"

        gen.generate_response(
            query="multi-step query",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )
        # The messages list is mutated in-place across rounds, so by the time we
        # inspect call_args_list[1] it already contains all 5 messages. Search for
        # the specific tool_use_id we care about rather than "any" tool_result.
        third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
        messages = third_call_kwargs["messages"]
        r1_tool_result = None
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for block in msg["content"]:
                    if block.get("type") == "tool_result" and block.get("tool_use_id") == "tu_r1":
                        r1_tool_result = block
        assert r1_tool_result is not None
        assert r1_tool_result["content"] == "round-1-result"

    def test_tool_error_string_passed_through_to_claude(self, generator_and_mock_client):
        """
        When execute_tool returns an error string (e.g. tool not found),
        it is forwarded to Claude as a tool_result — not swallowed.
        Claude can then reason about the failure and provide a text response.
        """
        gen, mock_client = generator_and_mock_client
        mock_client.messages.create.side_effect = [
            _tool_use_message("search_course_content", {"query": "test"}, tool_id="tu_err"),
            _text_message("I could not find that course."),
        ]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Tool 'search_course_content' not found"

        result = gen.generate_response(
            query="test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        tool_result_block = None
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for block in msg["content"]:
                    if block.get("type") == "tool_result":
                        tool_result_block = block
        assert tool_result_block is not None
        assert "not found" in tool_result_block["content"]
        assert result == "I could not find that course."


# ---------------------------------------------------------------------------
# Error propagation — AIGenerator has no try/except
# ---------------------------------------------------------------------------

class TestAIGeneratorErrorPropagation:
    """
    AIGenerator has NO try/except. Exceptions from the Anthropic SDK propagate
    unhandled to RAGSystem, then caught there for a graceful error response.
    """

    def test_sdk_exception_propagates_from_generate_response(
        self, generator_and_mock_client
    ):
        """If messages.create raises on the first call, generate_response propagates it."""
        import anthropic as anthropic_module
        gen, mock_client = generator_and_mock_client

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_client.messages.create.side_effect = anthropic_module.NotFoundError(
            message="model: claude-sonnet-4-20250514 not found",
            response=mock_response,
            body={"type": "error", "error": {"type": "not_found_error"}},
        )

        with pytest.raises(anthropic_module.NotFoundError) as exc_info:
            gen.generate_response(query="anything")

        assert "claude-sonnet-4-20250514" in str(exc_info.value)

    def test_sdk_exception_propagates_from_second_api_call_in_tool_flow(
        self, generator_and_mock_client
    ):
        """If the second messages.create call raises (inside _run_agentic_loop), it propagates."""
        import anthropic as anthropic_module
        gen, mock_client = generator_and_mock_client

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_client.messages.create.side_effect = [
            _tool_use_message("search_course_content", {"query": "test"}),
            anthropic_module.NotFoundError(
                message="model: claude-sonnet-4-20250514 not found",
                response=mock_response,
                body={"type": "error", "error": {"type": "not_found_error"}},
            ),
        ]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "tool result"

        with pytest.raises(anthropic_module.NotFoundError):
            gen.generate_response(
                query="test",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager,
            )
