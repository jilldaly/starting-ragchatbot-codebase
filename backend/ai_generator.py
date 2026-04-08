import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use `get_course_outline` for any question about a course's structure, lessons list, or outline
- Use `search_course_content` for questions about specific course content or detailed educational materials
- **Up to two sequential tool calls allowed per query** — use a second tool call when the first result reveals you need additional information (e.g. find a lesson title, then search for related content)
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Delegate to agentic loop if tool use is requested
        if response.stop_reason == "tool_use" and tool_manager:
            messages = api_params["messages"].copy()
            return self._run_agentic_loop(
                initial_response=response,
                messages=messages,
                system=system_content,
                tools=tools,
                tool_manager=tool_manager
            )

        # Return direct response
        return response.content[0].text

    def _run_agentic_loop(self, initial_response, messages: List, system: str,
                          tools: List, tool_manager, max_rounds: int = 2) -> str:
        """
        Execute sequential tool calls in separate API rounds, up to max_rounds.

        Each round: execute tool(s) from the current response, append results to
        the conversation, then call Claude again. Tools are included in every call
        within budget; the call that exhausts the budget strips tools to force a
        final text synthesis.

        Args:
            initial_response: First tool-use response from generate_response
            messages: Conversation so far (will be extended in-place)
            system: Assembled system prompt string
            tools: Tool definitions to include while within budget
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool-use rounds allowed

        Returns:
            Final text response from Claude
        """
        current_response = initial_response
        tool_rounds_used = 0

        while current_response.stop_reason == "tool_use":
            # Append the assistant's tool-use response to conversation history
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute all tool calls in this response and collect results
            tool_results = []
            for block in current_response.content:
                if block.type == "tool_use":
                    result = tool_manager.execute_tool(block.name, **block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            tool_rounds_used += 1

            # Build next API call — include tools only while within budget
            call_params = {
                **self.base_params,
                "messages": messages,
                "system": system
            }
            if tool_rounds_used < max_rounds:
                call_params["tools"] = tools
                call_params["tool_choice"] = {"type": "auto"}

            current_response = self.client.messages.create(**call_params)

        return current_response.content[0].text
