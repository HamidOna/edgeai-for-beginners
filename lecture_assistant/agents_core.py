"""
Base Agent class with tool calling capabilities for Foundry Local.

This module demonstrates proper Foundry Local connection using FoundryLocalManager
which automatically handles:
- Service discovery (dynamic port allocation)
- Model loading
- OpenAI-compatible API client setup
"""

from foundry_local import FoundryLocalManager
from openai import OpenAI
from typing import List, Dict, Any, Optional, Callable
import json


class Agent:
    """
    Base Agent class with tool calling support.

    Uses FoundryLocalManager for automatic Foundry Local service discovery.
    Supports OpenAI-style function calling with local SLMs.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model_alias: str = "phi-4-mini",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_functions: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize the agent with Foundry Local.

        Args:
            name: Agent name for logging
            system_prompt: System prompt defining agent behavior
            model_alias: Foundry Local model alias (default: phi-4-mini)
            tools: List of tool definitions in OpenAI format
            tool_functions: Dict mapping tool names to callable functions
        """
        self.name = name
        self.system_prompt = system_prompt
        self.model_alias = model_alias
        self.tools = tools or []
        self.tool_functions = tool_functions or {}
        self.memory: List[Dict[str, str]] = []

        # Initialize FoundryLocalManager - automatically discovers endpoint
        print(f"[{self.name}] Initializing Foundry Local with model: {model_alias}")
        self.manager = FoundryLocalManager(model_alias)

        # Get model info
        self.model_info = self.manager.get_model_info(model_alias)
        print(f"[{self.name}] Connected to: {self.manager.endpoint}")
        print(f"[{self.name}] Using model ID: {self.model_info.id}")

        # Create OpenAI client with discovered endpoint
        self.client = OpenAI(
            base_url=self.manager.endpoint,
            api_key=self.manager.api_key
        )

    def chat(
        self,
        user_message: str,
        max_tokens: int = 800,
        temperature: float = 0.7,
        include_memory: bool = True
    ) -> str:
        """
        Send a chat message and get response.

        Handles tool calling automatically if tools are available.

        Args:
            user_message: User input message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            include_memory: Whether to include conversation history

        Returns:
            Agent's response text
        """
        # Add user message to memory
        self.memory.append({"role": "user", "content": user_message})

        # Prepare messages
        messages = [{"role": "system", "content": self.system_prompt}]
        if include_memory:
            messages.extend(self.memory)
        else:
            messages.append({"role": "user", "content": user_message})

        # Make API call with tool support
        kwargs = {
            "model": self.model_info.id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        # Add tools if available
        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)

        # Handle tool calls if present
        message = response.choices[0].message

        # Check for tool calls (handle both old and new format)
        tool_calls = getattr(message, 'tool_calls', None)

        if tool_calls:
            print(f"[{self.name}] Model requested {len(tool_calls)} tool call(s)")

            # Add assistant message with tool calls to memory
            self.memory.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
            })

            # Execute each tool call
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"[{self.name}] Calling: {function_name}({function_args})")

                # Execute the function
                if function_name in self.tool_functions:
                    try:
                        result = self.tool_functions[function_name](**function_args)
                        result_str = json.dumps({"success": True, "result": result})
                        print(f"[{self.name}] Tool result: {result_str}")
                    except Exception as e:
                        result_str = json.dumps({"success": False, "error": str(e)})
                        print(f"[{self.name}] Tool error: {result_str}")
                else:
                    result_str = json.dumps({"success": False, "error": f"Function {function_name} not found"})
                    print(f"[{self.name}] Tool not found: {function_name}")

                # Add tool result to memory
                self.memory.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str
                })

            # Make another API call with tool results
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.memory)

            second_response = self.client.chat.completions.create(
                model=self.model_info.id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            final_content = second_response.choices[0].message.content
            self.memory.append({"role": "assistant", "content": final_content})
            return final_content
        else:
            # No tool calls, just return the message
            content = message.content or ""
            self.memory.append({"role": "assistant", "content": content})
            return content

    def clear_memory(self):
        """Clear conversation memory."""
        self.memory = []
        print(f"[{self.name}] Memory cleared")

    def get_memory(self) -> List[Dict[str, str]]:
        """Get current conversation memory."""
        return self.memory.copy()


def test_connection(model_alias: str = "phi-4-mini") -> bool:
    """
    Test connection to Foundry Local.

    Args:
        model_alias: Model to test (default: phi-4-mini)

    Returns:
        True if connection successful, False otherwise
    """
    try:
        print(f"\n{'='*60}")
        print(f"Testing Foundry Local Connection")
        print(f"{'='*60}")

        # Initialize manager
        manager = FoundryLocalManager(model_alias)
        print(f"✅ Service endpoint: {manager.endpoint}")

        # Get model info
        model_info = manager.get_model_info(model_alias)
        print(f"✅ Model ID: {model_info.id}")

        # Create client
        client = OpenAI(
            base_url=manager.endpoint,
            api_key=manager.api_key
        )

        # Test simple chat
        response = client.chat.completions.create(
            model=model_info.id,
            messages=[{"role": "user", "content": "Say 'Connection successful!' and nothing else."}],
            max_tokens=20
        )

        result = response.choices[0].message.content
        print(f"✅ Model response: {result}")
        print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Check service status: foundry service status")
        print(f"2. Start model: foundry model run {model_alias}")
        print(f"3. List models: foundry model list")
        print(f"{'='*60}\n")
        return False
