import os
import litellm
from litellm import acompletion


class TargetAdapter:
    """
    Wraps the model being tested.
    Takes a system prompt + user message, returns the raw response.
    Works with any OpenAI-compatible endpoint via litellm.
    """

    def __init__(self, system_prompt: str, model: str = None):
        self.system_prompt = system_prompt
        self.model = model or os.getenv("TARGET_MODEL", "gpt-4o-mini")
        self.call_count = 0

    async def send(self, user_message: str, history: list[dict] = None) -> str:
        """
        Send a single message to the target.
        history: optional list of prior turns for multi-turn sessions.
        Returns the raw text response.
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_message})

        response = await acompletion(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )

        self.call_count += 1
        return response.choices[0].message.content

    async def send_with_tools(
        self,
        user_message: str,
        tools: list[dict],
        history: list[dict] = None,
    ) -> tuple[str, list[dict]]:
        """
        Send a message to an agentic target that has tools.
        Returns (text_response, tool_calls_made).
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_message})

        response = await acompletion(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=0.7,
        )

        self.call_count += 1
        message = response.choices[0].message
        text = message.content or ""
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                    "id": tc.id,
                })

        return text, tool_calls
