"""
Probe — HTTP Target Adapter

Lets Probe attack real deployed chatbots over HTTP instead of calling
an LLM API directly. Works with any REST-based chat endpoint.

Built-in presets:
  - Botpress Cloud  (chat.botpress.cloud)
  - Voiceflow       (general-runtime.voiceflow.com)
  - Generic REST    (configurable for anything else)

Usage:
  python run.py --http-target targets/http/botpress_example.yaml

Finding a company's endpoint:
  1. Open their website in Chrome
  2. DevTools → Network tab → filter "Fetch/XHR"
  3. Type a message in their chat widget
  4. Look for the POST request — that's your endpoint + payload shape
"""

import asyncio
import uuid
import json
import httpx


class BotpressTarget:
    """
    Calls a Botpress Cloud bot via the Chat API.

    The webhook_id is visible in the company's page source — search for
    'chat.botpress.cloud' or 'webhookId' in the page HTML/JS.

    API: https://chat.botpress.cloud/{webhook_id}/messages
    Docs: https://botpress.com/docs/api-reference/chat-api/introduction
    """

    def __init__(
        self,
        webhook_id: str,
        system_prompt: str = "",
        user_key: str = None,
        base_url: str = "https://chat.botpress.cloud",
    ):
        self.webhook_id = webhook_id
        self.system_prompt = system_prompt  # kept for interface compatibility
        self.user_key = user_key
        self.base_url = base_url.rstrip("/")
        self.call_count = 0
        self.model = f"botpress:{webhook_id[:8]}"

        # Botpress requires a persistent conversation ID
        self._conversation_id: str | None = None
        self._user_id: str = f"probe-{uuid.uuid4().hex[:8]}"

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.user_key:
            h["x-user-key"] = self.user_key
        return h

    async def _ensure_conversation(self) -> str:
        """Create a new conversation if we don't have one."""
        if self._conversation_id:
            return self._conversation_id

        async with httpx.AsyncClient(timeout=20) as client:
            # Create or get user
            r = await client.post(
                f"{self.base_url}/{self.webhook_id}/users",
                headers=self._headers(),
                json={"id": self._user_id},
            )
            # Create conversation
            r = await client.post(
                f"{self.base_url}/{self.webhook_id}/conversations",
                headers=self._headers(),
                json={"id": uuid.uuid4().hex},
            )
            r.raise_for_status()
            self._conversation_id = r.json().get("id") or r.json().get("conversation", {}).get("id")

        return self._conversation_id

    async def send(self, user_message: str, history: list[dict] = None) -> str:
        """Send a message to the Botpress bot and return the response."""
        # For multi-turn, Botpress maintains state via conversation ID
        # so we don't need to replay history explicitly
        conv_id = await self._ensure_conversation()

        async with httpx.AsyncClient(timeout=30) as client:
            # Send message
            r = await client.post(
                f"{self.base_url}/{self.webhook_id}/messages",
                headers=self._headers(),
                json={
                    "conversationId": conv_id,
                    "userId": self._user_id,
                    "payload": {"type": "text", "text": user_message},
                },
            )
            r.raise_for_status()
            self.call_count += 1

            # Poll for response (Botpress is async)
            return await self._poll_response(conv_id, client)

    async def _poll_response(self, conv_id: str, client: httpx.AsyncClient, max_wait: int = 15) -> str:
        """Poll the conversation for the bot's reply (Botpress replies asynchronously)."""
        for _ in range(max_wait * 2):  # check every 0.5s
            await asyncio.sleep(0.5)
            r = await client.get(
                f"{self.base_url}/{self.webhook_id}/conversations/{conv_id}/messages",
                headers=self._headers(),
            )
            if r.status_code == 200:
                messages = r.json().get("messages", [])
                # Find the latest bot message
                bot_msgs = [m for m in messages if m.get("userId") != self._user_id]
                if bot_msgs:
                    latest = sorted(bot_msgs, key=lambda m: m.get("createdAt", ""))[-1]
                    payload = latest.get("payload", {})
                    return payload.get("text") or json.dumps(payload)

        return "[No response received within timeout]"

    async def send_with_tools(self, user_message: str, tools: list[dict], history: list[dict] = None):
        """Botpress handles tools internally — we just get the final text response."""
        response = await self.send(user_message, history)
        return response, []

    async def ping(self) -> bool:
        """Check if the bot endpoint is reachable."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    f"{self.base_url}/{self.webhook_id}/hello",
                    headers=self._headers(),
                )
                return r.status_code == 200
        except Exception:
            return False


class VoiceflowTarget:
    """
    Calls a Voiceflow agent via the Dialog Manager API.

    You need the API key (find it in the company's page source — search
    for 'general-runtime.voiceflow.com' or 'VF_DM_API_KEY').

    API: https://general-runtime.voiceflow.com/state/user/{user_id}/interact
    Docs: https://docs.voiceflow.com/reference/overview
    """

    def __init__(
        self,
        api_key: str,
        version_id: str = "production",
        system_prompt: str = "",
        base_url: str = "https://general-runtime.voiceflow.com",
    ):
        self.api_key = api_key
        self.version_id = version_id
        self.system_prompt = system_prompt
        self.base_url = base_url.rstrip("/")
        self.call_count = 0
        self.model = "voiceflow"
        self._user_id = f"probe-{uuid.uuid4().hex[:8]}"

    def _headers(self) -> dict:
        return {
            "Authorization": self.api_key,
            "versionID": self.version_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def send(self, user_message: str, history: list[dict] = None) -> str:
        """Send a message and return the agent's text response."""
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{self.base_url}/state/user/{self._user_id}/interact",
                headers=self._headers(),
                json={"action": {"type": "text", "payload": user_message}},
            )
            r.raise_for_status()
            self.call_count += 1

            traces = r.json()
            texts = []
            for trace in traces:
                if trace.get("type") == "text":
                    msg = trace.get("payload", {}).get("message", "")
                    if msg:
                        texts.append(msg)
                elif trace.get("type") == "speak":
                    msg = trace.get("payload", {}).get("message", "")
                    if msg:
                        texts.append(msg)

            return " ".join(texts) if texts else "[Empty response]"

    async def send_with_tools(self, user_message: str, tools: list[dict], history: list[dict] = None):
        response = await self.send(user_message, history)
        return response, []

    async def reset(self):
        """Reset the user state (start a fresh conversation)."""
        async with httpx.AsyncClient(timeout=10) as client:
            await client.delete(
                f"{self.base_url}/state/user/{self._user_id}",
                headers=self._headers(),
            )


class GenericHTTPTarget:
    """
    Calls any REST chat endpoint.

    Configure with a YAML target file that specifies the endpoint,
    headers, and payload template. Use {message} as the placeholder
    for the user's message in the payload template.

    Example target YAML:
        http_target:
          type: generic
          endpoint: https://api.example.com/chat
          headers:
            Authorization: "Bearer your-token"
            Content-Type: "application/json"
          payload_template: |
            {"message": "{message}", "session_id": "{session_id}"}
          response_path: "reply.text"  # dot-notation path to the response text
    """

    def __init__(
        self,
        endpoint: str,
        headers: dict = None,
        payload_template: str = None,
        response_path: str = "text",
        system_prompt: str = "",
    ):
        self.endpoint = endpoint
        self.headers = headers or {"Content-Type": "application/json"}
        self.payload_template = payload_template or '{"message": "{message}"}'
        self.response_path = response_path
        self.system_prompt = system_prompt
        self.call_count = 0
        self.model = f"http:{endpoint[:40]}"
        self._session_id = uuid.uuid4().hex

    async def send(self, user_message: str, history: list[dict] = None) -> str:
        payload_str = (
            self.payload_template
            .replace("{message}", user_message.replace('"', '\\"'))
            .replace("{session_id}", self._session_id)
        )
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            payload = {"message": user_message}

        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(self.endpoint, headers=self.headers, json=payload)
            r.raise_for_status()
            self.call_count += 1
            return self._extract_response(r.json())

    def _extract_response(self, data: dict) -> str:
        """Navigate dot-notation path to extract text response."""
        parts = self.response_path.split(".")
        result = data
        for part in parts:
            if isinstance(result, dict):
                result = result.get(part, data)
            else:
                break
        if isinstance(result, str):
            return result
        return json.dumps(result)

    async def send_with_tools(self, user_message: str, tools: list[dict], history: list[dict] = None):
        response = await self.send(user_message, history)
        return response, []


def load_http_target(config: dict) -> "BotpressTarget | VoiceflowTarget | GenericHTTPTarget":
    """
    Load an HTTP target from a YAML config dict (the http_target section).

    Supported types: botpress, voiceflow, generic
    """
    target_type = config.get("type", "generic").lower()
    system_prompt = config.get("system_prompt", "")

    if target_type == "botpress":
        return BotpressTarget(
            webhook_id=config["webhook_id"],
            system_prompt=system_prompt,
            user_key=config.get("user_key"),
            base_url=config.get("base_url", "https://chat.botpress.cloud"),
        )

    elif target_type == "voiceflow":
        return VoiceflowTarget(
            api_key=config["api_key"],
            version_id=config.get("version_id", "production"),
            system_prompt=system_prompt,
            base_url=config.get("base_url", "https://general-runtime.voiceflow.com"),
        )

    else:  # generic
        return GenericHTTPTarget(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            payload_template=config.get("payload_template", '{"message": "{message}"}'),
            response_path=config.get("response_path", "text"),
            system_prompt=system_prompt,
        )
