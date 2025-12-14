from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .llm_config import LLMConfig

logger = logging.getLogger(__name__)

# System prompt for agentic robot control
ROBOT_SYSTEM_PROMPT = """You are an AI assistant controlling an SO101 robotic arm with multiple cameras.

You can see live camera feeds from the robot and use tools to control it. When the user asks you to manipulate objects:

1. First observe what is currently visible in the cameras
2. If the target object is not visible, use the search_object tool to find it
3. Once the object is visible, use the grasp_object tool to pick it up
4. Report success or failure and describe what you observe

Always explain your reasoning before taking actions. Be helpful and descriptive about what you see in the camera feeds."""

# Tool definitions for robot control
ROBOT_TOOLS = [
    {
        "name": "search_object",
        "description": "Search for an object by moving the robot arm and cameras to find it. Use this when the target object is not currently visible in the camera feeds.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object_description": {
                    "type": "string",
                    "description": "Description of the object to search for (e.g., 'red cup', 'tennis ball', 'blue block')",
                }
            },
            "required": ["object_description"],
        },
    },
    {
        "name": "grasp_object",
        "description": "Grasp an object that is currently visible in the camera view. Use this after the object has been located and is visible.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object_description": {
                    "type": "string",
                    "description": "Description of the object to grasp",
                }
            },
            "required": ["object_description"],
        },
    },
    {
        "name": "describe_scene",
        "description": "Describe what is currently visible in all camera views. Use this to report observations to the user.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


class BaseLLMEngine(ABC):
    """
    Abstract LLM interface so the demo can work with Gemini, Claude, Qwen, etc.

    Implementations should expose a simple `chat` method that accepts:
    - messages: list of {"role": "user" | "assistant" | "system", "content": str}
    - optional tool / function-calling specs if you want structured calls
    - optional images for vision models

    For the hackathon you can start with plain text responses and add tools later.
    """

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class GeminiEngine(BaseLLMEngine):
    """
    Placeholder Gemini client using LLMConfig.
    """

    def __init__(self, cfg: Optional[LLMConfig] = None):
        cfg = cfg or LLMConfig()
        self.model_name = cfg.model_name
        self.api_key_env = cfg.api_key_env

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "GeminiEngine.chat is a stub. Implement the HTTP call to Google Gemini here."
        )


class ClaudeEngine(BaseLLMEngine):
    """
    Claude VLM client using Anthropic SDK.
    Supports vision (images in messages) and tool calling for agentic control.
    """

    def __init__(self, cfg: Optional[LLMConfig] = None):
        cfg = cfg or LLMConfig(
            provider="claude",
            model_name="claude-sonnet-4-20250514",
            api_key_env="ANTHROPIC_API_KEY",
        )
        self.model_name = cfg.model_name
        self.api_key_env = cfg.api_key_env
        self._client = None

    def _get_client(self):
        """Lazy-load the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise ValueError(
                    f"Missing API key. Set the {self.api_key_env} environment variable."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Send messages to Claude with optional images and tool definitions.

        Args:
            messages: List of message dicts with role and content
            tools: Optional list of tool definitions for function calling
            images: Optional dict of camera_name -> base64 encoded JPEG images

        Returns:
            Dict with role, content, and optional tool_calls
        """
        client = self._get_client()

        # Build message content with images if provided
        processed_messages = []
        for msg in messages:
            if msg["role"] == "user" and images:
                # Include images in user message
                content = []
                for cam_name, img_b64 in images.items():
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    })
                content.append({
                    "type": "text",
                    "text": f"[Camera feeds attached: {', '.join(images.keys())}]\n\n{msg.get('content', '')}",
                })
                processed_messages.append({"role": "user", "content": content})
            else:
                processed_messages.append(msg)

        # Build request kwargs
        kwargs = {
            "model": self.model_name,
            "max_tokens": 1024,
            "system": ROBOT_SYSTEM_PROMPT,
            "messages": processed_messages,
        }

        # Add tools if provided
        if tools:
            kwargs["tools"] = tools

        # Run synchronous API call in executor to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: client.messages.create(**kwargs)
            )
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return {
                "role": "assistant",
                "content": f"Error calling Claude API: {e}",
                "tool_calls": [],
            }

        # Parse response
        result = {
            "role": "assistant",
            "content": "",
            "tool_calls": [],
        }

        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                result["tool_calls"].append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return result


class QwenEngine(BaseLLMEngine):
    """
    Placeholder Qwen client using LLMConfig.
    """

    def __init__(self, cfg: Optional[LLMConfig] = None):
        cfg = cfg or LLMConfig(provider="qwen", model_name="qwen-vl", api_key_env="QWEN_API_KEY")
        self.model_name = cfg.model_name
        self.api_key_env = cfg.api_key_env

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "QwenEngine.chat is a stub. Implement the HTTP call to Qwen here."
        )


class StubEngine(BaseLLMEngine):
    """
    Local stub LLM used for debugging without any external API.

    It simply echoes the last user message with a short prefix.
    """

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
                break

        # In stub mode, mention how many images were received
        image_info = ""
        if images:
            image_info = f" (received {len(images)} camera images: {', '.join(images.keys())})"

        return {
            "role": "assistant",
            "content": f"[STUB LLM] I received: {last_user!r}{image_info}. Configure a real LLM (set provider='claude' and ANTHROPIC_API_KEY) to get meaningful answers.",
            "tool_calls": [],
        }


def make_llm_engine(cfg: Optional[LLMConfig] = None) -> BaseLLMEngine:
    """
    Factory to build an LLM engine based on LLMConfig.provider.

    provider:
      - "gemini"  -> GeminiEngine
      - "claude"  -> ClaudeEngine
      - "qwen"    -> QwenEngine
      - anything else -> StubEngine
    """
    cfg = cfg or LLMConfig()
    provider = cfg.provider.lower()
    if provider == "gemini":
        return GeminiEngine(cfg)
    if provider == "claude":
        return ClaudeEngine(cfg)
    if provider == "qwen":
        return QwenEngine(cfg)
    return StubEngine()
