from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .llm_config import LLMConfig


class BaseLLMEngine(ABC):
    """
    Abstract LLM interface so the demo can work with Gemini, Claude, Qwen, etc.

    Implementations should expose a simple `chat` method that accepts:
    - messages: list of {"role": "user" | "assistant" | "system", "content": str}
    - optional tool / function-calling specs if you want structured calls

    For the hackathon you can start with plain text responses and add tools later.
    """

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
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
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "GeminiEngine.chat is a stub. Implement the HTTP call to Google Gemini here."
        )


class ClaudeEngine(BaseLLMEngine):
    """
    Placeholder Claude client.
    """

    def __init__(self, cfg: Optional[LLMConfig] = None):
        cfg = cfg or LLMConfig(
            provider="claude",
            model_name="claude-3-5-sonnet",
            api_key_env="ANTHROPIC_API_KEY",
        )
        self.model_name = cfg.model_name
        self.api_key_env = cfg.api_key_env

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "ClaudeEngine.chat is a stub. Implement the HTTP call to Anthropic Claude here."
        )


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
    ) -> Dict[str, Any]:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
                break
        return {
            "role": "assistant",
            "content": f"[STUB LLM] I received: {last_user!r}. Configure a real LLM to get meaningful answers.",
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
