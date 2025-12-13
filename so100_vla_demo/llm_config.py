from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LLMConfig:
    """
    Simple configuration holder for LLM provider and model.

    Load order:
    1. JSON file at `config_path` if it exists.
    2. Environment variables:
       - LLM_PROVIDER (e.g. "gemini", "claude")
       - LLM_MODEL (e.g. "gemini-1.5-flash" or "claude-3-5-sonnet")
       - LLM_API_KEY_ENV (e.g. "GEMINI_API_KEY", "ANTHROPIC_API_KEY")
    3. Hard-coded defaults.
    """

    provider: str = "gemini"
    model_name: str = "gemini-1.5-flash"
    api_key_env: str = "GEMINI_API_KEY"

    @classmethod
    def load(cls, config_path: str | os.PathLike[str] | None = None) -> "LLMConfig":
        # 1) Try JSON file
        if config_path is not None:
            path = Path(config_path)
            if path.is_file():
                data: dict[str, Any] = json.loads(path.read_text())
                return cls(
                    provider=data.get("provider", cls.provider),
                    model_name=data.get("model_name", cls.model_name),
                    api_key_env=data.get("api_key_env", cls.api_key_env),
                )

        # 2) Environment variables
        provider = os.environ.get("LLM_PROVIDER", cls.provider)
        model = os.environ.get("LLM_MODEL", cls.model_name)
        api_key_env = os.environ.get("LLM_API_KEY_ENV", cls.api_key_env)

        return cls(provider=provider, model_name=model, api_key_env=api_key_env)


