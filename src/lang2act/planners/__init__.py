"""Planner registry: build a planner from a spec string.

======================================  ===========================================
``anthropic:claude-haiku-5-5``          Claude via the Anthropic SDK (ANTHROPIC_API_KEY)
``ollama:qwen2.5:7b``                   a local model served by Ollama
``openai:<model>``                      an OpenAI-compatible server at $OPENAI_BASE_URL
``oracle``                              the reference plan (execution ceiling)
``noisy:0.5``                           synthetic, corrupted reference plans
======================================  ===========================================
"""

from __future__ import annotations

import os

from .base import Completion, Planner, PlannerSession, PlanningRequest
from .scripted import NoisyPlanner, OraclePlanner


def make_planner(
    spec: str,
    *,
    prompt: str = "basic",
    effort: str | None = "medium",
    structured: bool = True,
    cache_dir: str | None = ".cache/llm",
) -> Planner:
    provider, _, model = spec.partition(":")
    if provider == "oracle":
        return OraclePlanner()
    if provider == "noisy":
        return NoisyPlanner(p_error=float(model or 0.5))
    if provider == "anthropic":
        from .anthropic_planner import AnthropicPlanner

        return AnthropicPlanner(
            model or "claude-opus-5-5",
            prompt=prompt,
            effort=effort,
            structured=structured,
            cache_dir=cache_dir,
        )
    if provider in ("ollama", "openai"):
        from .openai_compat import OpenAICompatPlanner

        if not model:
            raise ValueError(f"{provider} needs a model name, e.g. {provider}:qwen2.5:7b")
        if provider == "ollama":
            base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            return OpenAICompatPlanner(
                model,
                base_url=base,
                provider="ollama",
                prompt=prompt,
                structured=structured,
                cache_dir=cache_dir,
            )
        return OpenAICompatPlanner(
            model,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key_env="OPENAI_API_KEY",
            provider="openai",
            prompt=prompt,
            structured=structured,
            cache_dir=cache_dir,
        )
    raise ValueError(f"unknown planner {spec!r}")


__all__ = [
    "Completion",
    "NoisyPlanner",
    "OraclePlanner",
    "Planner",
    "PlannerSession",
    "PlanningRequest",
    "make_planner",
]
