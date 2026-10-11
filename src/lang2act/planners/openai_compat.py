"""Any server that speaks the OpenAI chat-completions protocol.

That covers local models through Ollama, vLLM or llama.cpp, which is the free
way to compare models: ``ollama:qwen2.5:7b`` talks to
``http://localhost:11434/v1``. Local models take a temperature and a seed, so
these runs are seeded end to end.

Uses only the standard library, so it adds no dependency.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from ..plan import PLAN_JSON_SCHEMA
from .base import ChatPlanner, Completion


class OpenAICompatPlanner(ChatPlanner):
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434/v1",
        api_key_env: str | None = None,
        provider: str = "ollama",
        prompt: str = "basic",
        structured: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_s: float = 300.0,
        cache_dir=".cache/llm",
    ):
        super().__init__(prompt=prompt, cache_dir=cache_dir)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.provider = provider
        self.structured = structured
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.name = f"{provider}:{model}"

    def describe(self) -> dict:
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "prompt": self.prompt,
            "structured": self.structured,
            "temperature": self.temperature,
        }

    def request_payload(self, messages: list[dict], sample: int) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system}, *messages],
            "temperature": self.temperature,
            "seed": sample,
            "max_tokens": self.max_tokens,
        }
        if self.structured:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "plan", "schema": PLAN_JSON_SCHEMA, "strict": True},
            }
        return payload

    def send(self, payload: dict) -> tuple[Completion, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                return Completion(model=self.model, error=f"{self.api_key_env} is not set"), None
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            return Completion(model=self.model, error=f"http {exc.code}: {detail}"), None
        except (urllib.error.URLError, TimeoutError) as exc:
            return Completion(model=self.model, error=f"connection_error: {exc}"), None

        choice = body["choices"][0]
        text = choice["message"].get("content") or ""
        usage = body.get("usage") or {}
        completion = Completion(
            text=text,
            model=body.get("model", self.model),
            stop_reason=str(choice.get("finish_reason")),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )
        return completion, {"role": "assistant", "content": text}
