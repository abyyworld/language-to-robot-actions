"""Claude as the planner, through the Anthropic SDK.

Choices that affect what the numbers mean:

* **No refusal fallbacks.** The API can re-run a refused request on another
  model. That is useful in production and wrong here: the row would say one
  model and the plan would come from another. A refusal is recorded as one.
* **Effort is set explicitly** (default ``medium``) because the models'
  defaults differ, and comparing two models at different effort levels would
  confound the comparison. It is recorded in every run.
* **Structured outputs on by default.** The response is constrained to the
  plan schema. Turn it off (``--no-structured``) to measure format-following
  as part of the task.
* **The served model is recorded** from the response, not assumed from the
  request.
"""

from __future__ import annotations

from typing import Any

from ..plan import PLAN_JSON_SCHEMA
from .base import ChatPlanner, Completion


def _nullable(schema: dict) -> dict:
    """Rewrite ``{"type": [T, "null"]}`` as ``anyOf``, the form structured outputs documents."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            out[k] = _nullable(v) if isinstance(v, dict | list) else v
        t = out.get("type")
        if isinstance(t, list) and "null" in t and len(t) == 2:
            base = next(x for x in t if x != "null")
            out.pop("type")
            return {"anyOf": [{"type": base, **out}, {"type": "null"}]}
        return out
    if isinstance(schema, list):
        return [_nullable(v) for v in schema]
    return schema


class AnthropicPlanner(ChatPlanner):
    def __init__(
        self,
        model: str = "claude-opus-5-5",
        *,
        prompt: str = "basic",
        effort: str | None = "medium",
        structured: bool = True,
        max_tokens: int = 16000,
        cache_dir=".cache/llm",
        client: Any = None,
    ):
        super().__init__(prompt=prompt, cache_dir=cache_dir)
        self.model = model
        self.effort = effort
        self.structured = structured
        self.max_tokens = max_tokens
        self._client = client
        self.name = f"anthropic:{model}"

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def describe(self) -> dict:
        return {
            "name": self.name,
            "provider": "anthropic",
            "model": self.model,
            "prompt": self.prompt,
            "effort": self.effort,
            "structured": self.structured,
            "max_tokens": self.max_tokens,
        }

    def request_payload(self, messages: list[dict], sample: int) -> dict:
        payload: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system,
            # A copy: the session appends to its list after this request is made.
            "messages": list(messages),
        }
        output_config: dict = {}
        if self.effort:
            output_config["effort"] = self.effort
        if self.structured:
            output_config["format"] = {"type": "json_schema", "schema": _nullable(PLAN_JSON_SCHEMA)}
        if output_config:
            payload["output_config"] = output_config
        return payload

    def send(self, payload: dict) -> tuple[Completion, Any]:
        import anthropic

        try:
            response = self.client.messages.create(**payload)
        except anthropic.BadRequestError as exc:
            return Completion(model=self.model, error=f"bad_request: {exc.message}"), None
        except anthropic.AuthenticationError:
            return Completion(
                model=self.model, error="authentication failed: set ANTHROPIC_API_KEY"
            ), None
        except anthropic.NotFoundError as exc:
            return Completion(model=self.model, error=f"not_found: {exc.message}"), None
        except anthropic.RateLimitError as exc:
            return Completion(model=self.model, error=f"rate_limited: {exc.message}"), None
        except anthropic.APIStatusError as exc:
            return Completion(
                model=self.model, error=f"api_error {exc.status_code}: {exc.message}"
            ), None
        except anthropic.APIConnectionError as exc:
            return Completion(model=self.model, error=f"connection_error: {exc}"), None

        text = "".join(b.text for b in response.content if b.type == "text")
        completion = Completion(
            text=text,
            model=response.model,
            stop_reason=str(response.stop_reason),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        if response.stop_reason == "refusal":
            completion.refusal = True
            category = getattr(getattr(response, "stop_details", None), "category", None)
            completion.error = f"refusal ({category})"
            return completion, None

        # Append the assistant turn exactly as returned (thinking blocks
        # included, unmodified), so a revision continues the same conversation.
        blocks = []
        for b in response.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "thinking":
                blocks.append(
                    {"type": "thinking", "thinking": b.thinking, "signature": b.signature}
                )
            elif b.type == "redacted_thinking":
                blocks.append({"type": "redacted_thinking", "data": b.data})
        return completion, {"role": "assistant", "content": blocks}
