"""Planner interface.

A planner opens a *session* per episode. ``propose`` returns the first plan;
``revise`` sends the verifier's findings back and returns a corrected one. LLM
planners keep the conversation in the session so a revision sees its own
earlier answer.

Every response is returned as raw text and parsed by the caller. Parsing is
part of what is being evaluated: a model that cannot follow the format fails,
and the failure is recorded as such.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .prompts import system_prompt, user_message

if TYPE_CHECKING:
    from ..tasks import Task


@dataclass
class PlanningRequest:
    instruction: str
    scene: dict
    #: The ground-truth task. Only scripted planners may read it; LLM planners
    #: see ``instruction`` and ``scene`` and nothing else.
    task: Task | None = None


@dataclass
class Completion:
    text: str = ""
    model: str = ""
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    error: str | None = None
    refusal: bool = False

    def summary(self) -> dict:
        return asdict(self)


class PlannerSession(ABC):
    @abstractmethod
    def propose(self) -> Completion: ...

    @abstractmethod
    def revise(self, feedback: str) -> Completion: ...


class Planner(ABC):
    name: str = "planner"

    @abstractmethod
    def session(self, request: PlanningRequest, sample: int = 0) -> PlannerSession: ...

    def describe(self) -> dict:
        return {"name": self.name}


# ---------------------------------------------------------------------------
# Chat-model planners


class LLMCache:
    """Responses on disk, keyed by the full request.

    Makes reruns free and exactly reproducible. Claude's current models take no
    temperature or seed, so the only way to replay a run is to keep what it
    returned. Delete the directory to resample.
    """

    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        if not self.root:
            return None
        path = self.root / key[:2] / f"{key}.json"
        if path.is_file():
            return json.loads(path.read_text())
        return None

    def put(self, key: str, value: dict) -> None:
        if not self.root:
            return
        path = self.root / key[:2] / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value))
        tmp.replace(path)


class ChatPlanner(Planner):
    """Shared session logic for chat-completion planners."""

    def __init__(self, prompt: str = "basic", cache_dir: str | Path | None = ".cache/llm"):
        self.prompt = prompt
        self.system = system_prompt(prompt)
        self.cache = LLMCache(cache_dir)

    @abstractmethod
    def request_payload(self, messages: list[dict], sample: int) -> dict:
        """The provider request, minus credentials. Also the cache key."""

    @abstractmethod
    def send(self, payload: dict) -> tuple[Completion, Any]:
        """Call the provider. Returns the completion and the assistant message to append."""

    def complete(self, messages: list[dict], sample: int) -> tuple[Completion, Any]:
        payload = self.request_payload(messages, sample)
        key = LLMCache.key({"planner": self.name, "payload": payload, "sample": sample})
        hit = self.cache.get(key)
        if hit is not None:
            c = Completion(**hit["completion"])
            c.cached = True
            return c, hit["assistant"]
        start = time.perf_counter()
        completion, assistant = self.send(payload)
        completion.latency_s = time.perf_counter() - start
        if completion.error is None:
            self.cache.put(key, {"completion": asdict(completion), "assistant": assistant})
        return completion, assistant

    def session(self, request: PlanningRequest, sample: int = 0) -> PlannerSession:
        return ChatSession(self, user_message(request.scene, request.instruction), sample)

    def user_turn(self, text: str) -> dict:
        return {"role": "user", "content": text}


class ChatSession(PlannerSession):
    def __init__(self, planner: ChatPlanner, first_message: str, sample: int):
        self.planner = planner
        self.sample = sample
        self.messages: list[dict] = [planner.user_turn(first_message)]

    def _turn(self) -> Completion:
        completion, assistant = self.planner.complete(self.messages, self.sample)
        if assistant is not None:
            self.messages.append(assistant)
        return completion

    def propose(self) -> Completion:
        return self._turn()

    def revise(self, feedback: str) -> Completion:
        self.messages.append(self.planner.user_turn(feedback))
        return self._turn()


@dataclass
class PlannerSpec:
    """Parsed form of a planner string such as ``anthropic:claude-haiku-5-5``."""

    provider: str
    model: str = ""
    options: dict = field(default_factory=dict)
