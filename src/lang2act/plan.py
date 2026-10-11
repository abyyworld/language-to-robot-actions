"""The plan format: the one contract between the LLM and everything else.

Three actions, deliberately few. ``pick`` and ``place_on`` name objects;
``place_at`` names table coordinates, which is how spatial instructions ("to the
left of the blue block") have to be expressed. Motion is not the LLM's job:
grasp poses, approach heights and bowl slots are chosen by the executor.

The step schema is flat (every field present, unused ones null) rather than a
union of three shapes. Flat schemas are accepted by every provider's
structured-output mode, and a step with the wrong fields filled in becomes a
verifier finding with a readable message instead of an opaque schema error.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["pick", "place_on", "place_at"]
    object: str | None = None
    target: str | None = None
    x: float | None = None
    y: float | None = None

    def describe(self) -> str:
        if self.action == "pick":
            return f"pick({self.object})"
        if self.action == "place_on":
            return f"place_on({self.target})"
        x = "?" if self.x is None else f"{self.x:.3f}"
        y = "?" if self.y is None else f"{self.y:.3f}"
        return f"place_at({x}, {y})"


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feasible: bool = True
    reason: str = ""
    steps: list[Step] = []

    def describe(self) -> str:
        if not self.feasible:
            return f"INFEASIBLE: {self.reason}"
        return " -> ".join(s.describe() for s in self.steps) or "(no steps)"

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), separators=(",", ":"))


def pick(obj: str) -> Step:
    return Step(action="pick", object=obj)


def place_on(target: str) -> Step:
    return Step(action="place_on", target=target)


def place_at(x: float, y: float) -> Step:
    return Step(action="place_at", x=round(float(x), 3), y=round(float(y), 3))


_NULLABLE_STR = {"type": ["string", "null"]}
_NULLABLE_NUM = {"type": ["number", "null"]}

PLAN_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "feasible": {"type": "boolean"},
        "reason": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["pick", "place_on", "place_at"]},
                    "object": _NULLABLE_STR,
                    "target": _NULLABLE_STR,
                    "x": _NULLABLE_NUM,
                    "y": _NULLABLE_NUM,
                },
                "required": ["action", "object", "target", "x", "y"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["feasible", "reason", "steps"],
    "additionalProperties": False,
}


class PlanParseError(ValueError):
    """The response was not a plan. The message is written to be shown to the LLM."""


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> str:
    m = _FENCE.search(text)
    if m:
        return m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise PlanParseError("the response contains no JSON object")
    return text[start : end + 1]


def parse_plan(text: str) -> Plan:
    """Parse an LLM response into a ``Plan``, or raise ``PlanParseError``."""
    try:
        data = json.loads(_extract_json(text))
    except json.JSONDecodeError as exc:
        raise PlanParseError(f"the JSON is malformed: {exc}") from exc
    if not isinstance(data, dict):
        raise PlanParseError("the top-level JSON value must be an object")
    try:
        return Plan.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in exc.errors()
        )
        raise PlanParseError(f"the JSON does not match the plan format: {problems}") from exc
