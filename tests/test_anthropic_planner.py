"""The Claude planner, against a fake client: no network, no key."""

from types import SimpleNamespace

from lang2act.plan import parse_plan
from lang2act.planners.anthropic_planner import AnthropicPlanner, _nullable
from lang2act.planners.base import PlanningRequest

PLAN = '{"feasible": true, "reason": "", "steps": [{"action": "pick", "object": "red_block", "target": null, "x": null, "y": null}, {"action": "place_on", "object": null, "target": "bowl", "x": null, "y": null}]}'


class FakeMessages:
    def __init__(self, stop_reason="end_turn"):
        self.calls = []
        self.stop_reason = stop_reason

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="", signature="sig"),
                SimpleNamespace(type="text", text=PLAN),
            ],
            model=kwargs["model"],
            stop_reason=self.stop_reason,
            stop_details=SimpleNamespace(category="other")
            if self.stop_reason == "refusal"
            else None,
            usage=SimpleNamespace(input_tokens=100, output_tokens=40),
        )


def planner(tmp_path, stop_reason="end_turn"):
    fake = SimpleNamespace(messages=FakeMessages(stop_reason))
    return AnthropicPlanner("claude-haiku-5-5", client=fake, cache_dir=tmp_path), fake.messages


def test_request_shape(tmp_path):
    p, msgs = planner(tmp_path)
    session = p.session(PlanningRequest("Put the red block in the bowl.", {"objects": []}))
    c = session.propose()
    assert parse_plan(c.text).steps[0].object == "red_block"
    call = msgs.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["output_config"]["effort"] == "medium"
    assert call["output_config"]["format"]["type"] == "json_schema"
    # Current Claude models reject sampling parameters; none may be sent.
    assert "temperature" not in call and "top_p" not in call
    assert call["messages"][0]["content"].endswith("Instruction: Put the red block in the bowl.")


def test_revision_continues_the_conversation_with_thinking_intact(tmp_path):
    p, msgs = planner(tmp_path)
    session = p.session(PlanningRequest("x", {"objects": []}))
    session.propose()
    session.revise("Your plan was rejected: ...")
    second = msgs.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig"}


def test_responses_are_cached(tmp_path):
    p, msgs = planner(tmp_path)
    p.session(PlanningRequest("x", {"objects": []})).propose()
    c = p.session(PlanningRequest("x", {"objects": []})).propose()
    assert c.cached and len(msgs.calls) == 1
    # A different sample index is a different request.
    p.session(PlanningRequest("x", {"objects": []}), sample=1).propose()
    assert len(msgs.calls) == 2


def test_refusal_is_recorded_not_retried_elsewhere(tmp_path):
    p, msgs = planner(tmp_path, stop_reason="refusal")
    c = p.session(PlanningRequest("x", {"objects": []})).propose()
    assert c.refusal and c.error.startswith("refusal")
    assert "fallbacks" not in msgs.calls[0]


def test_nullable_schema_rewrite():
    out = _nullable({"type": "object", "properties": {"a": {"type": ["string", "null"]}}})
    assert out["properties"]["a"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
