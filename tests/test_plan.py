import json

import pytest

from lang2act.plan import (
    PLAN_JSON_SCHEMA,
    Plan,
    PlanParseError,
    parse_plan,
    pick,
    place_at,
    place_on,
)


def test_round_trip():
    plan = Plan(
        steps=[pick("red_block"), place_on("bowl"), pick("blue_block"), place_at(0.5, -0.1)]
    )
    assert parse_plan(plan.to_json()) == plan
    assert (
        plan.describe()
        == "pick(red_block) -> place_on(bowl) -> pick(blue_block) -> place_at(0.500, -0.100)"
    )


def test_accepts_code_fences_and_surrounding_prose():
    text = 'Here is the plan:\n```json\n{"feasible": true, "reason": "", "steps": [{"action": "pick", "object": "a"}]}\n```'
    assert parse_plan(text).steps[0].object == "a"
    assert (
        parse_plan('Sure. {"feasible": false, "reason": "no", "steps": []} Done.').feasible is False
    )


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("no json here", "no JSON object"),
        ("{not json}", "malformed"),
        ('{"feasible": true, "steps": [{"action": "grab", "object": "a"}]}', "steps.0.action"),
        ('{"feasible": true, "steps": [], "extra": 1}', "extra"),
    ],
)
def test_parse_errors_are_readable(text, fragment):
    with pytest.raises(PlanParseError, match=fragment):
        parse_plan(text)


def test_schema_matches_the_model():
    """The JSON schema sent to providers must accept what the model produces."""
    step = PLAN_JSON_SCHEMA["properties"]["steps"]["items"]
    assert set(step["required"]) == {"action", "object", "target", "x", "y"}
    assert step["properties"]["action"]["enum"] == ["pick", "place_on", "place_at"]
    dumped = json.loads(Plan(steps=[pick("a")]).to_json())
    assert set(dumped) == set(PLAN_JSON_SCHEMA["required"])
