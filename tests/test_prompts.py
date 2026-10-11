"""The planner must be told every rule it is graded against, with the same numbers."""

from lang2act import predicates as P
from lang2act import verifier as V
from lang2act.planners.prompts import SYSTEM_PROMPTS, user_message, violation_feedback


def test_system_prompt_states_the_grading_thresholds():
    for prompt in SYSTEM_PROMPTS.values():
        for value in (
            P.SPATIAL_MIN,
            P.SPATIAL_MAX,
            P.SPATIAL_ALIGN,
            P.BETWEEN_TOL,
            P.MIN_CLEARANCE,
            V.MIN_BASE_DISTANCE,
            V.BOWL_KEEP_OUT,
            V.MAX_STACK,
        ):
            assert str(value) in prompt, f"{value} missing from the prompt"


def test_bowl_keep_out_is_consistent_with_the_bowl():
    from lang2act.sim.scene import BLOCK_HALF, BOWL_OUTER_RADIUS

    # A block placed at the keep-out distance must clear the outer wall.
    assert V.BOWL_KEEP_OUT >= BOWL_OUTER_RADIUS + BLOCK_HALF * 2**0.5


def test_fewshot_extends_basic():
    assert SYSTEM_PROMPTS["fewshot"].startswith(SYSTEM_PROMPTS["basic"])


def test_messages():
    msg = user_message({"objects": []}, "Do nothing.")
    assert msg.endswith("Instruction: Do nothing.")
    fb = violation_feedback(["Step 1 pick(x): there is no object 'x'."])
    assert "- Step 1 pick(x)" in fb and "corrected plan" in fb
