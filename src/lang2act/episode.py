"""One episode: instruction in, plan, verify, repair, execute, check the goal.

Every episode ends with exactly one ``stage`` explaining where it ended. That
breakdown is the second half of the analysis: a success rate says *how often*
a planner fails, the stage says *where*.

=====================  ====================================================
``success``            the goal holds (or, for an infeasible task, the
                       robot correctly did nothing)
``api_error``          the provider call failed; excluded from comparisons
``refusal``            the model declined to answer
``parse_error``        the response was not a valid plan
``verifier_rejected``  the plan still broke the rules after every repair
``declared_infeasible`` the planner said a feasible task was impossible
``execution_invalid``  a step could not be attempted (verifier off only)
``execution_physical`` a step was attempted and did not have its effect
``goal_not_met``       every step worked; the result is not what was asked
``acted_on_infeasible`` the task was impossible and the robot moved things
``wrong_response``     the task was impossible; the planner did not say so,
                       but nothing moved either (e.g. an empty plan)
=====================  ====================================================
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from .executor import Executor
from .plan import Plan, PlanParseError, parse_plan
from .planners.base import Planner, PlanningRequest
from .planners.prompts import parse_feedback, violation_feedback
from .scenes import describe
from .sim.world import TabletopWorld
from .tasks import Task
from .verifier import VERIFIER_MODES, Verifier

STAGES = (
    "success",
    "api_error",
    "refusal",
    "parse_error",
    "verifier_rejected",
    "declared_infeasible",
    "execution_invalid",
    "execution_physical",
    "goal_not_met",
    "acted_on_infeasible",
    "wrong_response",
)


@dataclass(frozen=True)
class Condition:
    """Everything about how a run plans. Two runs are comparable when only one field differs."""

    planner: str
    verifier: str = "symbolic"
    max_repairs: int = 2
    prompt: str = "basic"
    effort: str | None = "medium"
    structured: bool = True
    sample: int = 0

    def __post_init__(self):
        if self.verifier not in VERIFIER_MODES:
            raise ValueError(f"verifier must be one of {VERIFIER_MODES}")

    def to_dict(self) -> dict:
        return asdict(self)


def run_episode(
    task: Task,
    planner: Planner,
    condition: Condition,
    on_step: Callable[[TabletopWorld], None] | None = None,
    world: TabletopWorld | None = None,
) -> dict:
    t_start = time.perf_counter()
    if world is None:
        world = TabletopWorld(task.scene.to_sim_objects())
        world.reset()
    s0 = world.object_states()
    scene = describe(world)
    verifier = Verifier(world, condition.verifier)
    session = planner.session(
        PlanningRequest(task.instruction, scene, task), sample=condition.sample
    )

    rounds: list[dict] = []
    plan: Plan | None = None
    violations = []
    stage: str | None = None
    tokens_in = tokens_out = 0
    llm_seconds = 0.0

    completion = session.propose()
    for r in range(condition.max_repairs + 1):
        tokens_in += completion.input_tokens
        tokens_out += completion.output_tokens
        llm_seconds += completion.latency_s
        entry: dict = {"round": r, "completion": completion.summary()}
        rounds.append(entry)
        if completion.error:
            stage = "refusal" if completion.refusal else "api_error"
            break

        parse_error = None
        try:
            plan = parse_plan(completion.text)
            entry["plan"] = plan.model_dump()
            entry["plan_text"] = plan.describe()
        except PlanParseError as exc:
            plan = None
            parse_error = str(exc)
            entry["parse_error"] = parse_error

        violations = verifier.check(plan) if plan is not None else []
        entry["violations"] = [v.to_dict() for v in violations]

        if condition.verifier == "none" or (plan is not None and not violations):
            break
        if r == condition.max_repairs:
            break
        feedback = (
            parse_feedback(parse_error)
            if plan is None
            else violation_feedback([v.message for v in violations])
        )
        entry["feedback"] = feedback
        completion = session.revise(feedback)

    execution = None
    declared_infeasible = bool(plan is not None and not plan.feasible)
    if stage is None:
        if plan is None:
            stage = "parse_error"
        elif violations and condition.verifier != "none":
            stage = "verifier_rejected"
        elif plan.feasible:
            result = Executor(world, on_step=on_step).run(plan)
            execution = result.to_dict()
            if not result.completed:
                stage = f"execution_{result.error_kind}"

    s1 = world.object_states()
    goal_ok, goal_detail = task.goal(s0, s1)

    if task.feasible:
        if stage is None:
            stage = (
                "declared_infeasible"
                if declared_infeasible
                else ("success" if goal_ok else "goal_not_met")
            )
        success = stage == "success"
    else:
        # Correct behaviour on an impossible instruction: say so (or be stopped
        # by the verifier) and leave the table alone.
        if stage in ("api_error", "refusal"):
            success = False
        elif not goal_ok:
            stage = "acted_on_infeasible"
            success = False
        elif declared_infeasible or stage == "verifier_rejected":
            stage = "success"
            success = True
        else:
            # Tried to act, but nothing moved (an unparseable answer, a first
            # step that could not be attempted, an empty "feasible" plan).
            stage = stage or "wrong_response"
            success = False

    first_codes = sorted({v["code"] for v in rounds[0].get("violations", [])}) if rounds else []
    return {
        "template": task.template,
        "category": task.category,
        "seed": task.seed,
        "feasible": task.feasible,
        "instruction": task.instruction,
        "scene": scene,
        "condition": condition.to_dict(),
        "planner": planner.name,
        "success": bool(success),
        "stage": stage,
        "goal_detail": goal_detail,
        "declared_infeasible": declared_infeasible,
        "n_rounds": len(rounds),
        "first_plan_valid": bool(
            rounds and "plan" in rounds[0] and not rounds[0].get("violations")
        ),
        "first_violation_codes": first_codes,
        "repaired": bool(len(rounds) > 1 and plan is not None and not violations),
        "final_plan": plan.model_dump() if plan is not None else None,
        "final_plan_text": plan.describe() if plan is not None else None,
        "rounds": rounds,
        "execution": execution,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "llm_seconds": round(llm_seconds, 3),
        "wall_seconds": round(time.perf_counter() - t_start, 3),
    }
