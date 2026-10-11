"""Planners that are not LLMs. They exist to test the pipeline, not to be results.

``oracle``
    Returns the task's reference plan. Its success rate is the suite's
    **execution ceiling**: how often a correct plan succeeds in simulation.
    Every LLM number should be read against it.

``noisy:<p>``
    The reference plan, corrupted with probability ``p`` by one of the
    mistakes LLM planners are known to make (a hallucinated object id, a
    missing pick, the wrong object, an unreachable point, a false "infeasible",
    acting on an impossible instruction). Given verifier feedback it repairs
    itself with probability ``fix``. It reads the ground truth, so it is **synthetic**:
    useful for checking that the verifier, the repair loop and the statistics
    behave, meaningless as a finding about language models.
"""

from __future__ import annotations

import zlib

import numpy as np

from ..plan import Plan, Step, pick, place_at, place_on
from .base import Completion, Planner, PlannerSession, PlanningRequest


def _completion(plan: Plan, model: str) -> Completion:
    return Completion(text=plan.to_json(), model=model, stop_reason="end_turn")


class OraclePlanner(Planner):
    name = "oracle"

    def session(self, request: PlanningRequest, sample: int = 0) -> PlannerSession:
        if request.task is None:
            raise ValueError("the oracle planner needs the task")
        return _FixedSession(request.task.reference, self.name)


class _FixedSession(PlannerSession):
    def __init__(self, plan: Plan, model: str):
        self.plan = plan
        self.model = model

    def propose(self) -> Completion:
        return _completion(self.plan, self.model)

    def revise(self, feedback: str) -> Completion:
        return _completion(self.plan, self.model)


CORRUPTIONS = (
    "hallucinate_id",
    "drop_pick",
    "wrong_object",
    "unreachable_point",
    "claim_infeasible",
)


class NoisyPlanner(Planner):
    def __init__(self, p_error: float = 0.5, p_fix: float = 0.7):
        self.p_error = p_error
        self.p_fix = p_fix
        self.name = f"noisy:{p_error}"

    def describe(self) -> dict:
        return {"name": self.name, "synthetic": True, "p_error": self.p_error, "p_fix": self.p_fix}

    def session(self, request: PlanningRequest, sample: int = 0) -> PlannerSession:
        task = request.task
        if task is None:
            raise ValueError("the noisy planner needs the task")
        rng = np.random.default_rng([task.seed, zlib.crc32(task.template.encode()), sample, 7])
        if rng.random() >= self.p_error:
            return _FixedSession(task.reference, self.name)
        bad = corrupt(task, rng)
        return _NoisySession(bad, task.reference, rng, self.p_fix, self.name)


class _NoisySession(PlannerSession):
    def __init__(self, bad: Plan, good: Plan, rng: np.random.Generator, p_fix: float, model: str):
        self.current = bad
        self.good = good
        self.rng = rng
        self.p_fix = p_fix
        self.model = model

    def propose(self) -> Completion:
        return _completion(self.current, self.model)

    def revise(self, feedback: str) -> Completion:
        if self.rng.random() < self.p_fix:
            self.current = self.good
        return _completion(self.current, self.model)


def corrupt(task, rng: np.random.Generator) -> Plan:
    """One plausible LLM mistake applied to the reference plan."""
    scene = task.scene
    blocks = [o.id for o in scene.blocks]

    if not task.feasible:
        # The typical failure on an impossible request: do something anyway.
        far = max(scene.blocks, key=lambda o: o.x)
        guess = f"{task.instruction.split()[2]}_block"
        steps = {
            "missing_color": [pick(guess), place_on("bowl")],
            "unreachable_block": [pick(far.id), place_on("bowl")],
            "no_bowl": [pick(blocks[0]), place_on("bowl")],
            "move_bowl": [pick("bowl"), place_on(blocks[0])],
        }.get(task.template, [pick(blocks[0]), place_at(0.5, 0.0)])
        return Plan(feasible=True, reason="", steps=steps)

    steps = [s.model_copy() for s in task.reference.steps]
    kind = str(rng.choice(CORRUPTIONS))
    picks = [i for i, s in enumerate(steps) if s.action == "pick"]
    if kind == "hallucinate_id" and picks:
        i = int(rng.choice(picks))
        steps[i] = Step(action="pick", object=steps[i].object.replace("_block", "_cube"))
    elif kind == "drop_pick" and picks:
        del steps[int(rng.choice(picks))]
    elif kind == "wrong_object" and picks:
        i = int(rng.choice(picks))
        others = [b for b in blocks if b != steps[i].object]
        if others:
            steps[i] = pick(str(rng.choice(others)))
    elif kind == "unreachable_point":
        places = [i for i, s in enumerate(steps) if s.action != "pick"]
        if places:
            steps[int(rng.choice(places))] = place_at(0.95, float(rng.uniform(-0.3, 0.3)))
    else:
        return Plan(feasible=False, reason="I cannot find a way to do this.", steps=[])
    return Plan(feasible=True, reason="", steps=steps)
