"""Run a plan in the simulator, checking each step's effect as it goes.

This is the "after" half of checking a plan. Every step has a postcondition
measured from the simulator: after ``pick(x)`` the gripper holds ``x``; after
``place_on(y)`` the block rests on ``y``; after ``place_at`` it rests on the
table where it was sent. The first step whose postcondition fails stops the
episode and is recorded, so a failure can be attributed to a specific step
rather than discovered only in the final goal check.

Steps that cannot even be attempted (an object that does not exist, placing
with an empty hand) are recorded as *invalid* rather than *physical* failures.
With the verifier on, they never get this far; with it off, this is where they
surface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from . import predicates as P
from .plan import Plan, Step
from .sim.scene import BLOCK_HALF, BOWL_FLOOR_THICKNESS
from .sim.skills import SkillConfig, Skills
from .sim.world import TabletopWorld

PLACE_AT_TOL = 0.03


@dataclass
class StepOutcome:
    index: int
    step: str
    ok: bool
    kind: str = ""  # "" | "invalid" | "physical"
    error: str = ""
    sim_steps: int = 0


@dataclass
class ExecutionResult:
    outcomes: list[StepOutcome] = field(default_factory=list)
    completed: bool = True
    failed_step: int | None = None
    error: str = ""
    error_kind: str = ""
    sim_steps: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class Executor:
    def __init__(
        self,
        world: TabletopWorld,
        skill_config: SkillConfig | None = None,
        on_step: Callable[[TabletopWorld], None] | None = None,
    ):
        self.world = world
        self.skills = Skills(world, skill_config, on_step)
        self.holding: str | None = None

    def run(self, plan: Plan) -> ExecutionResult:
        result = ExecutionResult()
        if not plan.feasible:
            return result
        self.holding = None
        for i, step in enumerate(plan.steps):
            self.world.status = f"step {i + 1}/{len(plan.steps)}: {step.describe()}"
            before = self.skills.total_steps
            ok, kind, error = self._run_step(step)
            n = self.skills.total_steps - before
            result.outcomes.append(StepOutcome(i, step.describe(), ok, kind, error, n))
            result.sim_steps += n
            if not ok:
                result.completed = False
                result.failed_step = i
                result.error = error
                result.error_kind = kind
                break
        self.world.status = "done" if result.completed else f"failed: {result.error}"
        return result

    # ------------------------------------------------------------------ steps

    def _invalid(self, msg: str) -> tuple[bool, str, str]:
        return False, "invalid", msg

    def _physical(self, msg: str) -> tuple[bool, str, str]:
        return False, "physical", msg

    def _run_step(self, step: Step) -> tuple[bool, str, str]:
        w = self.world
        if step.action == "pick":
            o = step.object
            if not o:
                return self._invalid("pick has no object")
            if not w.has(o):
                return self._invalid(f"there is no object named {o!r}")
            if w.kind(o) == "bowl":
                return self._invalid("the bowl is fixed to the table")
            if self.holding is not None:
                return self._invalid(f"already holding {self.holding}")
            res = self.skills.pick(o)
            if not res.ok:
                return self._physical(res.reason)
            self.holding = o
            return True, "", ""

        if self.holding is None:
            return self._invalid("not holding anything")
        held = self.holding

        if step.action == "place_on":
            t = step.target
            if not t:
                return self._invalid("place_on has no target")
            if not w.has(t):
                return self._invalid(f"there is no object named {t!r}")
            if t == held:
                return self._invalid(f"cannot place {held} on itself")
            if w.kind(t) == "bowl":
                states = w.object_states()
                occupied = [
                    states[b].pos[:2]
                    for b in w.block_names
                    if b != held and P.support_of(b, states) == "bowl"
                ]
                slot = P.free_bowl_slot(w.object_pos(t)[:2], occupied)
                if slot is None:
                    return self._physical("the bowl is full")
                res = self.skills.place(tuple(slot), BOWL_FLOOR_THICKNESS)
                if not res.ok:
                    return self._physical(res.reason)
                self.holding = None
                if not P.in_bowl(held, w.object_states()):
                    return self._physical(f"{held} did not end up in the bowl")
                return True, "", ""
            tpos = w.object_pos(t)
            res = self.skills.place((tpos[0], tpos[1]), tpos[2] + BLOCK_HALF)
            if not res.ok:
                return self._physical(res.reason)
            self.holding = None
            if not P.on(held, t, w.object_states()):
                return self._physical(f"{held} did not end up on {t}")
            return True, "", ""

        # place_at
        if step.x is None or step.y is None:
            return self._invalid("place_at needs x and y")
        target = np.array([step.x, step.y])
        res = self.skills.place((step.x, step.y), 0.0)
        if not res.ok:
            return self._physical(res.reason)
        self.holding = None
        states = w.object_states()
        err = float(np.linalg.norm(np.array(states[held].pos[:2]) - target))
        if not P.on_table(held, states) or err > PLACE_AT_TOL:
            return self._physical(
                f"{held} did not come to rest on the table at ({step.x:.3f}, {step.y:.3f}) "
                f"(off by {err:.3f} m)"
            )
        return True, "", ""
