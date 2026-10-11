"""Check a plan before anything moves, and explain what is wrong with it.

The verifier knows the scene and the robot. It does **not** know the
instruction or the goal. It answers "can this plan be executed?", never "does
this plan do what was asked?". Giving it the goal would make it an oracle, and
the improvement it buys would measure nothing but the leak.

Two modes:

``symbolic``
    Steps the plan through an abstract model of the table (who holds what,
    what rests on what) and checks every precondition: the object exists, it
    is graspable, nothing is on top of it, the hand is free, the target is
    clear, the stack is not too tall, the bowl is not full, the placement is
    on the table, clear of other objects, and **kinematically reachable**. Reach
    is checked by solving the Panda's inverse kinematics for the grasp and
    approach poses, not by a distance rule.

``twin``
    ``symbolic``, then a dry run of the plan in the simulator from a snapshot,
    after which the world is restored exactly. It catches what the abstract
    model cannot (a placement that topples a stack, a grasp clipped by a
    neighbour). Because the twin *is* the simulator here, this is an upper
    bound on what a perfect digital twin could add, not a claim about a real
    robot.

Every finding is phrased to be read by the LLM, because it is: the repair loop
in ``episode.py`` sends the findings back and asks for a corrected plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import predicates as P
from .plan import Plan
from .sim.scene import BLOCK_HALF, BOWL_FLOOR_THICKNESS, BOWL_INNER_RADIUS
from .sim.skills import SkillConfig
from .sim.world import TabletopWorld

VERIFIER_MODES = ("none", "symbolic", "twin")

TABLE_X = (0.0, 0.88)
TABLE_Y = (-0.48, 0.48)
#: Closer than this to the base, the forearm clips tall stacks (measured).
MIN_BASE_DISTANCE = 0.33
MAX_STACK = 4
#: Blocks placed on the table must be at least this far from the bowl's centre.
BOWL_KEEP_OUT = 0.16


@dataclass
class Violation:
    step: int | None  # 0-based index; None for whole-plan findings
    code: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def format_violations(violations: list[Violation]) -> str:
    return "\n".join(f"- {v.message}" for v in violations)


class SymbolicVerifier:
    def __init__(self, world: TabletopWorld, skill_config: SkillConfig | None = None):
        self.world = world
        self.skill_config = skill_config or SkillConfig()
        self._reach_cache: dict[tuple[int, int, int], bool] = {}

    def _reachable(self, xyz: np.ndarray) -> bool:
        """Both the grasp pose and the approach pose above it must have IK solutions."""
        key = tuple(int(round(v * 1000)) for v in xyz)
        if key not in self._reach_cache:
            cfg = self.skill_config
            grasp = np.array([xyz[0], xyz[1], xyz[2] + cfg.grasp_z_offset])
            above = np.array([xyz[0], xyz[1], cfg.carry_height])
            ok = self.world.reachable(grasp)[0] and self.world.reachable(above)[0]
            self._reach_cache[key] = ok
        return self._reach_cache[key]

    def check(self, plan: Plan) -> list[Violation]:
        if not plan.feasible:
            return []
        w = self.world
        states = w.object_states()
        pos = {n: np.array(states[n].pos) for n in w.block_names}
        sup: dict[str, str | None] = {n: P.support_of(n, states) for n in w.block_names}
        bowl = w.bowl_name
        bowl_xy = np.array(w.object_pos(bowl)[:2]) if bowl else None
        names = ", ".join(w.block_names + ([bowl] if bowl else []))

        out: list[Violation] = []
        holding: str | None = None

        def level(b: str) -> int:
            n = 1
            while sup.get(b) not in (None, "table", "bowl"):
                b = sup[b]  # type: ignore[assignment]
                n += 1
            return n

        for i, step in enumerate(plan.steps):
            where = f"Step {i + 1} {step.describe()}"

            def bad(code: str, msg: str, i: int = i, where: str = where) -> None:
                out.append(Violation(i, code, f"{where}: {msg}"))

            if step.action == "pick":
                o = step.object
                if not o:
                    bad("malformed", "'pick' needs the id of an object in 'object'.")
                    continue
                if holding is not None:
                    bad("hand_full", f"the gripper is still holding {holding}. Place it first.")
                    continue
                if not w.has(o):
                    bad("unknown_object", f"there is no object '{o}'. Objects present: {names}.")
                    holding = o
                    continue
                if w.kind(o) == "bowl":
                    bad("not_graspable", "the bowl is fixed to the table and cannot be picked up.")
                    holding = o
                    continue
                on_top = [b for b, s in sup.items() if s == o]
                if on_top:
                    bad("not_clear", f"{', '.join(on_top)} is on top of {o}. Move it off first.")
                elif not self._reachable(pos[o]):
                    bad(
                        "unreachable",
                        f"{o} at ({pos[o][0]:.3f}, {pos[o][1]:.3f}) is outside the arm's reach.",
                    )
                holding = o
                sup[o] = None
                continue

            if holding is None:
                bad("hand_empty", "the gripper is not holding anything. Pick an object first.")
                continue
            held = holding
            holding = None
            tracked = held in pos

            if step.action == "place_on":
                t = step.target
                if not t:
                    bad("malformed", "'place_on' needs the id of an object in 'target'.")
                    continue
                if not w.has(t):
                    bad("unknown_object", f"there is no object '{t}'. Objects present: {names}.")
                    continue
                if t == held:
                    bad("self_target", f"cannot place {held} on itself.")
                    continue
                if w.kind(t) == "bowl":
                    in_bowl = [b for b, s in sup.items() if s == "bowl" and b != held]
                    slot = P.free_bowl_slot(bowl_xy, [pos[b][:2] for b in in_bowl])
                    if slot is None:
                        bad("bowl_full", f"the bowl already holds {P.BOWL_CAPACITY} blocks.")
                        continue
                    if tracked:
                        pos[held] = np.array([slot[0], slot[1], BOWL_FLOOR_THICKNESS + BLOCK_HALF])
                        sup[held] = "bowl"
                    continue
                if sup.get(t) is None and t in sup:
                    bad("target_in_hand", f"{t} is not on the table.")
                    continue
                on_top = [b for b, s in sup.items() if s == t]
                if on_top:
                    bad("not_clear", f"{', '.join(on_top)} is already on top of {t}.")
                    continue
                if level(t) + 1 > MAX_STACK:
                    bad("stack_too_high", f"stacks may be at most {MAX_STACK} blocks tall.")
                    continue
                if not self._reachable(pos[t] + np.array([0, 0, 2 * BLOCK_HALF])):
                    bad("unreachable", f"the top of {t} is outside the arm's reach.")
                    continue
                if tracked:
                    pos[held] = pos[t] + np.array([0.0, 0.0, 2 * BLOCK_HALF])
                    sup[held] = t
                continue

            # place_at
            if step.x is None or step.y is None:
                bad("malformed", "'place_at' needs numeric 'x' and 'y'.")
                continue
            xy = np.array([step.x, step.y])
            if not (TABLE_X[0] <= xy[0] <= TABLE_X[1] and TABLE_Y[0] <= xy[1] <= TABLE_Y[1]):
                bad("off_table", f"({xy[0]:.3f}, {xy[1]:.3f}) is not on the table.")
                continue
            if np.hypot(*xy) < MIN_BASE_DISTANCE:
                bad(
                    "unreachable",
                    f"({xy[0]:.3f}, {xy[1]:.3f}) is too close to the robot base "
                    f"(it must be at least {MIN_BASE_DISTANCE} m away).",
                )
                continue
            if not self._reachable(np.array([xy[0], xy[1], BLOCK_HALF])):
                bad("unreachable", f"({xy[0]:.3f}, {xy[1]:.3f}) is outside the arm's reach.")
                continue
            if bowl_xy is not None:
                d = float(np.linalg.norm(xy - bowl_xy))
                if d < BOWL_INNER_RADIUS:
                    bad(
                        "use_place_on",
                        "that point is inside the bowl. Use place_on with target 'bowl'.",
                    )
                    continue
                if d < BOWL_KEEP_OUT:
                    bad(
                        "collision",
                        f"that point is on or against the bowl wall ({d:.3f} m from its centre).",
                    )
                    continue
            clash = [
                (b, float(np.linalg.norm(xy - pos[b][:2])))
                for b, s in sup.items()
                if b != held and s == "table" and np.linalg.norm(xy - pos[b][:2]) < P.MIN_CLEARANCE
            ]
            if clash:
                b, d = min(clash, key=lambda c: c[1])
                bad(
                    "collision",
                    f"that point is {d:.3f} m from {b}; keep at least {P.MIN_CLEARANCE} m "
                    "between block centres.",
                )
                continue
            if tracked:
                pos[held] = np.array([xy[0], xy[1], BLOCK_HALF])
                sup[held] = "table"

        if holding is not None:
            out.append(
                Violation(
                    None,
                    "ends_holding",
                    f"The plan ends while still holding {holding}. Place it somewhere.",
                )
            )
        return out


class Verifier:
    """``none`` / ``symbolic`` / ``twin``; see the module docstring."""

    def __init__(
        self, world: TabletopWorld, mode: str = "symbolic", skill_config: SkillConfig | None = None
    ):
        if mode not in VERIFIER_MODES:
            raise ValueError(f"verifier mode must be one of {VERIFIER_MODES}, got {mode!r}")
        self.world = world
        self.mode = mode
        self.skill_config = skill_config
        self.symbolic = SymbolicVerifier(world, skill_config)

    def check(self, plan: Plan) -> list[Violation]:
        if self.mode == "none":
            return []
        found = self.symbolic.check(plan)
        if found or self.mode == "symbolic" or not plan.feasible:
            return found
        return self.dry_run(plan)

    def dry_run(self, plan: Plan) -> list[Violation]:
        from .executor import Executor

        snapshot = self.world.get_state()
        try:
            result = Executor(self.world, self.skill_config).run(plan)
        finally:
            self.world.set_state(snapshot)
        if result.completed:
            return []
        i = result.failed_step
        assert i is not None
        return [
            Violation(
                i,
                "sim_failure",
                f"Step {i + 1} {plan.steps[i].describe()}: in a simulated dry run this step "
                f"failed: {result.error}.",
            )
        ]
