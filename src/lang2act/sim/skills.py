"""Pick and place motion primitives.

Adapted from ``icm.control.scripted.ScriptedExpert`` in
abyyworld/Interactive-correction-for-manipulation (see VENDORED.md). The
P-controller, the gains, the stage timeouts and the yaw folding are unchanged;
that expert scored 100/100 on its pick-and-place task. What changed is the
structure: one state machine for a fixed pick-then-place-on-pad becomes two
primitives, ``pick(block)`` and ``place(xy, support_height)``, that an LLM plan
can sequence in any order.

The primitives use ground-truth object poses. That is deliberate: the
experiment is about whether the *plan* is right, so motion has to be as close
to perfect as possible. Anything it fails at on its own is noise in the
measurement, which is why the suite's oracle run exists (see README).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .scene import BLOCK_HALF
from .world import TabletopWorld


@dataclass
class SkillConfig:
    hover_height: float = 0.10  # above the block before descending
    grasp_z_offset: float = 0.002  # TCP height relative to block centre when grasping
    carry_height: float = 0.22  # TCP altitude while transporting; clears a 4-block stack
    place_clearance: float = 0.012  # block bottom above its support before release
    # Gain on the target-pose update: target += kp*(waypoint - tcp). A per-step
    # convergence rate, not a stiffness: above ~1.5 it overshoots every step.
    pos_gain: float = 0.8
    yaw_gain: float = 0.8
    pos_tol: float = 0.006
    xy_tol: float = 0.004  # tighter tolerance before committing to a descent
    yaw_tol: float = 0.08
    close_steps: int = 12
    release_steps: int = 10
    stage_timeout: int = 90  # per-stage step budget; prevents silent hangs
    #: Open only part-way to release. Fully open fingers span 10 cm, enough to
    #: clip the bowl wall or a block in the next bowl slot.
    release_opening: float = 0.70
    #: Speed cap (fraction of the per-step maximum) for the final descent with
    #: a block. Full-speed descents swing the block by up to 2 cm.
    lower_speed: float = 0.35
    settle_steps: int = 4  # hold still over the target before releasing
    descend_speed: float = 0.5  # speed cap descending onto a block to grasp it


@dataclass
class SkillResult:
    ok: bool
    reason: str = ""
    steps: int = 0


def wrap_to_quarter(angle: float) -> float:
    """Fold an angle into [-pi/4, pi/4]: a cube presents the same grasp every 90 degrees."""
    return float((angle + np.pi / 4) % (np.pi / 2) - np.pi / 4)


class Skills:
    """Waypoint-following pick and place on a ``TabletopWorld``."""

    def __init__(
        self,
        world: TabletopWorld,
        config: SkillConfig | None = None,
        on_step: Callable[[TabletopWorld], None] | None = None,
    ):
        self.world = world
        self.config = config or SkillConfig()
        self.on_step = on_step
        self.total_steps = 0

    # ------------------------------------------------------------------ low level

    def _action(
        self, waypoint: np.ndarray, yaw_target: float, gripper: float, speed: float = 1.0
    ) -> np.ndarray:
        cfg = self.config
        w = self.world
        err = waypoint - w.robot.tcp_pos
        action = np.zeros(7)
        action[:3] = np.clip(cfg.pos_gain * err / w.config.max_delta_pos, -speed, speed)
        yaw_err = wrap_to_quarter(yaw_target - w.yaw)
        action[5] = float(np.clip(cfg.yaw_gain * yaw_err / w.config.max_delta_rot, -1.0, 1.0))
        action[6] = gripper
        return action

    def _step(self, action: np.ndarray) -> None:
        self.world.step(action)
        self.total_steps += 1
        if self.on_step is not None:
            self.on_step(self.world)

    def _move(
        self,
        waypoint: np.ndarray,
        yaw: float,
        gripper: float,
        arrived: Callable[[np.ndarray], bool],
        speed: float = 1.0,
    ) -> bool:
        for _ in range(self.config.stage_timeout):
            self._step(self._action(waypoint, yaw, gripper, speed))
            if arrived(self.world.robot.tcp_pos):
                return True
        return False

    def _dwell(self, waypoint: np.ndarray, yaw: float, gripper: float, steps: int) -> None:
        for _ in range(steps):
            self._step(self._action(waypoint, yaw, gripper))

    @staticmethod
    def _gripper_action(opening: float) -> float:
        return 2.0 * opening - 1.0

    def _out_of_reach(self, *poses: np.ndarray, yaw: float) -> bool:
        """Refuse goals with no IK solution before moving, as a motion planner would.

        Without this the arm strains toward the goal until every stage times
        out: the same failure, forty seconds later.
        """
        return not all(self.world.reachable(p, yaw)[0] for p in poses)

    # ------------------------------------------------------------------ primitives

    def pick(self, name: str) -> SkillResult:
        cfg = self.config
        w = self.world
        start = self.total_steps
        obj = w.object_pos(name)
        yaw = wrap_to_quarter(-w.object_yaw(name))
        grasp = obj + np.array([0.0, 0.0, cfg.grasp_z_offset])
        open_ = self._gripper_action(1.0)
        closed = self._gripper_action(0.0)

        hover = np.array([grasp[0], grasp[1], max(obj[2] + cfg.hover_height, cfg.carry_height)])
        if self._out_of_reach(hover, grasp, yaw=yaw):
            return SkillResult(False, f"{name} is out of reach (no inverse-kinematics solution)")
        self._move(
            hover,
            yaw,
            open_,
            lambda tcp: (
                np.linalg.norm(tcp[:2] - hover[:2]) < cfg.xy_tol
                and abs(tcp[2] - hover[2]) < cfg.pos_tol
            ),
        )
        self._move(
            grasp,
            yaw,
            open_,
            lambda tcp: np.linalg.norm(tcp - grasp) < cfg.pos_tol,
            speed=cfg.descend_speed,
        )
        self._dwell(grasp, yaw, closed, cfg.close_steps)
        lift = np.array([grasp[0], grasp[1], cfg.carry_height])
        self._move(lift, yaw, closed, lambda tcp: tcp[2] > cfg.carry_height - cfg.pos_tol)

        steps = self.total_steps - start
        if not w.is_held(name):
            return SkillResult(False, f"grasp failed: {name} is not in the gripper", steps)
        if w.object_pos(name)[2] < obj[2] + 0.05:
            return SkillResult(False, f"{name} did not lift", steps)
        return SkillResult(True, "", steps)

    def place(self, xy: tuple[float, float], support_top: float) -> SkillResult:
        """Carry the held block so that *its centre* lands on ``xy``, resting on ``support_top``."""
        cfg = self.config
        w = self.world
        start = self.total_steps
        yaw = w.yaw
        closed = self._gripper_action(0.0)
        release = self._gripper_action(cfg.release_opening)

        # The block rarely sits dead centre between the fingers; aim the block,
        # not the TCP, or every placement inherits the grasp offset.
        held = w.held_object()
        offset = np.zeros(2)
        if held is not None:
            offset = (w.object_pos(held) - w.robot.tcp_pos)[:2]
            if np.linalg.norm(offset) > 0.02:  # implausible; the measurement is wrong
                offset = np.zeros(2)
        goal_xy = np.asarray(xy, dtype=float) - offset

        above = np.array([goal_xy[0], goal_xy[1], cfg.carry_height])
        lower_z = support_top + cfg.place_clearance + BLOCK_HALF + cfg.grasp_z_offset
        lower = np.array([goal_xy[0], goal_xy[1], lower_z])
        if self._out_of_reach(above, lower, yaw=yaw):
            return SkillResult(
                False,
                f"({xy[0]:.3f}, {xy[1]:.3f}) is out of reach (no inverse-kinematics solution)",
            )
        self._move(
            above,
            yaw,
            closed,
            lambda tcp: (
                np.linalg.norm(tcp[:2] - above[:2]) < cfg.xy_tol
                and abs(tcp[2] - above[2]) < cfg.pos_tol
            ),
        )
        self._move(
            lower, yaw, closed, lambda tcp: tcp[2] < lower_z + cfg.pos_tol, speed=cfg.lower_speed
        )
        self._dwell(lower, yaw, closed, cfg.settle_steps)
        tcp = w.robot.tcp_pos
        self._dwell(tcp, yaw, release, cfg.release_steps)
        retreat = np.array([tcp[0], tcp[1], cfg.carry_height])
        self._move(retreat, yaw, release, lambda t: t[2] > cfg.carry_height - 0.02)
        return SkillResult(True, "", self.total_steps - start)
