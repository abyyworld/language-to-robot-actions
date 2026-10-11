"""The tabletop world: a Panda, some blocks, maybe a bowl.

Adapted from ``icm.envs.pick_place.PickPlaceEnv`` in
abyyworld/Interactive-correction-for-manipulation (see VENDORED.md). Kept as-is:

* **Cartesian delta actions with an integrated target pose.** ``step``
  accumulates deltas into an internal setpoint instead of adding them to the
  measured TCP pose, so servo lag never integrates into drift.
* **Full snapshot/restore.** ``get_state``/``set_state`` capture everything
  needed to resume bit-exactly, including the controller setpoint and the
  solver warm start. The "twin" verifier uses this to dry-run a plan and then
  put the world back exactly as it was.
* **Grasp detection** from finger contacts, backed by a geometric test, because
  MuJoCo drops finger contacts for a few steps under lift acceleration.

Changed: there is no single target object, phase tracker or reward. Which
object matters is decided by the plan, and whether the episode succeeded is
decided by the task's goal check, not by the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .panda import HOME_ARM_QPOS, PandaRobot, grasp_quat
from .scene import BLOCK_HALF, SimObject, SimSceneSpec, build_model

ACTION_DIM = 7  # dx, dy, dz, droll, dpitch, dyaw, gripper


@dataclass
class WorldConfig:
    control_hz: float = 20.0
    physics_dt: float = 0.002
    max_delta_pos: float = 0.035  # metres per control step
    max_delta_rot: float = 0.18  # radians per control step
    ik_iters: int = 40  # per control step; seeded from the previous solution
    gravity_compensation: bool = True
    workspace_low: tuple[float, float, float] = (0.05, -0.48, 0.012)
    workspace_high: tuple[float, float, float] = (0.90, 0.48, 0.42)
    #: Physics steps run after placing objects, so stacked blocks settle before
    #: anyone (the LLM's scene description included) looks at them.
    settle_steps: int = 150

    def substeps(self) -> int:
        n = round((1.0 / self.control_hz) / self.physics_dt)
        if n < 1:
            raise ValueError("control_hz too high for physics_dt")
        return int(n)


@dataclass(frozen=True)
class ObjectState:
    """Where one object is, as measured in the simulator."""

    name: str
    kind: str
    color: str
    pos: tuple[float, float, float]
    yaw: float = 0.0


class TabletopWorld:
    """MuJoCo tabletop scene with exact state restore."""

    def __init__(self, objects: tuple[SimObject, ...], config: WorldConfig | None = None):
        self.config = config or WorldConfig()
        cfg = self.config
        self.objects = tuple(objects)
        self.model, self.scene_xml = build_model(
            SimSceneSpec(
                objects=self.objects,
                timestep=cfg.physics_dt,
                gravity_compensation=cfg.gravity_compensation,
            )
        )
        self.data = mujoco.MjData(self.model)
        self.robot = PandaRobot(self.model, self.data)

        self._spec = {o.name: o for o in self.objects}
        self._body = {
            o.name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, o.name)
            for o in self.objects
        }
        self._qadr: dict[str, int] = {}
        self._dofadr: dict[str, int] = {}
        for o in self.objects:
            if o.kind != "block":
                continue
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{o.name}_free")
            self._qadr[o.name] = int(self.model.jnt_qposadr[jid])
            self._dofadr[o.name] = int(self.model.jnt_dofadr[jid])
        self._left_finger = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self._right_finger = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

        self._substeps = cfg.substeps()
        self._step_count = 0
        self._target_pos = np.zeros(3)
        self._target_quat = grasp_quat(0.0)
        self._yaw = 0.0
        self._gripper_cmd = 1.0
        self._renderer: mujoco.Renderer | None = None
        #: Free-text label of what the robot is doing, for captions in demos.
        self.status = ""

    # ------------------------------------------------------------------ objects

    @property
    def block_names(self) -> list[str]:
        return [o.name for o in self.objects if o.kind == "block"]

    @property
    def bowl_name(self) -> str | None:
        return next((o.name for o in self.objects if o.kind == "bowl"), None)

    def has(self, name: str) -> bool:
        return name in self._spec

    def kind(self, name: str) -> str:
        return self._spec[name].kind

    # ------------------------------------------------------------------ reset

    def reset(self) -> None:
        """Arm to the ready pose, objects to their spawn poses, then settle."""
        mujoco.mj_resetData(self.model, self.data)
        ready = self.robot.ready_qpos()
        self.robot.reset_arm(ready, gripper_opening=1.0)

        for o in self.objects:
            if o.kind != "block":
                continue
            adr = self._qadr[o.name]
            self.data.qpos[adr : adr + 3] = o.pos
            yaw = o.yaw
            self.data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            dof = self._dofadr[o.name]
            self.data.qvel[dof : dof + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self._target_pos = self.robot.tcp_pos.copy()
        self._target_quat = self.robot.tcp_quat.copy()
        self._yaw = 0.0
        self._gripper_cmd = 1.0
        self._step_count = 0
        for _ in range(self.config.settle_steps):
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------ step

    def step(self, action: np.ndarray) -> None:
        """Apply one normalised Cartesian-delta action (see ``ACTION_DIM``)."""
        cfg = self.config
        action = np.clip(np.asarray(action, dtype=float).reshape(ACTION_DIM), -1.0, 1.0)

        self._target_pos = np.clip(
            self._target_pos + action[:3] * cfg.max_delta_pos,
            np.array(cfg.workspace_low),
            np.array(cfg.workspace_high),
        )
        # Only yaw is integrated; roll/pitch stay at the top-down grasp.
        # Re-deriving the quaternion each step stops drift from tilting the gripper.
        self._yaw = float(np.clip(self._yaw + action[5] * cfg.max_delta_rot, -np.pi / 2, np.pi / 2))
        self._target_quat = grasp_quat(self._yaw)
        self._gripper_cmd = float((action[6] + 1.0) / 2.0)

        result = self.robot.solve_ik(
            self._target_pos,
            self._target_quat,
            seed_qpos=self.robot.arm_qpos,
            max_iters=cfg.ik_iters,
        )
        self.robot.set_arm_ctrl(result.qpos)
        self.robot.set_gripper_ctrl(self._gripper_cmd)
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

    @property
    def yaw(self) -> float:
        return self._yaw

    @property
    def step_count(self) -> int:
        return self._step_count

    # ------------------------------------------------------------------ queries

    def object_pos(self, name: str) -> np.ndarray:
        if self._spec[name].kind == "bowl":
            return np.array(self._spec[name].pos, dtype=float)
        return self.data.xpos[self._body[name]].copy()

    def object_quat(self, name: str) -> np.ndarray:
        return self.data.xquat[self._body[name]].copy()

    def object_yaw(self, name: str) -> float:
        w, x, y, z = self.object_quat(name)
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def object_speed(self, name: str) -> float:
        dof = self._dofadr[name]
        return float(np.linalg.norm(self.data.qvel[dof : dof + 3]))

    def contact_grasp(self, name: str) -> bool:
        """Raw contact test: both fingers touching ``name``."""
        body = self._body[name]
        left = right = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if b1 == body:
                other = b2
            elif b2 == body:
                other = b1
            else:
                continue
            if other == self._left_finger:
                left = True
            elif other == self._right_finger:
                right = True
            if left and right:
                return True
        return False

    def geometric_hold(self, name: str) -> bool:
        """Fingers closed *on the block* rather than on air.

        Closing on empty air drives the width to ~0, which never matches a
        block width, so this cannot mistake a missed grasp for a held block.
        """
        width_matches = abs(self.robot.gripper_width - 2.0 * BLOCK_HALF) < 0.008
        if not width_matches:
            return False
        return float(np.linalg.norm(self.object_pos(name) - self.robot.tcp_pos)) < 0.05

    def is_held(self, name: str) -> bool:
        return self.contact_grasp(name) or self.geometric_hold(name)

    def held_object(self) -> str | None:
        for name in self.block_names:
            if self.is_held(name):
                return name
        return None

    def object_states(self) -> dict[str, ObjectState]:
        out = {}
        for o in self.objects:
            pos = self.object_pos(o.name)
            yaw = 0.0 if o.kind == "bowl" else self.object_yaw(o.name)
            out[o.name] = ObjectState(
                name=o.name,
                kind=o.kind,
                color=o.color,
                pos=tuple(float(v) for v in pos),
                yaw=yaw,
            )
        return out

    def reachable(
        self, pos: np.ndarray, yaw: float = 0.0, tol: float = 0.005
    ) -> tuple[bool, float]:
        """Whether a top-down TCP pose at ``pos`` is kinematically reachable.

        Solved from the home posture on scratch state, so it never disturbs the
        simulation. This is the check behind "a grasp it can't reach".
        """
        result = self.robot.solve_ik(
            np.asarray(pos, dtype=float), grasp_quat(yaw), seed_qpos=HOME_ARM_QPOS, max_iters=200
        )
        ok = bool(result.converged or (result.pos_err < tol and result.rot_err < 0.05))
        return ok, float(result.pos_err)

    # ------------------------------------------------------------------ snapshot / restore

    def get_state(self) -> dict[str, Any]:
        """Full snapshot sufficient to resume bit-exactly."""
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "act": self.data.act.copy() if self.data.act.size else np.zeros(0),
            "qacc_warmstart": self.data.qacc_warmstart.copy(),
            "time": float(self.data.time),
            "target_pos": self._target_pos.copy(),
            "target_quat": self._target_quat.copy(),
            "yaw": float(self._yaw),
            "gripper_cmd": float(self._gripper_cmd),
            "step_count": int(self._step_count),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.data.qpos[:] = state["qpos"]
        self.data.qvel[:] = state["qvel"]
        self.data.ctrl[:] = state["ctrl"]
        if self.data.act.size:
            self.data.act[:] = state["act"]
        self.data.qacc_warmstart[:] = state["qacc_warmstart"]
        self.data.time = state["time"]
        self._target_pos = np.array(state["target_pos"], dtype=float)
        self._target_quat = np.array(state["target_quat"], dtype=float)
        self._yaw = float(state["yaw"])
        self._gripper_cmd = float(state["gripper_cmd"])
        self._step_count = int(state["step_count"])
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ rendering

    def render(self, camera: str = "scene", width: int = 640, height: int = 480) -> np.ndarray:
        """RGB frame. One renderer is created and reused: under software GL,
        every renderer after the first returns black frames."""
        if self._renderer is None or (self._renderer.width, self._renderer.height) != (
            width,
            height,
        ):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height, width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
