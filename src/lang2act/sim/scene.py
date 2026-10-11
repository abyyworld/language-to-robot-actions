"""Programmatic MJCF construction for the tabletop language tasks.

Adapted from ``icm.envs.assets.scene`` in
abyyworld/Interactive-correction-for-manipulation (see VENDORED.md). The table,
cameras, Menagerie include, wrist camera, TCP site and gravity compensation are
unchanged. What changed is the object set: instead of one target, two
distractors and a goal pad, a scene is any number of coloured blocks plus an
optional bowl, because the language tasks are about *which* objects end up
where.

Geometry conventions
--------------------
* The robot base sits at the world origin, bolted to the table surface, which
  is the plane ``z = 0``. Every object z-coordinate is a height above the table.
* The robot looks along +x. +y is the robot's left. This is the frame the LLM is
  told about, so "left of" has exactly one meaning.
* Blocks rest on the table at ``z = BLOCK_HALF``.
* The bowl is fixed to the table. It is a container, not something to pick up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .assets import fetch

# --- Table / workspace geometry (metres) ------------------------------------
TABLE_HEIGHT = 0.75  # table top above the floor
TABLE_HALF = (0.55, 0.50, 0.025)  # half-extents of the table slab
TABLE_CENTER_X = 0.35  # slab centre; robot base at x=0 sits near the back edge

BLOCK_HALF = 0.021  # 4.2 cm cube: fits the 8 cm Panda gripper stroke with margin
BLOCK_MASS = 0.05

# The bowl: a flat floor ringed by short walls. Built from boxes rather than a
# mesh so it needs no extra assets and its contact geometry is exact.
BOWL_INNER_RADIUS = 0.11
BOWL_WALL_THICKNESS = 0.008
BOWL_WALL_HEIGHT = 0.05
BOWL_FLOOR_THICKNESS = 0.006
BOWL_SEGMENTS = 18
BOWL_OUTER_RADIUS = BOWL_INNER_RADIUS + BOWL_WALL_THICKNESS

# Distance from the Panda hand flange (the ``hand`` body origin) to the point
# midway between the closed fingertips. Every controller target is expressed in
# this frame.
TCP_OFFSET_Z = 0.1034

COLOR_RGBA: dict[str, tuple[float, float, float, float]] = {
    "red": (0.85, 0.15, 0.15, 1.0),
    "green": (0.15, 0.65, 0.25, 1.0),
    "blue": (0.15, 0.35, 0.85, 1.0),
    "yellow": (0.92, 0.80, 0.15, 1.0),
    "purple": (0.55, 0.25, 0.75, 1.0),
    "orange": (0.95, 0.50, 0.10, 1.0),
    "gray": (0.42, 0.47, 0.55, 1.0),
}


@dataclass(frozen=True)
class SimObject:
    """One body in the scene. ``name`` is the MuJoCo body name and the object id."""

    name: str
    kind: str  # "block" | "bowl"
    color: str
    pos: tuple[float, float, float]
    yaw: float = 0.0


@dataclass
class SimSceneSpec:
    """Everything needed to emit a scene MJCF."""

    objects: tuple[SimObject, ...] = ()
    timestep: float = 0.002
    camera_fovy: float = 58.0
    scene_cam_pos: tuple[float, float, float] = (0.98, -0.62, 0.58)
    scene_cam_target: tuple[float, float, float] = (0.46, 0.0, 0.03)
    front_cam_pos: tuple[float, float, float] = (1.35, 0.0, 0.60)
    front_cam_target: tuple[float, float, float] = (0.45, 0.0, 0.05)
    # Eye-in-hand camera, injected into the Panda ``hand`` body post-parse.
    wrist_cam_pos: tuple[float, float, float] = (0.0, -0.050, 0.012)
    wrist_cam_target: tuple[float, float, float] = (0.0, 0.0, 0.13)
    wrist_cam_up: tuple[float, float, float] = (0.0, -1.0, 0.0)
    #: Compensate gravity on the robot links (not the objects). A real Franka
    #: does this inside its own controller.
    gravity_compensation: bool = True
    _panda_xml: Path | None = field(default=None, repr=False)


def lookat_xyaxes(pos, target, up=(0.0, 0.0, 1.0)) -> str:
    """Return a MuJoCo ``xyaxes`` string orienting a camera at ``pos`` toward ``target``.

    MuJoCo cameras look down their local -z axis with +x right and +y up. Solving
    for that basis numerically is far less error-prone than hand-writing
    quaternions, and it means moving a camera is a one-line change.
    """
    pos = np.asarray(pos, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - pos
    n = np.linalg.norm(forward)
    if n < 1e-9:
        raise ValueError("camera position and target coincide")
    forward /= n
    cam_z = -forward
    up = np.asarray(up, dtype=float)
    x_axis = np.cross(up, cam_z)
    if np.linalg.norm(x_axis) < 1e-6:  # looking straight up/down: pick another up
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), cam_z)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(cam_z, x_axis)
    return " ".join(f"{v:.6f}" for v in (*x_axis, *y_axis))


def lookat_basis(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Return the 3x3 camera rotation (columns = x, y, z axes) looking at ``target``."""
    pos = np.asarray(pos, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - pos
    n = np.linalg.norm(forward)
    if n < 1e-9:
        raise ValueError("camera position and target coincide")
    forward /= n
    cam_z = -forward
    up = np.asarray(up, dtype=float)
    x_axis = np.cross(up, cam_z)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), cam_z)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(cam_z, x_axis)
    return np.column_stack([x_axis, y_axis, cam_z])


def lookat_quat(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Quaternion (w, x, y, z) orienting a camera at ``pos`` toward ``target``."""
    import mujoco

    basis = lookat_basis(pos, target, up)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, basis.flatten())
    return quat


def _panda_include_path() -> Path:
    """Path to a keyframe-stripped copy of the Menagerie ``panda.xml``.

    The upstream file ends with a ``<keyframe>`` whose ``qpos`` has exactly 9
    entries (7 arm joints + 2 fingers). Adding free-jointed objects raises ``nq``
    to 9 + 7*n_objects, and MuJoCo rejects a keyframe whose size no longer
    matches. We cache a stripped copy rather than mutating the fetched assets so
    that re-running the fetcher stays idempotent, and we set the home pose from
    Python instead (see ``lang2act.sim.panda.HOME_ARM_QPOS``).
    """
    if not fetch.is_available():
        raise FileNotFoundError(
            "Franka Panda assets are not installed.\n"
            "Run:  make assets       (or)  python -m lang2act.sim.assets.fetch\n"
            f"Expected them at: {fetch.TARGET_DIR}"
        )
    src = fetch.TARGET_DIR / "panda.xml"
    dst = fetch.TARGET_DIR / "_panda_nokeyframe.xml"
    if not dst.is_file() or dst.stat().st_mtime < src.stat().st_mtime:
        text = src.read_text()
        text = re.sub(r"<keyframe>.*?</keyframe>", "", text, flags=re.DOTALL)
        dst.write_text(text)
    return dst


def _table_xml() -> str:
    hx, hy, hz = TABLE_HALF
    top_z = -hz  # slab centre so that its top surface is exactly z = 0
    leg_h = (TABLE_HEIGHT - 2 * hz) / 2.0
    leg_z = -2 * hz - leg_h
    legs = []
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        lx = TABLE_CENTER_X + sx * (hx - 0.06)
        ly = sy * (hy - 0.06)
        legs.append(
            f'      <geom name="table_leg{i}" type="box" size="0.03 0.03 {leg_h:.4f}" '
            f'pos="{lx:.4f} {ly:.4f} {leg_z:.4f}" material="table_leg" contype="0" conaffinity="0"/>'
        )
    legs_xml = "\n".join(legs)
    return f"""    <body name="table" pos="0 0 0">
      <geom name="table_top" type="box" size="{hx} {hy} {hz}" pos="{TABLE_CENTER_X} 0 {top_z:.4f}"
            material="table" friction="1.0 0.02 0.001" solimp="0.98 0.99 0.001" solref="0.005 1"/>
{legs_xml}
    </body>"""


def _block_xml(obj: SimObject) -> str:
    # Free joint: ``TabletopWorld.reset`` writes the real pose into qpos, so the
    # body position here only has to be somewhere harmless.
    rgba = " ".join(str(c) for c in COLOR_RGBA[obj.color])
    x, y, z = obj.pos
    h = BLOCK_HALF
    return f"""    <body name="{obj.name}" pos="{x:.4f} {y:.4f} {z:.4f}">
      <freejoint name="{obj.name}_free"/>
      <inertial pos="0 0 0" mass="{BLOCK_MASS}" diaginertia="1e-4 1e-4 1e-4"/>
      <geom name="{obj.name}_geom" type="box" size="{h} {h} {h}" rgba="{rgba}"
            friction="1.0 0.02 0.001" solimp="0.98 0.99 0.001" solref="0.005 1"
            condim="4" priority="1"/>
    </body>"""


def _bowl_xml(obj: SimObject) -> str:
    rgba = " ".join(str(c) for c in COLOR_RGBA.get(obj.color, COLOR_RGBA["gray"]))
    x, y, _ = obj.pos
    wall_r = BOWL_INNER_RADIUS + BOWL_WALL_THICKNESS / 2
    # Segment length: chord of one sector plus a little overlap so no gaps open.
    seg_half = wall_r * np.tan(np.pi / BOWL_SEGMENTS) + 0.002
    hz = BOWL_WALL_HEIGHT / 2
    walls = []
    for i in range(BOWL_SEGMENTS):
        a = 2 * np.pi * i / BOWL_SEGMENTS
        walls.append(
            f'      <geom name="{obj.name}_wall{i}" type="box" '
            f'size="{BOWL_WALL_THICKNESS / 2:.4f} {seg_half:.4f} {hz:.4f}" '
            f'pos="{wall_r * np.cos(a):.4f} {wall_r * np.sin(a):.4f} {hz:.4f}" '
            f'euler="0 0 {a:.5f}" rgba="{rgba}" friction="1.0 0.02 0.001"/>'
        )
    walls_xml = "\n".join(walls)
    return f"""    <body name="{obj.name}" pos="{x:.4f} {y:.4f} 0">
      <geom name="{obj.name}_floor" type="cylinder"
            size="{BOWL_OUTER_RADIUS:.4f} {BOWL_FLOOR_THICKNESS / 2:.4f}"
            pos="0 0 {BOWL_FLOOR_THICKNESS / 2:.4f}" rgba="{rgba}" friction="1.0 0.02 0.001"/>
{walls_xml}
    </body>"""


def build_scene_xml(spec: SimSceneSpec) -> str:
    """Emit the complete task MJCF as a string."""
    panda_xml = spec._panda_xml or _panda_include_path()
    meshdir = (panda_xml.parent / "assets").as_posix()

    bodies = []
    for obj in spec.objects:
        if obj.kind == "block":
            bodies.append(_block_xml(obj))
        elif obj.kind == "bowl":
            bodies.append(_bowl_xml(obj))
        else:
            raise ValueError(f"unknown object kind {obj.kind!r}")
    objects_xml = "\n".join(bodies)

    scene_axes = lookat_xyaxes(spec.scene_cam_pos, spec.scene_cam_target)
    front_axes = lookat_xyaxes(spec.front_cam_pos, spec.front_cam_target)

    return f"""<mujoco model="lang2act_tabletop">
  <option timestep="{spec.timestep}" integrator="implicitfast" cone="elliptic" impratio="10"/>

  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="2048" offsamples="4"/>
    <map znear="0.01" zfar="6"/>
    <global azimuth="140" elevation="-25" offwidth="1280" offheight="960"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.58" rgb2="0.08 0.10 0.14"
             width="512" height="1024"/>
    <texture type="2d" name="floor_tex" builtin="checker" mark="edge"
             rgb1="0.22 0.24 0.27" rgb2="0.16 0.18 0.21" markrgb="0.4 0.4 0.4"
             width="300" height="300"/>
    <material name="floor" texture="floor_tex" texuniform="true" texrepeat="4 4" reflectance="0.05"/>
    <texture type="2d" name="table_tex" builtin="flat" rgb1="0.62 0.52 0.40" rgb2="0.62 0.52 0.40"
             width="64" height="64"/>
    <material name="table" texture="table_tex" texuniform="true" reflectance="0.02" shininess="0.1"/>
    <material name="table_leg" rgba="0.25 0.26 0.28 1"/>
  </asset>

  <worldbody>
    <light name="key" pos="0.6 -0.4 1.6" dir="-0.3 0.25 -1" directional="true"
           diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    <light name="fill" pos="0.2 0.6 1.4" dir="0.15 -0.4 -1" directional="true" diffuse="0.28 0.28 0.3"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 {-TABLE_HEIGHT}" material="floor"/>

    <camera name="scene" pos="{" ".join(str(v) for v in spec.scene_cam_pos)}"
            xyaxes="{scene_axes}" fovy="{spec.camera_fovy}" mode="fixed"/>
    <camera name="front" pos="{" ".join(str(v) for v in spec.front_cam_pos)}"
            xyaxes="{front_axes}" fovy="{spec.camera_fovy}" mode="fixed"/>

{_table_xml()}
{objects_xml}
  </worldbody>

  <include file="{panda_xml.as_posix()}"/>

  <!-- MuJoCo applies the LAST <compiler> it parses, so this must follow the
       include: it redirects mesh lookup to the fetched Menagerie assets, whose
       own relative meshdir does not survive textual inclusion. -->
  <compiler angle="radian" meshdir="{meshdir}" autolimits="true"/>
</mujoco>"""


def _mujoco_version() -> str:
    import mujoco

    return getattr(mujoco, "__version__", "unknown")


def build_model(spec: SimSceneSpec):
    """Compile the scene and return ``(mujoco.MjModel, xml_string)``.

    The wrist camera cannot be written into the MJCF text: it must be a child of
    the Panda's ``hand`` body, which only exists after the Menagerie include is
    parsed. We therefore parse to an ``MjSpec``, attach the camera to the real
    body, and compile from there. That keeps camera placement a Python-level
    parameter instead of file surgery on fetched third-party assets.
    """
    import mujoco

    xml = build_scene_xml(spec)
    try:
        mj_spec = mujoco.MjSpec.from_string(xml)
    except ValueError as exc:  # surface the offending XML, MuJoCo errors are terse
        raise ValueError(f"failed to parse scene MJCF: {exc}") from exc

    hand = find_spec_body(mj_spec, "hand")
    if hand is None:
        raise RuntimeError(
            "Panda 'hand' body not found - the pinned Menagerie assets may have changed "
            f"shape, or this MuJoCo version ({_mujoco_version()}) exposes MjSpec bodies "
            "differently. Bodies present: "
            + ", ".join(sorted(getattr(b, "name", "?") for b in getattr(mj_spec, "bodies", [])))
        )
    cam = hand.add_camera()
    cam.name = "wrist"
    cam.pos = np.asarray(spec.wrist_cam_pos, dtype=float)
    cam.quat = lookat_quat(spec.wrist_cam_pos, spec.wrist_cam_target, spec.wrist_cam_up)
    cam.fovy = spec.camera_fovy

    # Grasp frame. Menagerie ships no site on the hand, so every downstream
    # component would otherwise re-derive the fingertip offset independently.
    tcp = hand.add_site()
    tcp.name = "tcp"
    tcp.pos = np.array([0.0, 0.0, TCP_OFFSET_Z])
    tcp.size = np.array([0.005, 0.005, 0.005])
    tcp.rgba = np.array([1.0, 0.2, 0.2, 0.0])  # invisible; used for kinematics only

    if spec.gravity_compensation:
        # Must happen before compile: MuJoCo counts gravity-compensated bodies at
        # compile time (``ngravcomp``) and skips the computation entirely when
        # that count is zero. Writing ``model.body_gravcomp`` on an already
        # compiled model silently does nothing - the field changes, the force
        # never appears.
        for body in _iter_bodies(mj_spec.worldbody):
            if body.name.startswith("link") or body.name in ("hand", "left_finger", "right_finger"):
                body.gravcomp = 1.0

    model = mj_spec.compile()
    return model, xml


def find_spec_body(mj_spec, name: str):
    """Look up a body in an ``MjSpec`` across MuJoCo versions.

    The accessor was renamed: ``MjSpec.find_body(name)`` in 3.2.x became
    ``MjSpec.body(name)`` in 3.3.x. Supporting a range of MuJoCo versions in
    pyproject while calling only one of these means the package installs happily
    and then fails at the first environment construction - which is exactly what
    happened on a machine that resolved 3.3.7 instead of the 3.2.7 used during
    development. Iterating ``spec.bodies`` is the final fallback and works on
    every version.
    """
    for accessor in ("body", "find_body"):
        fn = getattr(mj_spec, accessor, None)
        if fn is None:
            continue
        try:
            found = fn(name)
        except (KeyError, ValueError, TypeError):
            continue
        if found is not None:
            return found
    for body in getattr(mj_spec, "bodies", []):
        if getattr(body, "name", None) == name:
            return body
    return None


def _iter_bodies(body):
    """Depth-first walk over an MjSpec body tree."""
    for child in body.bodies:
        yield child
        yield from _iter_bodies(child)
