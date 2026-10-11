"""Geometric relations between objects, measured from simulator state.

These are the single definition of what "on", "in the bowl" and "left of"
mean. Three things read them: the task goal checks (did the episode succeed),
the executor's per-step postconditions (did that step do what it said), and the
system prompt, which states the same thresholds to the LLM in words. If a
threshold changes here it must change in ``planners/prompts.py`` too, or the
planner is being graded against a rule it was never told.

Frame: the robot looks along +x, so +y is the robot's left and "in front of"
means closer to the robot (smaller x).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .sim.scene import BLOCK_HALF, BOWL_FLOOR_THICKNESS, BOWL_INNER_RADIUS, BOWL_WALL_HEIGHT
from .sim.world import ObjectState

#: A block is "on" another when their centres are this close horizontally...
STACK_XY_TOL = 0.025
#: ...and its centre sits one block-height higher, within this tolerance.
STACK_Z_TOL = 0.012
TABLE_Z_TOL = 0.010

#: "a left of b": a is between these distances further left than b...
SPATIAL_MIN = 0.08
SPATIAL_MAX = 0.20
#: ...and no further than this off the line through b.
SPATIAL_ALIGN = 0.06
#: "a between b and c": within this of the midpoint of b and c.
BETWEEN_TOL = 0.04

#: Minimum centre-to-centre distance between blocks on the table.
MIN_CLEARANCE = 0.08

States = Mapping[str, ObjectState]


def _xy(s: ObjectState) -> np.ndarray:
    return np.array(s.pos[:2], dtype=float)


def bowl_state(states: States) -> ObjectState | None:
    return next((s for s in states.values() if s.kind == "bowl"), None)


def in_bowl(name: str, states: States) -> bool:
    bowl = bowl_state(states)
    if bowl is None or name not in states:
        return False
    s = states[name]
    inside = np.linalg.norm(_xy(s) - _xy(bowl)) < BOWL_INNER_RADIUS
    return bool(inside and s.pos[2] < BOWL_WALL_HEIGHT + 2 * BLOCK_HALF + 0.02)


def support_of(name: str, states: States) -> str | None:
    """What block ``name`` rests on: ``"table"``, ``"bowl"``, a block id, or None."""
    s = states[name]
    best = None
    for other, o in states.items():
        if other == name or o.kind != "block":
            continue
        dz = s.pos[2] - o.pos[2]
        if (
            np.linalg.norm(_xy(s) - _xy(o)) < STACK_XY_TOL
            and abs(dz - 2 * BLOCK_HALF) < STACK_Z_TOL
        ):
            best = other
    if best is not None:
        return best
    if in_bowl(name, states) and abs(s.pos[2] - (BOWL_FLOOR_THICKNESS + BLOCK_HALF)) < TABLE_Z_TOL:
        return "bowl"
    if abs(s.pos[2] - BLOCK_HALF) < TABLE_Z_TOL and not in_bowl(name, states):
        return "table"
    return None


def on(a: str, b: str, states: States) -> bool:
    if b == "bowl":
        return in_bowl(a, states)
    if a not in states or b not in states:
        return False
    return support_of(a, states) == b


def on_table(a: str, states: States) -> bool:
    return support_of(a, states) == "table"


def blocks_on_top(name: str, states: States) -> list[str]:
    return [
        o
        for o in states
        if o != name and states[o].kind == "block" and support_of(o, states) == name
    ]


def _relative(a: str, b: str, states: States) -> np.ndarray:
    return _xy(states[a]) - _xy(states[b])


def left_of(a: str, b: str, states: States) -> bool:
    dx, dy = _relative(a, b, states)
    return bool(SPATIAL_MIN <= dy <= SPATIAL_MAX and abs(dx) <= SPATIAL_ALIGN)


def right_of(a: str, b: str, states: States) -> bool:
    dx, dy = _relative(a, b, states)
    return bool(SPATIAL_MIN <= -dy <= SPATIAL_MAX and abs(dx) <= SPATIAL_ALIGN)


def in_front_of(a: str, b: str, states: States) -> bool:
    dx, dy = _relative(a, b, states)
    return bool(SPATIAL_MIN <= -dx <= SPATIAL_MAX and abs(dy) <= SPATIAL_ALIGN)


def behind(a: str, b: str, states: States) -> bool:
    dx, dy = _relative(a, b, states)
    return bool(SPATIAL_MIN <= dx <= SPATIAL_MAX and abs(dy) <= SPATIAL_ALIGN)


def between(a: str, b: str, c: str, states: States) -> bool:
    mid = (_xy(states[b]) + _xy(states[c])) / 2
    return bool(np.linalg.norm(_xy(states[a]) - mid) <= BETWEEN_TOL and on_table(a, states))


def towers(states: States) -> list[list[str]]:
    """Every maximal stack, listed bottom to top."""
    above: dict[str, str] = {}
    for name, s in states.items():
        if s.kind != "block":
            continue
        sup = support_of(name, states)
        if sup not in (None, "table", "bowl"):
            above[sup] = name
    bottoms = [
        n
        for n, s in states.items()
        if s.kind == "block" and support_of(n, states) in ("table", "bowl")
    ]
    out = []
    for b in bottoms:
        chain = [b]
        while chain[-1] in above:
            chain.append(above[chain[-1]])
        out.append(chain)
    return out


# ---------------------------------------------------------------------------
# Bowl slots, shared by the verifier and the executor so they agree on capacity.

# Four corner slots, 8.4 cm apart: two blocks rotated 45 degrees need 6 cm
# between centres, and a centre slot plus a ring (tried first) let the second
# block land on the first's corner and shove it over the wall.
BOWL_SLOT_OFFSETS = (
    (0.042, 0.042),
    (-0.042, 0.042),
    (-0.042, -0.042),
    (0.042, -0.042),
)
BOWL_CAPACITY = len(BOWL_SLOT_OFFSETS)


def free_bowl_slot(bowl_xy, occupied_xy) -> np.ndarray | None:
    """First bowl slot at least a block-width away from every occupied position."""
    bowl_xy = np.asarray(bowl_xy, dtype=float)
    for off in BOWL_SLOT_OFFSETS:
        slot = bowl_xy + np.array(off)
        if all(np.linalg.norm(slot - np.asarray(p)) >= 0.06 for p in occupied_xy):
            return slot
    return None
