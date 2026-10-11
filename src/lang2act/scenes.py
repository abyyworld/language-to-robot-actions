"""Seeded scene generation and the scene description the planner sees.

A ``Scene`` is the task-level description of the table: which blocks exist,
their colours, where they start, and what they start on. It is generated from a
seed, so the same (task, seed) produces the same table for every planner
being compared. That pairing is what makes the comparisons in ``compare.py``
valid.

The planner never sees this object. It sees ``describe(world)``: positions
*measured in the simulator after the scene has settled*, rounded to the
millimetre, which is what a perception stack would hand it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .predicates import support_of
from .sim.scene import (
    BLOCK_HALF,
    BOWL_FLOOR_THICKNESS,
    BOWL_INNER_RADIUS,
    BOWL_OUTER_RADIUS,
    SimObject,
)
from .sim.world import ObjectState, TabletopWorld

BLOCK_COLORS = ("red", "green", "blue", "yellow", "purple", "orange")

# Where objects spawn. Measured, not guessed: top-down grasps are reachable out
# to ~0.78 m from the base, and an arm working closer than ~0.33 m clips tall
# stacks with its forearm. The spawn region keeps a margin from both.
SPAWN_X = (0.36, 0.62)
SPAWN_Y = (-0.30, 0.30)
BOWL_X = (0.42, 0.58)
BOWL_Y = (-0.20, 0.20)
#: Out of reach but still on the table (the table ends at x = 0.90).
UNREACHABLE_X = (0.84, 0.87)
UNREACHABLE_Y = (-0.25, 0.25)

#: Spawn spacing between block centres. Wider than the 0.08 m the planner is
#: told to keep, so a scene never starts out violating its own rules.
SPAWN_SEPARATION = 0.10
BOWL_SEPARATION = BOWL_OUTER_RADIUS + 0.07


class Resample(Exception):
    """Raised by a task template when the sampled scene cannot host the task."""


@dataclass(frozen=True)
class SceneObject:
    id: str
    kind: str  # "block" | "bowl"
    color: str
    x: float
    y: float
    on: str = "table"  # initial support: "table", "bowl", or a block id
    yaw: float = 0.0


@dataclass(frozen=True)
class Scene:
    objects: tuple[SceneObject, ...]

    def get(self, object_id: str) -> SceneObject:
        for o in self.objects:
            if o.id == object_id:
                return o
        raise KeyError(object_id)

    @property
    def ids(self) -> list[str]:
        return [o.id for o in self.objects]

    @property
    def blocks(self) -> list[SceneObject]:
        return [o for o in self.objects if o.kind == "block"]

    @property
    def bowl(self) -> SceneObject | None:
        return next((o for o in self.objects if o.kind == "bowl"), None)

    def _z(self, obj: SceneObject) -> float:
        if obj.on == "table":
            return BLOCK_HALF
        if obj.on == "bowl":
            return BOWL_FLOOR_THICKNESS + BLOCK_HALF
        return self._z(self.get(obj.on)) + 2 * BLOCK_HALF + 0.001

    def to_sim_objects(self) -> tuple[SimObject, ...]:
        out = []
        for o in self.objects:
            z = 0.0 if o.kind == "bowl" else self._z(o)
            out.append(SimObject(o.id, o.kind, o.color, (o.x, o.y, z), o.yaw))
        return tuple(out)

    def to_dict(self) -> dict:
        return {"objects": [asdict(o) for o in self.objects]}

    @staticmethod
    def from_dict(data: dict) -> Scene:
        return Scene(tuple(SceneObject(**o) for o in data["objects"]))


class SceneBuilder:
    """Rejection-samples a non-overlapping table layout.

    Blocks are added by *handle*; ids are assigned at ``build`` time so that a
    colour that appears once is ``red_block`` and a colour that appears twice is
    ``red_block_1`` / ``red_block_2``. ``build`` returns the handle-to-id map.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self._items: list[dict] = []

    def _xy_taken(self) -> list[tuple[np.ndarray, float]]:
        out = []
        for it in self._items:
            if it["on"] != "table":
                continue
            sep = BOWL_SEPARATION if it["kind"] == "bowl" else SPAWN_SEPARATION
            out.append((np.array([it["x"], it["y"]]), sep))
        return out

    def is_free(
        self, xy, separation: float = SPAWN_SEPARATION, ignore: tuple[str, ...] = ()
    ) -> bool:
        xy = np.asarray(xy, dtype=float)
        for it in self._items:
            if it["handle"] in ignore or it["on"] != "table":
                continue
            sep = BOWL_SEPARATION if it["kind"] == "bowl" else separation
            if np.linalg.norm(xy - np.array([it["x"], it["y"]])) < sep:
                return False
        return True

    def sample_xy(
        self, x_range=SPAWN_X, y_range=SPAWN_Y, separation=SPAWN_SEPARATION
    ) -> np.ndarray:
        for _ in range(500):
            xy = np.array([self.rng.uniform(*x_range), self.rng.uniform(*y_range)])
            if self.is_free(xy, separation):
                return xy
        raise Resample("no free spot")

    def add_bowl(self) -> str:
        for _ in range(500):
            xy = np.array([self.rng.uniform(*BOWL_X), self.rng.uniform(*BOWL_Y)])
            if all(np.linalg.norm(xy - p) >= BOWL_SEPARATION for p, _ in self._xy_taken()):
                break
        else:
            raise Resample("no room for the bowl")
        self._items.append(
            {
                "handle": "bowl",
                "kind": "bowl",
                "color": "gray",
                "x": float(xy[0]),
                "y": float(xy[1]),
                "on": "table",
                "yaw": 0.0,
            }
        )
        return "bowl"

    def add_block(
        self,
        color: str,
        *,
        on: str = "table",
        xy=None,
        region: str = "reach",
    ) -> str:
        handle = f"#{len(self._items)}"
        if on != "table":
            support = next(it for it in self._items if it["handle"] == on)
            x, y = support["x"], support["y"]
        elif xy is not None:
            if not self.is_free(xy):
                raise Resample("requested spot is taken")
            x, y = float(xy[0]), float(xy[1])
        elif region == "unreachable":
            x, y = self.sample_xy(UNREACHABLE_X, UNREACHABLE_Y)
        else:
            x, y = self.sample_xy()
        self._items.append(
            {
                "handle": handle,
                "kind": "block",
                "color": color,
                "x": float(x),
                "y": float(y),
                "on": on,
                "yaw": float(self.rng.uniform(-np.pi, np.pi)),
            }
        )
        return handle

    def build(self) -> tuple[Scene, dict[str, str]]:
        counts: dict[str, int] = {}
        for it in self._items:
            if it["kind"] == "block":
                counts[it["color"]] = counts.get(it["color"], 0) + 1
        seen: dict[str, int] = {}
        ids: dict[str, str] = {"bowl": "bowl", "table": "table"}
        for it in self._items:
            if it["kind"] != "block":
                continue
            c = it["color"]
            seen[c] = seen.get(c, 0) + 1
            ids[it["handle"]] = f"{c}_block" if counts[c] == 1 else f"{c}_block_{seen[c]}"
        objects = tuple(
            SceneObject(
                id=ids[it["handle"]],
                kind=it["kind"],
                color=it["color"],
                x=it["x"],
                y=it["y"],
                on=ids[it["on"]],
                yaw=it["yaw"],
            )
            for it in self._items
        )
        return Scene(objects), ids


# ---------------------------------------------------------------------------
# What the planner sees


def describe(world: TabletopWorld) -> dict:
    """Scene description handed to the planner, measured from the simulator."""
    states = world.object_states()
    objects = []
    for name, s in states.items():
        entry: dict = {
            "id": name,
            "type": s.kind,
            "color": s.color,
            "position": [round(v, 3) for v in s.pos],
        }
        if s.kind == "block":
            entry["on"] = support_of(name, states) or "unknown"
        else:
            entry["inner_radius"] = BOWL_INNER_RADIUS
        objects.append(entry)
    return {"objects": objects}


def max_displacement(before: dict[str, ObjectState], after: dict[str, ObjectState]) -> float:
    d = 0.0
    for name, s in before.items():
        if s.kind != "block" or name not in after:
            continue
        d = max(d, float(np.linalg.norm(np.array(after[name].pos) - np.array(s.pos))))
    return d
