"""The task suite: instruction templates, their scenes, goals and reference plans.

Each template is a function from a random generator to a task: a scene, an
instruction, a goal check and a reference plan. ``make_task(template, seed)``
is deterministic, so a (template, seed) pair names the same episode for every
planner, prompt and verifier setting. That is the pairing the statistics rely on.

Categories exist to answer one of the two questions the project is built for:
*which kinds of instruction break LLM planners?*

``simple``        one object, one destination; the floor any planner should clear.
``multi_step``    several dependent actions in a required order.
``precondition``  the obvious plan is blocked (something is on top) and the
                  planner has to notice and clear it first.
``spatial``       left/right/in front/behind/between, and superlatives
                  ("leftmost", "closest"), which need arithmetic on coordinates.
``counting``      "two blocks", "all the red blocks", "every block except".
``infeasible``    the instruction cannot be carried out (the object is missing,
                  out of reach, or immovable). Success is doing nothing.

Goals are checked on measured simulator state with ``predicates``. The
reference plan is how the oracle planner solves the task; the oracle's
success rate is the suite's execution ceiling (see README).
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import predicates as P
from .plan import Plan, Step, pick, place_at, place_on
from .scenes import (
    BLOCK_COLORS,
    SPAWN_SEPARATION,
    SPAWN_X,
    SPAWN_Y,
    Resample,
    Scene,
    SceneBuilder,
    max_displacement,
)
from .sim.world import ObjectState

States = dict[str, ObjectState]
Goal = Callable[[States, States], tuple[bool, str]]

CATEGORIES = ("simple", "multi_step", "precondition", "spatial", "counting", "infeasible")

#: Offset used by reference plans for spatial placements. Inside the
#: [SPATIAL_MIN, SPATIAL_MAX] band with margin on both sides.
SPATIAL_OFFSET = 0.12

#: For infeasible tasks: nothing may move further than this.
UNTOUCHED_TOL = 0.02


@dataclass
class Task:
    template: str
    category: str
    seed: int
    instruction: str
    scene: Scene
    goal: Goal
    reference: Plan
    feasible: bool = True


@dataclass
class _Draft:
    instruction: str
    builder: SceneBuilder
    goal: Callable[[dict[str, str]], Goal]
    reference: Callable[[dict[str, str], Scene], list[Step]] | None


TEMPLATES: dict[str, tuple[str, Callable[[np.random.Generator], _Draft]]] = {}


def template(name: str, category: str):
    assert category in CATEGORIES

    def register(fn):
        if name in TEMPLATES:
            raise ValueError(f"duplicate template {name}")
        TEMPLATES[name] = (category, fn)
        return fn

    return register


def template_ids(categories: list[str] | None = None) -> list[str]:
    return [t for t, (c, _) in TEMPLATES.items() if categories is None or c in categories]


def category_of(template_id: str) -> str:
    return TEMPLATES[template_id][0]


def make_task(template_id: str, seed: int, max_attempts: int = 50) -> Task:
    """Build the task for ``(template_id, seed)``. Deterministic."""
    category, fn = TEMPLATES[template_id]
    # Seed from both the template and the seed, so seed 3 of one template is
    # unrelated to seed 3 of another. crc32 is stable across processes; hash() is not.
    rng = np.random.default_rng([seed, zlib.crc32(template_id.encode())])
    for _ in range(max_attempts):
        try:
            draft = fn(rng)
            scene, ids = draft.builder.build()
            if draft.reference is None:
                reference = Plan(
                    feasible=False, reason="the instruction cannot be carried out", steps=[]
                )
            else:
                reference = Plan(steps=draft.reference(ids, scene))
        except Resample:
            continue
        return Task(
            template=template_id,
            category=category,
            seed=seed,
            instruction=draft.instruction,
            scene=scene,
            goal=draft.goal(ids),
            reference=reference,
            feasible=draft.reference is not None,
        )
    raise RuntimeError(f"template {template_id} could not build a scene for seed {seed}")


# ---------------------------------------------------------------------------
# helpers


def _colors(rng: np.random.Generator, k: int, exclude: tuple[str, ...] = ()) -> list[str]:
    pool = [c for c in BLOCK_COLORS if c not in exclude]
    return [str(c) for c in rng.choice(pool, size=k, replace=False)]


def _ok(cond: bool, detail: str) -> tuple[bool, str]:
    return bool(cond), ("" if cond else detail)


def _all(*checks: tuple[bool, str]) -> tuple[bool, str]:
    failed = [d for ok, d in checks if not ok]
    return (not failed), "; ".join(failed)


def _xy(scene: Scene, oid: str) -> np.ndarray:
    o = scene.get(oid)
    return np.array([o.x, o.y])


def _free_spot(
    scene: Scene, rng: np.random.Generator, moving: tuple[str, ...] = (), taken=()
) -> np.ndarray:
    """A table spot clear of everything that is not about to move."""
    others = [
        (np.array([o.x, o.y]), P.MIN_CLEARANCE + 0.02 if o.kind == "block" else 0.23)
        for o in scene.objects
        if o.id not in moving and o.on == "table"
    ]
    others += [(np.asarray(t), P.MIN_CLEARANCE + 0.02) for t in taken]
    for _ in range(500):
        xy = np.array([rng.uniform(*SPAWN_X), rng.uniform(*SPAWN_Y)])
        if all(np.linalg.norm(xy - p) >= sep for p, sep in others):
            return xy
    raise Resample("no free spot for a reference placement")


def _blocks_in_bowl(states: States) -> list[str]:
    return [n for n, s in states.items() if s.kind == "block" and P.in_bowl(n, states)]


def _scatter(b: SceneBuilder, colors: list[str]) -> list[str]:
    return [b.add_block(c) for c in colors]


# ---------------------------------------------------------------------------
# simple


def _bowl_one(rng, phrase: str) -> _Draft:
    b = SceneBuilder(rng)
    b.add_bowl()
    cs = _colors(rng, int(rng.integers(3, 5)))
    hs = _scatter(b, cs)
    t = hs[0]
    return _Draft(
        phrase.format(c=cs[0]),
        b,
        lambda ids: lambda s0, s1: _ok(P.in_bowl(ids[t], s1), f"{ids[t]} is not in the bowl"),
        lambda ids, sc: [pick(ids[t]), place_on("bowl")],
    )


@template("bowl_one", "simple")
def _t_bowl_one(rng):
    return _bowl_one(rng, "Put the {c} block in the bowl.")


@template("bowl_one_synonym", "simple")
def _t_bowl_one_syn(rng):
    return _bowl_one(rng, "Drop the {c} cube into the bowl.")


def _stack_two(rng, phrase: str) -> _Draft:
    b = SceneBuilder(rng)
    if rng.random() < 0.5:
        b.add_bowl()
    cs = _colors(rng, int(rng.integers(3, 5)))
    hs = _scatter(b, cs)
    a, base = hs[0], hs[1]
    return _Draft(
        phrase.format(a=cs[0], b=cs[1]),
        b,
        lambda ids: (
            lambda s0, s1: _ok(P.on(ids[a], ids[base], s1), f"{ids[a]} is not on {ids[base]}")
        ),
        lambda ids, sc: [pick(ids[a]), place_on(ids[base])],
    )


@template("stack_two", "simple")
def _t_stack_two(rng):
    return _stack_two(rng, "Stack the {a} block on the {b} block.")


@template("stack_two_synonym", "simple")
def _t_stack_two_syn(rng):
    return _stack_two(rng, "Place the {a} one on top of the {b} one.")


# ---------------------------------------------------------------------------
# multi_step


@template("bowl_then_stack", "multi_step")
def _t_bowl_then_stack(rng):
    b = SceneBuilder(rng)
    b.add_bowl()
    cs = _colors(rng, int(rng.integers(3, 5)))
    hs = _scatter(b, cs)
    x, y = hs[0], hs[1]
    return _Draft(
        f"Put the {cs[0]} block in the bowl, then stack the {cs[1]} one on it.",
        b,
        lambda ids: (
            lambda s0, s1: _all(
                _ok(P.in_bowl(ids[x], s1), f"{ids[x]} is not in the bowl"),
                _ok(P.on(ids[y], ids[x], s1), f"{ids[y]} is not on {ids[x]}"),
            )
        ),
        lambda ids, sc: [pick(ids[x]), place_on("bowl"), pick(ids[y]), place_on(ids[x])],
    )


@template("two_in_bowl", "multi_step")
def _t_two_in_bowl(rng):
    b = SceneBuilder(rng)
    b.add_bowl()
    cs = _colors(rng, int(rng.integers(3, 5)))
    hs = _scatter(b, cs)
    x, y = hs[0], hs[1]
    return _Draft(
        f"Put the {cs[0]} block and the {cs[1]} block in the bowl.",
        b,
        lambda ids: (
            lambda s0, s1: _all(
                _ok(P.in_bowl(ids[x], s1), f"{ids[x]} is not in the bowl"),
                _ok(P.in_bowl(ids[y], s1), f"{ids[y]} is not in the bowl"),
            )
        ),
        lambda ids, sc: [pick(ids[x]), place_on("bowl"), pick(ids[y]), place_on("bowl")],
    )


def _tower_goal(ids, hs):
    def goal(s0, s1):
        chain = [ids[h] for h in hs]
        checks = [_ok(P.on_table(chain[0], s1), f"{chain[0]} is not on the table")]
        for lower, upper in zip(chain, chain[1:]):
            checks.append(_ok(P.on(upper, lower, s1), f"{upper} is not on {lower}"))
        return _all(*checks)

    return goal


def _tower_ref(hs):
    def ref(ids, sc):
        steps = []
        for lower, upper in zip(hs, hs[1:]):
            steps += [pick(ids[upper]), place_on(ids[lower])]
        return steps

    return ref


@template("tower_three", "multi_step")
def _t_tower_three(rng):
    b = SceneBuilder(rng)
    cs = _colors(rng, int(rng.integers(3, 5)))
    hs = _scatter(b, cs)
    return _Draft(
        f"Build a tower with the {cs[0]} block at the bottom, the {cs[1]} block in the "
        f"middle and the {cs[2]} block on top.",
        b,
        lambda ids: _tower_goal(ids, hs[:3]),
        _tower_ref(hs[:3]),
    )


@template("tower_four", "multi_step")
def _t_tower_four(rng):
    b = SceneBuilder(rng)
    cs = _colors(rng, int(rng.integers(4, 6)))
    hs = _scatter(b, cs)
    return _Draft(
        f"Stack the blocks into one tower, from the bottom up: {cs[0]}, {cs[1]}, {cs[2]}, {cs[3]}.",
        b,
        lambda ids: _tower_goal(ids, hs[:4]),
        _tower_ref(hs[:4]),
    )


@template("swap_stack", "multi_step")
def _t_swap_stack(rng):
    b = SceneBuilder(rng)
    cs = _colors(rng, int(rng.integers(3, 5)))
    bottom = b.add_block(cs[1])
    top = b.add_block(cs[0], on=bottom)
    _scatter(b, cs[2:])
    return _Draft(
        f"The {cs[0]} block is sitting on the {cs[1]} block. Swap them so the {cs[1]} block is on top.",
        b,
        lambda ids: (
            lambda s0, s1: _ok(
                P.on(ids[bottom], ids[top], s1), f"{ids[bottom]} is not on {ids[top]}"
            )
        ),
        lambda ids, sc: [
            pick(ids[top]),
            place_at(*_free_spot(sc, rng, moving=(ids[top],))),
            pick(ids[bottom]),
            place_on(ids[top]),
        ],
    )


# ---------------------------------------------------------------------------
# precondition


@template("clear_then_bowl", "precondition")
def _t_clear_then_bowl(rng):
    b = SceneBuilder(rng)
    b.add_bowl()
    cs = _colors(rng, int(rng.integers(3, 5)))
    t = b.add_block(cs[0])
    cover = b.add_block(cs[1], on=t)
    _scatter(b, cs[2:])
    return _Draft(
        f"Put the {cs[0]} block in the bowl.",
        b,
        lambda ids: lambda s0, s1: _ok(P.in_bowl(ids[t], s1), f"{ids[t]} is not in the bowl"),
        lambda ids, sc: [
            pick(ids[cover]),
            place_at(*_free_spot(sc, rng, moving=(ids[cover],))),
            pick(ids[t]),
            place_on("bowl"),
        ],
    )


@template("clear_then_stack", "precondition")
def _t_clear_then_stack(rng):
    b = SceneBuilder(rng)
    cs = _colors(rng, int(rng.integers(3, 5)))
    t = b.add_block(cs[0])
    cover = b.add_block(cs[1], on=t)
    dest = b.add_block(cs[2])
    _scatter(b, cs[3:])
    return _Draft(
        f"Stack the {cs[0]} block on the {cs[2]} block.",
        b,
        lambda ids: (
            lambda s0, s1: _ok(P.on(ids[t], ids[dest], s1), f"{ids[t]} is not on {ids[dest]}")
        ),
        lambda ids, sc: [
            pick(ids[cover]),
            place_at(*_free_spot(sc, rng, moving=(ids[cover],))),
            pick(ids[t]),
            place_on(ids[dest]),
        ],
    )


@template("unstack_all", "precondition")
def _t_unstack_all(rng):
    b = SceneBuilder(rng)
    cs = _colors(rng, int(rng.integers(3, 5)))
    h0 = b.add_block(cs[0])
    h1 = b.add_block(cs[1], on=h0)
    h2 = b.add_block(cs[2], on=h1)
    _scatter(b, cs[3:])

    def ref(ids, sc):
        s1 = _free_spot(sc, rng)
        s2 = _free_spot(sc, rng, taken=[s1])
        return [pick(ids[h2]), place_at(*s1), pick(ids[h1]), place_at(*s2)]

    def goal(ids):
        def g(s0, s1):
            blocks = [n for n, s in s1.items() if s.kind == "block"]
            return _all(*[_ok(P.on_table(n, s1), f"{n} is not on the table") for n in blocks])

        return g

    return _Draft("Unstack the tower so that every block sits directly on the table.", b, goal, ref)


# ---------------------------------------------------------------------------
# spatial


def _relation(rng, relation: str, phrase: str, offset_xy: tuple[float, float]) -> _Draft:
    b = SceneBuilder(rng)
    if rng.random() < 0.5:
        b.add_bowl()
    cs = _colors(rng, int(rng.integers(3, 5)))
    mover = b.add_block(cs[0])
    anchor = b.add_block(cs[1])
    anchor_xy = np.array([b._items[-1]["x"], b._items[-1]["y"]])
    # The spot the reference plan uses must exist; otherwise resample the scene.
    spot = anchor_xy + SPATIAL_OFFSET * np.array(offset_xy)
    _scatter(b, cs[2:])
    if not (0.35 <= spot[0] <= 0.68 and abs(spot[1]) <= 0.36) or not b.is_free(
        spot, separation=P.MIN_CLEARANCE + 0.02, ignore=(mover,)
    ):
        raise Resample("no room for the relation")
    rel = getattr(P, relation)
    return _Draft(
        phrase.format(a=cs[0], b=cs[1]),
        b,
        lambda ids: (
            lambda s0, s1: _all(
                _ok(
                    rel(ids[mover], ids[anchor], s1),
                    f"{ids[mover]} is not {relation.replace('_', ' ')} {ids[anchor]}",
                ),
                _ok(P.on_table(ids[mover], s1), f"{ids[mover]} is not on the table"),
            )
        ),
        lambda ids, sc: [pick(ids[mover]), place_at(*spot)],
    )


@template("left_of", "spatial")
def _t_left_of(rng):
    return _relation(rng, "left_of", "Move the {a} block to the left of the {b} block.", (0, 1))


@template("right_of", "spatial")
def _t_right_of(rng):
    return _relation(rng, "right_of", "Move the {a} block to the right of the {b} block.", (0, -1))


@template("in_front_of", "spatial")
def _t_in_front_of(rng):
    return _relation(rng, "in_front_of", "Put the {a} block in front of the {b} block.", (-1, 0))


@template("behind", "spatial")
def _t_behind(rng):
    return _relation(rng, "behind", "Put the {a} block behind the {b} block.", (1, 0))


@template("between", "spatial")
def _t_between(rng):
    b = SceneBuilder(rng)
    cs = _colors(rng, int(rng.integers(3, 5)))
    mover = b.add_block(cs[0])
    e1 = b.add_block(cs[1])
    xy1 = np.array([b._items[-1]["x"], b._items[-1]["y"]])
    for _ in range(200):
        xy2 = np.array([rng.uniform(*SPAWN_X), rng.uniform(*SPAWN_Y)])
        if 0.24 <= np.linalg.norm(xy2 - xy1) <= 0.40 and b.is_free(xy2):
            break
    else:
        raise Resample("no partner for between")
    e2 = b.add_block(cs[2], xy=xy2)
    _scatter(b, cs[3:])
    mid = (xy1 + xy2) / 2
    if not b.is_free(mid, separation=P.MIN_CLEARANCE + 0.02, ignore=(mover,)):
        raise Resample("midpoint occupied")
    return _Draft(
        f"Place the {cs[0]} block between the {cs[1]} block and the {cs[2]} block.",
        b,
        lambda ids: (
            lambda s0, s1: _ok(
                P.between(ids[mover], ids[e1], ids[e2], s1),
                f"{ids[mover]} is not between {ids[e1]} and {ids[e2]}",
            )
        ),
        lambda ids, sc: [pick(ids[mover]), place_at(*mid)],
    )


def _extreme(rng, key: Callable[[np.ndarray], float], margin: float) -> tuple[SceneBuilder, str]:
    """Scatter blocks so exactly one is the extreme by ``key``, by at least ``margin``."""
    b = SceneBuilder(rng)
    b.add_bowl()
    cs = _colors(rng, int(rng.integers(3, 5)))
    _scatter(b, cs)
    vals = sorted(
        (
            (key(np.array([it["x"], it["y"]])), it["handle"])
            for it in b._items
            if it["kind"] == "block"
        ),
        reverse=True,
    )
    if vals[0][0] - vals[1][0] < margin:
        raise Resample("extreme not distinct")
    return b, vals[0][1]


@template("leftmost_to_bowl", "spatial")
def _t_leftmost(rng):
    b, lm = _extreme(rng, lambda xy: xy[1], 0.05)
    return _Draft(
        "Put the leftmost block in the bowl.",
        b,
        lambda ids: (
            lambda s0, s1: _all(
                _ok(P.in_bowl(ids[lm], s1), f"{ids[lm]} is not in the bowl"),
                _ok(_blocks_in_bowl(s1) == [ids[lm]], f"bowl holds {_blocks_in_bowl(s1)}"),
            )
        ),
        lambda ids, sc: [pick(ids[lm]), place_on("bowl")],
    )


@template("closest_to_bowl", "spatial")
def _t_closest(rng):
    b = SceneBuilder(rng)
    b.add_bowl()
    bowl = np.array([b._items[0]["x"], b._items[0]["y"]])
    _scatter(b, _colors(rng, int(rng.integers(3, 5))))
    d = sorted(
        (float(np.linalg.norm(np.array([it["x"], it["y"]]) - bowl)), it["handle"])
        for it in b._items
        if it["kind"] == "block"
    )
    if d[1][0] - d[0][0] < 0.05:
        raise Resample("closest not distinct")
    near = d[0][1]
    return _Draft(
        "Put the block that is closest to the bowl into the bowl.",
        b,
        lambda ids: (
            lambda s0, s1: _all(
                _ok(P.in_bowl(ids[near], s1), f"{ids[near]} is not in the bowl"),
                _ok(_blocks_in_bowl(s1) == [ids[near]], f"bowl holds {_blocks_in_bowl(s1)}"),
            )
        ),
        lambda ids, sc: [pick(ids[near]), place_on("bowl")],
    )


@template("rightmost_on", "spatial")
def _t_rightmost_on(rng):
    b = SceneBuilder(rng)
    _scatter(b, _colors(rng, int(rng.integers(3, 5))))
    vals = sorted(((-it["y"], it["handle"], it["color"]) for it in b._items), reverse=True)
    if vals[0][0] - vals[1][0] < 0.05:
        raise Resample("rightmost not distinct")
    rm = vals[0][1]
    _, dest, dest_color = vals[1 + int(rng.integers(0, len(vals) - 1))]
    return _Draft(
        f"Stack the rightmost block on the {dest_color} block.",
        b,
        lambda ids: (
            lambda s0, s1: _ok(P.on(ids[rm], ids[dest], s1), f"{ids[rm]} is not on {ids[dest]}")
        ),
        lambda ids, sc: [pick(ids[rm]), place_on(ids[dest])],
    )


# ---------------------------------------------------------------------------
# counting


def _exactly_in_bowl(n: int, allowed: Callable[[str], bool] | None = None):
    def goal(ids):
        def g(s0, s1):
            inside = _blocks_in_bowl(s1)
            ok = len(inside) == n and all(allowed is None or allowed(x) for x in inside)
            return _ok(ok, f"bowl holds {inside}, expected {n}")

        return g

    return goal


@template("count_two_bowl", "counting")
def _t_count_two(rng):
    b = SceneBuilder(rng)
    b.add_bowl()
    hs = _scatter(b, _colors(rng, 4))
    return _Draft(
        "Put two blocks in the bowl.",
        b,
        _exactly_in_bowl(2),
        lambda ids, sc: [pick(ids[hs[0]]), place_on("bowl"), pick(ids[hs[1]]), place_on("bowl")],
    )


@template("count_three_bowl", "counting")
def _t_count_three(rng):
    b = SceneBuilder(rng)
    b.add_bowl()
    hs = _scatter(b, _colors(rng, 5))
    return _Draft(
        "Put exactly three blocks in the bowl.",
        b,
        _exactly_in_bowl(3),
        lambda ids, sc: [s for h in hs[:3] for s in (pick(ids[h]), place_on("bowl"))],
    )


@template("all_color_bowl", "counting")
def _t_all_color(rng):
    b = SceneBuilder(rng)
    b.add_bowl()
    c = _colors(rng, 1)[0]
    n = int(rng.integers(2, 4))
    same = [b.add_block(c) for _ in range(n)]
    _scatter(b, _colors(rng, int(rng.integers(1, 3)), exclude=(c,)))

    def goal(ids):
        def g(s0, s1):
            inside = set(_blocks_in_bowl(s1))
            want = {ids[h] for h in same}
            return _ok(inside == want, f"bowl holds {sorted(inside)}, expected {sorted(want)}")

        return g

    return _Draft(
        f"Put all the {c} blocks in the bowl.",
        b,
        goal,
        lambda ids, sc: [s for h in same for s in (pick(ids[h]), place_on("bowl"))],
    )


@template("all_except", "counting")
def _t_all_except(rng):
    b = SceneBuilder(rng)
    b.add_bowl()
    cs = _colors(rng, 4)
    hs = _scatter(b, cs)

    def goal(ids):
        def g(s0, s1):
            inside = set(_blocks_in_bowl(s1))
            want = {ids[h] for h in hs[1:]}
            return _ok(inside == want, f"bowl holds {sorted(inside)}, expected {sorted(want)}")

        return g

    return _Draft(
        f"Put every block except the {cs[0]} one in the bowl.",
        b,
        goal,
        lambda ids, sc: [s for h in hs[1:] for s in (pick(ids[h]), place_on("bowl"))],
    )


@template("count_color_two", "counting")
def _t_count_color_two(rng):
    b = SceneBuilder(rng)
    b.add_bowl()
    c = _colors(rng, 1)[0]
    same = [b.add_block(c) for _ in range(3)]
    _scatter(b, _colors(rng, 1, exclude=(c,)))

    def goal(ids):
        names = {ids[h] for h in same}
        return _exactly_in_bowl(2, allowed=lambda x: x in names)(ids)

    return _Draft(
        f"Put two of the {c} blocks in the bowl.",
        b,
        goal,
        lambda ids, sc: [s for h in same[:2] for s in (pick(ids[h]), place_on("bowl"))],
    )


@template("stack_three_any", "counting")
def _t_stack_three_any(rng):
    b = SceneBuilder(rng)
    hs = _scatter(b, _colors(rng, 4))

    def goal(ids):
        def g(s0, s1):
            tallest = max((len(t) for t in P.towers(s1)), default=0)
            return _ok(tallest == 3, f"tallest stack has {tallest} blocks, expected 3")

        return g

    return _Draft(
        "Stack three of the blocks into a single tower.",
        b,
        goal,
        _tower_ref(hs[:3]),
    )


# ---------------------------------------------------------------------------
# infeasible


def _untouched(ids):
    def g(s0, s1):
        d = max_displacement(s0, s1)
        return _ok(d < UNTOUCHED_TOL, f"objects moved by up to {d:.3f} m")

    return g


@template("missing_color", "infeasible")
def _t_missing(rng):
    b = SceneBuilder(rng)
    b.add_bowl()
    cs = _colors(rng, int(rng.integers(3, 5)))
    _scatter(b, cs)
    absent = _colors(rng, 1, exclude=tuple(cs))[0]
    return _Draft(f"Put the {absent} block in the bowl.", b, _untouched, None)


@template("unreachable_block", "infeasible")
def _t_unreachable(rng):
    b = SceneBuilder(rng)
    b.add_bowl()
    cs = _colors(rng, int(rng.integers(3, 5)))
    b.add_block(cs[0], region="unreachable")
    _scatter(b, cs[1:])
    return _Draft(f"Put the {cs[0]} block in the bowl.", b, _untouched, None)


@template("no_bowl", "infeasible")
def _t_no_bowl(rng):
    b = SceneBuilder(rng)
    cs = _colors(rng, int(rng.integers(3, 5)))
    _scatter(b, cs)
    return _Draft(f"Put the {cs[0]} block in the bowl.", b, _untouched, None)


@template("move_bowl", "infeasible")
def _t_move_bowl(rng):
    b = SceneBuilder(rng)
    b.add_bowl()
    cs = _colors(rng, int(rng.integers(3, 5)))
    _scatter(b, cs)
    return _Draft(f"Put the bowl on top of the {cs[0]} block.", b, _untouched, None)


# Spawn constants are part of the suite definition; re-exported for docs/tests.
__all__ = [
    "CATEGORIES",
    "TEMPLATES",
    "Task",
    "make_task",
    "template_ids",
    "category_of",
    "SPAWN_SEPARATION",
]
