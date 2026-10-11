"""Simulator layer: world, skills, snapshot/restore."""

import numpy as np
import pytest

from conftest import requires_assets

pytestmark = [requires_assets, pytest.mark.sim]


@pytest.fixture(scope="module")
def world():
    from lang2act.sim.scene import BLOCK_HALF, SimObject
    from lang2act.sim.world import TabletopWorld

    w = TabletopWorld(
        (
            SimObject("red_block", "block", "red", (0.45, -0.15, BLOCK_HALF), 0.4),
            SimObject("blue_block", "block", "blue", (0.55, 0.05, BLOCK_HALF), -0.7),
            SimObject("green_block", "block", "green", (0.55, 0.05, 3 * BLOCK_HALF + 0.001), 0.2),
            SimObject("bowl", "bowl", "white", (0.50, 0.25, 0.0)),
        )
    )
    return w


def test_reset_settles_a_stack(world):
    from lang2act import predicates as P

    world.reset()
    s = world.object_states()
    assert P.support_of("green_block", s) == "blue_block"
    assert P.on_table("red_block", s) and P.on_table("blue_block", s)
    assert world.held_object() is None


def test_pick_and_place_into_bowl_and_onto_block(world):
    from lang2act import predicates as P
    from lang2act.sim.scene import BLOCK_HALF, BOWL_FLOOR_THICKNESS
    from lang2act.sim.skills import Skills

    world.reset()
    sk = Skills(world)
    assert sk.pick("red_block").ok
    assert world.held_object() == "red_block"
    sk.place((0.5 + 0.042, 0.25 + 0.042), BOWL_FLOOR_THICKNESS)
    assert P.in_bowl("red_block", world.object_states())
    assert sk.pick("green_block").ok
    red = world.object_pos("red_block")
    sk.place(red[:2], red[2] + BLOCK_HALF)
    assert P.on("green_block", "red_block", world.object_states())


def test_restore_is_exact_enough_to_replay(world):
    """The twin verifier dry-runs a plan and restores: the replay must match."""
    from lang2act.sim.skills import Skills

    world.reset()
    snap = world.get_state()
    Skills(world).pick("red_block")
    first = world.object_pos("red_block")
    world.set_state(snap)
    assert world.held_object() is None
    Skills(world).pick("red_block")
    assert np.abs(world.object_pos("red_block") - first).max() < 1e-5


def test_reachability_matches_the_measured_envelope(world):
    assert world.reachable(np.array([0.5, 0.0, 0.03]))[0]
    assert not world.reachable(np.array([0.87, 0.2, 0.03]))[0]
