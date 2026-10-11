import pytest

from conftest import requires_assets

pytestmark = [requires_assets, pytest.mark.sim]


@pytest.fixture(scope="module")
def setup():
    """clear_then_bowl: one block sits on another, and there is a bowl."""
    from lang2act.sim.world import TabletopWorld
    from lang2act.tasks import make_task
    from lang2act.verifier import Verifier

    task = make_task("clear_then_bowl", 0)
    w = TabletopWorld(task.scene.to_sim_objects())
    w.reset()
    return task, w, Verifier(w, "symbolic")


def codes(v):
    return [x.code for x in v]


def covered(task):
    return next(o for o in task.scene.blocks if o.on != "table").on


def cover(task):
    return next(o for o in task.scene.blocks if o.on != "table").id


def test_reference_plan_passes(setup):
    task, w, v = setup
    assert v.check(task.reference) == []


def test_unknown_object(setup):
    from lang2act.plan import Plan, pick, place_on

    _, _, v = setup
    out = v.check(Plan(steps=[pick("pink_block"), place_on("bowl")]))
    assert codes(out) == ["unknown_object"]
    assert "Objects present" in out[0].message


def test_blocked_pick(setup):
    from lang2act.plan import Plan, pick, place_on

    task, _, v = setup
    out = v.check(Plan(steps=[pick(covered(task)), place_on("bowl")]))
    assert codes(out) == ["not_clear"]
    assert cover(task) in out[0].message


def test_sequence_rules(setup):
    from lang2act.plan import Plan, pick, place_on

    task, _, v = setup
    assert codes(v.check(Plan(steps=[place_on("bowl")]))) == ["hand_empty"]
    c = cover(task)
    assert codes(v.check(Plan(steps=[pick(c)]))) == ["ends_holding"]
    other = next(o.id for o in task.scene.blocks if o.on == "table" and o.id != covered(task))
    assert "hand_full" in codes(v.check(Plan(steps=[pick(c), pick(other), place_on("bowl")])))
    assert codes(v.check(Plan(steps=[pick("bowl"), place_on(c)]))) == ["not_graspable"]
    assert codes(v.check(Plan(steps=[pick(c), place_on(c)]))) == ["self_target"]


def test_placement_rules(setup):
    from lang2act.plan import Plan, pick, place_at

    task, w, v = setup
    c = cover(task)
    bowl = w.object_pos("bowl")
    other = next(o for o in task.scene.blocks if o.on == "table" and o.id != covered(task))
    assert codes(v.check(Plan(steps=[pick(c), place_at(0.95, 0.0)]))) == ["off_table"]
    assert codes(v.check(Plan(steps=[pick(c), place_at(0.80, 0.30)]))) == ["unreachable"]
    assert codes(v.check(Plan(steps=[pick(c), place_at(0.2, 0.0)]))) == ["unreachable"]
    assert codes(v.check(Plan(steps=[pick(c), place_at(bowl[0], bowl[1])]))) == ["use_place_on"]
    assert codes(v.check(Plan(steps=[pick(c), place_at(other.x + 0.03, other.y)]))) == ["collision"]


def test_infeasible_claims_are_not_checked(setup):
    from lang2act.plan import Plan

    _, _, v = setup
    assert v.check(Plan(feasible=False, reason="no", steps=[])) == []


def test_unreachable_block_is_caught():
    from lang2act.plan import Plan, pick, place_on
    from lang2act.sim.world import TabletopWorld
    from lang2act.tasks import make_task
    from lang2act.verifier import Verifier

    task = make_task("unreachable_block", 0)
    w = TabletopWorld(task.scene.to_sim_objects())
    w.reset()
    far = max(task.scene.blocks, key=lambda o: o.x).id
    assert codes(Verifier(w).check(Plan(steps=[pick(far), place_on("bowl")]))) == ["unreachable"]


def test_twin_dry_run_restores_the_world(setup):
    from lang2act.verifier import Verifier

    task, w, _ = setup
    before = w.object_states()
    assert Verifier(w, "twin").check(task.reference) == []
    after = w.object_states()
    for name, s in before.items():
        assert max(abs(a - b) for a, b in zip(s.pos, after[name].pos)) < 1e-9, name
