"""Task generation. No simulator needed."""

import pytest

from lang2act.tasks import CATEGORIES, TEMPLATES, make_task, template_ids


def test_every_category_has_templates():
    cats = {c for c, _ in TEMPLATES.values()}
    assert cats == set(CATEGORIES)
    assert len(TEMPLATES) == 30


@pytest.mark.parametrize("template_id", list(TEMPLATES))
def test_templates_build_and_are_deterministic(template_id):
    for seed in range(4):
        a = make_task(template_id, seed)
        b = make_task(template_id, seed)
        assert a.instruction == b.instruction
        assert a.scene == b.scene
        assert a.reference == b.reference
        ids = a.scene.ids
        assert len(ids) == len(set(ids)), "object ids must be unique"
        for step in a.reference.steps:
            for ref in (step.object, step.target):
                assert ref is None or ref in ids, f"reference plan names unknown object {ref}"
        assert a.feasible == (a.category != "infeasible")


def test_seeds_differ_across_templates():
    """Seed 0 of one template must not be a copy of seed 0 of another."""
    a = make_task("bowl_one", 0).scene
    b = make_task("bowl_one_synonym", 0).scene
    assert a != b


def test_duplicate_colours_get_numbered_ids():
    task = make_task("all_color_bowl", 0)
    colour = task.instruction.split()[3]
    same = [o.id for o in task.scene.blocks if o.color == colour]
    assert len(same) >= 2
    assert all(i.startswith(f"{colour}_block_") for i in same)


def test_template_ids_filter():
    assert all(TEMPLATES[t][0] == "spatial" for t in template_ids(["spatial"]))
