"""Whole episodes: plan, verify, repair, execute, check."""

import pytest

from conftest import requires_assets

pytestmark = [requires_assets, pytest.mark.sim]


def run(template, seed, planner, verifier="symbolic", repairs=2):
    from lang2act.episode import Condition, run_episode
    from lang2act.planners import make_planner
    from lang2act.tasks import make_task

    cond = Condition(planner=planner, verifier=verifier, max_repairs=repairs)
    return run_episode(make_task(template, seed), make_planner(planner, cache_dir=None), cond)


@pytest.mark.parametrize(
    "template", ["bowl_then_stack", "clear_then_stack", "left_of", "all_color_bowl"]
)
def test_oracle_solves(template):
    rec = run(template, 0, "oracle")
    assert rec["success"], rec["goal_detail"]
    assert rec["stage"] == "success" and rec["n_rounds"] == 1


def test_oracle_declines_infeasible():
    rec = run("missing_color", 0, "oracle")
    assert rec["success"] and rec["declared_infeasible"] and rec["execution"] is None


class ScriptedPlanner:
    """Answers with a fixed sequence of responses, one per round."""

    name = "scripted"

    def __init__(self, texts):
        self.texts = texts

    def session(self, request, sample=0):
        from lang2act.planners.base import Completion, PlannerSession

        texts = list(self.texts)

        class S(PlannerSession):
            def propose(self):
                return Completion(text=texts.pop(0), model="scripted")

            def revise(self, feedback):
                self.feedback = feedback
                return Completion(text=texts.pop(0) if texts else "{}", model="scripted")

        return S()


def run_scripted(template, texts, verifier="symbolic"):
    from lang2act.episode import Condition, run_episode
    from lang2act.tasks import make_task

    task = make_task(template, 0)
    return task, run_episode(
        task, ScriptedPlanner(texts), Condition(planner="scripted", verifier=verifier)
    )


def test_verifier_feedback_repairs_a_hallucinated_id():
    from lang2act.plan import Plan, pick, place_on
    from lang2act.tasks import make_task

    task = make_task("bowl_one", 0)
    target = task.reference.steps[0].object
    bad = Plan(steps=[pick(target.replace("_block", "_cube")), place_on("bowl")]).to_json()
    _, rec = run_scripted("bowl_one", [bad, task.reference.to_json()])
    assert rec["success"] and rec["repaired"]
    assert rec["first_violation_codes"] == ["unknown_object"]
    assert "there is no object" in rec["rounds"][0]["feedback"]


def test_without_verifier_the_same_mistake_fails_in_execution():
    from lang2act.plan import Plan, pick, place_on
    from lang2act.tasks import make_task

    task = make_task("bowl_one", 0)
    bad = Plan(steps=[pick("red_cube"), place_on("bowl")]).to_json()
    _, rec = run_scripted("bowl_one", [bad], verifier="none")
    assert not rec["success"] and rec["stage"] == "execution_invalid"
    assert task  # same scene as above


def test_unrepaired_plan_is_never_executed():
    from lang2act.plan import Plan, pick, place_on

    bad = Plan(steps=[pick("nope"), place_on("bowl")]).to_json()
    _, rec = run_scripted("bowl_one", [bad, bad, bad])
    assert rec["stage"] == "verifier_rejected" and rec["execution"] is None
    assert rec["n_rounds"] == 3


def test_parse_errors_get_feedback_too():
    from lang2act.tasks import make_task

    task = make_task("bowl_one", 0)
    _, rec = run_scripted("bowl_one", ["I would pick up the block.", task.reference.to_json()])
    assert rec["success"] and "could not be read as a plan" in rec["rounds"][0]["feedback"]


def test_verifier_is_safe_refusal_on_infeasible_task():
    """Stopped by the verifier and nothing moved: counted as the right outcome."""
    from lang2act.plan import Plan, pick, place_on

    bad = Plan(steps=[pick("bowl"), place_on("red_block")]).to_json()
    _, rec = run_scripted("move_bowl", [bad, bad, bad])
    assert rec["success"] and rec["stage"] == "success" and not rec["declared_infeasible"]


def test_acting_on_infeasible_task_fails():
    from lang2act.plan import Plan, pick, place_at

    task, rec = run_scripted("no_bowl", ["{}"], verifier="none")
    assert rec["stage"] == "wrong_response"
    first = task.scene.blocks[0].id
    _, rec = run_scripted(
        "no_bowl", [Plan(steps=[pick(first), place_at(0.5, 0.33)]).to_json()], verifier="none"
    )
    assert rec["stage"] == "acted_on_infeasible" and not rec["success"]


def test_noisy_planner_is_deterministic():
    a = run("tower_three", 3, "noisy:0.9")
    b = run("tower_three", 3, "noisy:0.9")
    assert a["final_plan_text"] == b["final_plan_text"] and a["stage"] == b["stage"]
