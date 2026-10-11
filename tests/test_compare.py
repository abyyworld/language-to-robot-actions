from pathlib import Path

import numpy as np
import pytest

from lang2act.compare import compare_runs, format_comparison
from lang2act.harness import Run

SUITE = {"name": "t", "seed_base": 0, "seeds_per_template": 20, "templates": ["a", "b", "c", "d"]}
CATS = {"a": "simple", "b": "simple", "c": "spatial", "d": "spatial"}


def make_run(name, success_by_key, stages=None, suite=SUITE):
    records = []
    for (t, s), ok in success_by_key.items():
        stage = (stages or {}).get((t, s), "success" if ok else "goal_not_met")
        records.append(
            {"template": t, "seed": s, "category": CATS[t], "success": bool(ok), "stage": stage}
        )
    return Run(Path(name), {"label": name, "suite": suite, "condition": {}, "planner": {}}, records)


def keys():
    return [(t, s) for t in SUITE["templates"] for s in range(20)]


def test_clear_improvement_is_better():
    rng = np.random.default_rng(0)
    base = {k: rng.random() < 0.4 for k in keys()}
    cand = {k: v or rng.random() < 0.8 for k, v in base.items()}
    cmp = compare_runs(make_run("base", base), make_run("cand", cand), n_boot=2000)
    assert cmp.overall.verdict == "BETTER"
    assert cmp.overall.losses == 0
    reverse = compare_runs(make_run("cand", cand), make_run("base", base), n_boot=2000)
    assert reverse.overall.verdict == "WORSE"


def test_small_noisy_difference_is_underpowered_and_says_how_many():
    rng = np.random.default_rng(1)
    base = {k: rng.random() < 0.5 for k in keys()}
    cand = dict(base)
    fails = [k for k, v in base.items() if not v]
    wins = [k for k, v in base.items() if v]
    for k in fails[:6]:
        cand[k] = True  # 6 wins
    for k in wins[:4]:
        cand[k] = False  # 4 losses
    cmp = compare_runs(make_run("base", base), make_run("cand", cand), n_boot=2000)
    assert cmp.overall.verdict == "UNDERPOWERED"
    assert cmp.overall.needed_pairs and cmp.overall.needed_pairs > cmp.overall.n
    assert "paired episodes needed" in cmp.headline()


def test_identical_runs_are_equivalent():
    base = {k: True for k in keys()}
    cmp = compare_runs(make_run("a", base), make_run("b", dict(base)), n_boot=500)
    assert cmp.overall.verdict == "EQUIVALENT"


def test_api_errors_are_dropped_from_pairs():
    base = {k: True for k in keys()}
    cand = dict(base)
    errs = {keys()[0]: "api_error", keys()[1]: "api_error"}
    cand[keys()[0]] = cand[keys()[1]] = False
    cmp = compare_runs(make_run("a", base), make_run("b", cand, errs), n_boot=500)
    assert cmp.dropped_api_errors == 2
    assert cmp.overall.n == len(keys()) - 2
    assert cmp.overall.losses == 0


def test_category_rows_use_holm_adjusted_p():
    rng = np.random.default_rng(2)
    base = {k: rng.random() < 0.5 for k in keys()}
    cand = dict(base)
    cmp = compare_runs(make_run("a", base), make_run("b", cand), n_boot=500)
    assert [r.group for r in cmp.groups] == ["simple", "spatial"]
    assert all(r.p_adjusted >= r.p_value for r in cmp.groups)
    assert "overall" in format_comparison(cmp)


def test_refuses_unpaired_suites():
    other = dict(SUITE, seed_base=100)
    base = {k: True for k in keys()}
    with pytest.raises(ValueError, match="different suites"):
        compare_runs(make_run("a", base), make_run("b", base, suite=other))


def test_interval_never_collapses_with_few_agreeing_templates():
    """Three templates with identical differences: the cluster bootstrap alone gives zero width."""
    base = {k: True for k in keys()}
    cand = {k: (k[1] % 10 != 0) for k in keys()}  # every template loses 2 of 20
    cmp = compare_runs(make_run("a", base), make_run("b", cand), n_boot=2000)
    row = cmp.overall
    assert row.ci_high - row.ci_low > 0.02
