"""Markdown report over one or more runs.

Sections, in the order a reader needs them:

1. **Runs**: each condition's overall success rate with a Wilson interval.
2. **By category**: where each condition succeeds and fails.
3. **Where episodes end**: the failure-stage breakdown.
4. **What the verifier catches**: rule violations in first-draft plans, i.e.
   the mistakes the planner makes before any feedback.
5. **Paired comparisons** of every run against the first one.

Every number in it comes from the ``episodes.jsonl`` files it was given.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .compare import compare_runs, format_comparison
from .episode import STAGES
from .harness import Run
from .stats import wilson_interval
from .tasks import CATEGORIES


def _rate(records: list[dict]) -> str:
    n = len(records)
    if n == 0:
        return "-"
    k = sum(r["success"] for r in records)
    ci = wilson_interval(k, n)
    return f"{ci.point:.0%} [{ci.low:.0%}, {ci.high:.0%}]"


def _scored(run: Run) -> list[dict]:
    return [r for r in run.records if r["stage"] != "api_error"]


def _condition(run: Run) -> str:
    c = run.meta["condition"]
    bits = [c["planner"], f"verifier={c['verifier']}", f"prompt={c['prompt']}"]
    if c["verifier"] != "none":
        bits.append(f"repairs={c['max_repairs']}")
    if c["planner"].startswith("anthropic") and c.get("effort"):
        bits.append(f"effort={c['effort']}")
    return ", ".join(bits)


def build_report(
    runs: list[Run], *, margin: float = 0.05, plot_path: str | Path | None = None
) -> str:
    out: list[str] = ["# Results", ""]
    synthetic = [r.name for r in runs if r.meta["planner"].get("synthetic")]
    if synthetic:
        out += [
            f"> **Synthetic planners present ({', '.join(synthetic)}).** They read the ground "
            "truth and exist to test the pipeline. Their numbers say nothing about language models.",
            "",
        ]

    out += [
        "## Runs",
        "",
        "| run | condition | episodes | success (95% CI) | API errors | mean rounds | tokens in / out |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for run in runs:
        recs = _scored(run)
        errs = len(run.records) - len(recs)
        rounds = sum(r["n_rounds"] for r in recs) / max(len(recs), 1)
        tin = sum(r["tokens_in"] for r in run.records)
        tout = sum(r["tokens_out"] for r in run.records)
        out.append(
            f"| {run.name} | {_condition(run)} | {len(recs)} | {_rate(recs)} | {errs} | {rounds:.2f} | {tin:,} / {tout:,} |"
        )

    out += [
        "",
        "## By category",
        "",
        "| category | " + " | ".join(r.name for r in runs) + " |",
        "|---|" + "---|" * len(runs),
    ]
    for cat in CATEGORIES:
        cells = [_rate([x for x in _scored(r) if x["category"] == cat]) for r in runs]
        out.append(f"| {cat} | " + " | ".join(cells) + " |")

    out += [
        "",
        "## Where episodes end",
        "",
        "| stage | " + " | ".join(r.name for r in runs) + " |",
        "|---|" + "---:|" * len(runs),
    ]
    counts = [Counter(x["stage"] for x in r.records) for r in runs]
    for stage in STAGES:
        if any(c.get(stage) for c in counts):
            out.append(f"| {stage} | " + " | ".join(str(c.get(stage, 0)) for c in counts) + " |")

    verified = [r for r in runs if r.meta["condition"]["verifier"] != "none"]
    if verified:
        out += [
            "",
            "## What the verifier catches in first drafts",
            "",
            "Rule violations in the planner's *first* plan, before any feedback. "
            "Repair rate is the share of rejected first drafts that a later round fixed.",
            "",
            "| run | first drafts rejected | repaired | top violations |",
            "|---|---:|---:|---|",
        ]
        for run in verified:
            recs = _scored(run)
            rejected = [r for r in recs if r["first_violation_codes"]]
            repaired = sum(r["repaired"] for r in rejected)
            codes = Counter(c for r in rejected for c in r["first_violation_codes"])
            top = ", ".join(f"{c} ({n})" for c, n in codes.most_common(5)) or "-"
            share = f"{len(rejected)}/{len(recs)}"
            fix = f"{repaired}/{len(rejected)}" if rejected else "-"
            out.append(f"| {run.name} | {share} | {fix} | {top} |")

    if len(runs) > 1:
        out += [
            "",
            "## Paired comparisons",
            "",
            f"Each run against **{runs[0].name}**. Equivalence margin ±{margin:.0%}.",
            "",
        ]
        for run in runs[1:]:
            try:
                cmp = compare_runs(runs[0], run, margin=margin)
            except ValueError as exc:
                out += [f"*{run.name}: not comparable ({exc})*", ""]
                continue
            out += ["```", format_comparison(cmp), "```", ""]

    if plot_path is not None:
        path = plot_category_chart(runs, plot_path)
        if path:
            out += ["", f"![success by category]({Path(path).name})", ""]
    return "\n".join(out) + "\n"


def plot_category_chart(runs: list[Run], path: str | Path) -> str | None:
    """Per-category success with Wilson intervals. Needs matplotlib (``pip install .[viz]``)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    import numpy as np

    # Categorical slots in fixed order (validated for colour-vision deficiency);
    # the category table above the chart is the table view the lighter slots require.
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    surface, ink, muted, grid = "#fcfcfb", "#2b2a27", "#898781", "#e1e0d9"
    plt.rcParams.update(
        {
            "font.size": 9,
            "text.color": ink,
            "axes.labelcolor": muted,
            "xtick.color": muted,
            "ytick.color": muted,
        }
    )
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    slot = 0.8 / len(runs)
    x = np.arange(len(CATEGORIES))
    for i, run in enumerate(runs):
        pts, lo, hi = [], [], []
        for cat in CATEGORIES:
            recs = [r for r in _scored(run) if r["category"] == cat]
            ci = wilson_interval(sum(r["success"] for r in recs), len(recs))
            pts.append(ci.point)
            lo.append(ci.point - ci.low)
            hi.append(ci.high - ci.point)
        # Narrower than the slot, so neighbouring bars are separated by surface, not a stroke.
        ax.bar(
            x + (i - (len(runs) - 1) / 2) * slot,
            pts,
            min(slot - 0.03, 0.22),
            yerr=[lo, hi],
            label=run.name,
            color=palette[i % len(palette)],
            error_kw={"elinewidth": 0.9, "ecolor": ink, "capsize": 0},
            zorder=2,
        )
    ax.set_xticks(x, CATEGORIES)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("success rate (95% CI)")
    ax.grid(axis="y", color=grid, linewidth=0.6, zorder=0)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.legend(
        frameon=False,
        fontsize=8,
        ncols=min(len(runs), 4),
        loc="upper left",
        bbox_to_anchor=(0, 1.16),
        labelcolor=ink,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return str(path)
