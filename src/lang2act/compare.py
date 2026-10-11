"""Paired comparison of two runs: better, worse, or underpowered.

The verdict logic follows vla-evals' regression gate, adapted to the question
asked here, which is not "may this ship" but "which configuration is better".

``BETTER`` / ``WORSE``
    McNemar's exact test rejects "no difference" at ``alpha`` **and** the 95%
    interval on the difference excludes zero. Requiring both keeps the verdict
    conservative where the two disagree, which happens near the boundary.
``EQUIVALENT``
    The whole interval lies within ``±margin``: whatever the difference is, it
    is too small to matter.
``UNDERPOWERED``
    Neither. The run cannot tell the two apart, and the report says roughly
    how many paired episodes would be needed to. This is the honest outcome
    for most small comparisons, and reporting it is the point: a 6-point gap on
    50 episodes is not a finding.

Per-category comparisons are corrected for multiple testing (Holm) and use
the corrected p-value in their verdicts. Intervals resample whole templates,
since seeds of one template are not independent. A category has only 3-8
templates, too few for a cluster bootstrap to be precise, which is one reason a
verdict also needs McNemar's test to agree.

Episodes where either side hit an API error are dropped from the pairing and
counted, because an outage is not evidence about a planner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from .harness import Run
from .stats import (
    Interval,
    holm,
    mcnemar_exact,
    paired_bootstrap_diff,
    paired_cluster_bootstrap_diff,
    required_pairs,
)

Verdict = Literal["BETTER", "WORSE", "EQUIVALENT", "UNDERPOWERED"]


@dataclass
class Row:
    group: str
    n: int
    baseline_rate: float
    candidate_rate: float
    difference: float
    ci_low: float
    ci_high: float
    wins: int
    losses: int
    p_value: float
    p_adjusted: float
    verdict: Verdict
    needed_pairs: int | None = None


@dataclass
class Comparison:
    baseline: str
    candidate: str
    overall: Row
    groups: list[Row] = field(default_factory=list)
    dropped_api_errors: int = 0
    unmatched: int = 0
    margin: float = 0.05
    alpha: float = 0.05

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def headline(self) -> str:
        o = self.overall
        s = (
            f"{self.candidate} vs {self.baseline}: {o.candidate_rate:.1%} vs {o.baseline_rate:.1%} "
            f"({o.difference:+.1%}, 95% CI [{o.ci_low:+.1%}, {o.ci_high:+.1%}], "
            f"McNemar p={o.p_value:.3g}, {o.wins} wins / {o.losses} losses) -> {o.verdict}"
        )
        if o.verdict == "UNDERPOWERED" and o.needed_pairs:
            s += f" (~{o.needed_pairs} paired episodes needed)"
        return s


def _verdict(lo: float, hi: float, p: float, alpha: float, margin: float) -> Verdict:
    if p < alpha and lo > 0:
        return "BETTER"
    if p < alpha and hi < 0:
        return "WORSE"
    if -margin <= lo and hi <= margin:
        return "EQUIVALENT"
    return "UNDERPOWERED"


def _row(
    group: str,
    b: np.ndarray,
    c: np.ndarray,
    clusters: np.ndarray,
    *,
    alpha: float,
    margin: float,
    p_adjusted: float | None = None,
    n_boot: int = 10_000,
) -> Row:
    clustered = paired_cluster_bootstrap_diff(
        b.astype(float), c.astype(float), clusters, n_samples=n_boot, alpha=alpha
    )
    flat = paired_bootstrap_diff(b.astype(float), c.astype(float), n_samples=n_boot, alpha=alpha)
    # Clustering exists to widen the interval for correlated seeds. With a
    # handful of templates whose differences happen to agree, it can instead
    # collapse to zero width, so never report it narrower than the per-episode one.
    ci = Interval(clustered.point, min(clustered.low, flat.low), max(clustered.high, flat.high))
    wins, losses, p = mcnemar_exact(b, c)
    p_use = p if p_adjusted is None else p_adjusted
    verdict = _verdict(ci.low, ci.high, p_use, alpha, margin)
    needed = None
    if verdict == "UNDERPOWERED" and b.size:
        needed = required_pairs((wins + losses) / b.size, max(abs(ci.point), margin), alpha=alpha)
    return Row(
        group=group,
        n=int(b.size),
        baseline_rate=float(b.mean()) if b.size else float("nan"),
        candidate_rate=float(c.mean()) if c.size else float("nan"),
        difference=ci.point,
        ci_low=ci.low,
        ci_high=ci.high,
        wins=wins,
        losses=losses,
        p_value=p,
        p_adjusted=p_use,
        verdict=verdict,
        needed_pairs=needed,
    )


def compare_runs(
    baseline: Run,
    candidate: Run,
    *,
    by: str = "category",
    margin: float = 0.05,
    alpha: float = 0.05,
    n_boot: int = 10_000,
) -> Comparison:
    if baseline.meta["suite"] != candidate.meta["suite"]:
        raise ValueError(
            "the runs used different suites, so their episodes are not paired. "
            "Re-run both on the same suite file."
        )
    bk, ck = baseline.by_key(), candidate.by_key()
    shared = [k for k in bk if k in ck]
    unmatched = len(set(bk) ^ set(ck))
    keep = [k for k in shared if bk[k]["stage"] != "api_error" and ck[k]["stage"] != "api_error"]
    dropped = len(shared) - len(keep)
    if not keep:
        raise ValueError("no paired episodes to compare")

    b = np.array([bk[k]["success"] for k in keep], dtype=bool)
    c = np.array([ck[k]["success"] for k in keep], dtype=bool)
    tmpl = np.array([k[0] for k in keep])
    overall = _row("overall", b, c, tmpl, alpha=alpha, margin=margin, n_boot=n_boot)

    groups_of = np.array([bk[k][by] if by != "template" else k[0] for k in keep])
    labels = list(dict.fromkeys(groups_of.tolist()))
    raw = []
    for g in labels:
        m = groups_of == g
        raw.append(mcnemar_exact(b[m], c[m])[2])
    adjusted = holm(raw) if len(raw) > 1 else raw
    rows = [
        _row(
            g,
            b[groups_of == g],
            c[groups_of == g],
            tmpl[groups_of == g],
            alpha=alpha,
            margin=margin,
            p_adjusted=pa,
            n_boot=n_boot,
        )
        for g, pa in zip(labels, adjusted)
    ]
    return Comparison(
        baseline=baseline.name,
        candidate=candidate.name,
        overall=overall,
        groups=rows,
        dropped_api_errors=dropped,
        unmatched=unmatched,
        margin=margin,
        alpha=alpha,
    )


def format_comparison(cmp: Comparison) -> str:
    lines = [cmp.headline(), ""]
    header = f"{'group':<16} {'n':>4} {'base':>7} {'cand':>7} {'diff':>7}  {'95% CI':<17} {'W/L':>7} {'p(adj)':>8}  verdict"
    lines.append(header)
    lines.append("-" * len(header))
    for r in [cmp.overall, *cmp.groups]:
        ci = f"[{r.ci_low:+.0%}, {r.ci_high:+.0%}]"
        extra = f" (~{r.needed_pairs} needed)" if r.needed_pairs else ""
        lines.append(
            f"{r.group:<16} {r.n:>4} {r.baseline_rate:>7.1%} {r.candidate_rate:>7.1%} {r.difference:>+7.1%}  "
            f"{ci:<17} {f'{r.wins}/{r.losses}':>7} {r.p_adjusted:>8.3g}  {r.verdict}{extra}"
        )
    if cmp.dropped_api_errors:
        lines.append(f"\n{cmp.dropped_api_errors} pairs dropped because of API errors.")
    if cmp.unmatched:
        lines.append(f"{cmp.unmatched} episodes present in only one run were ignored.")
    return "\n".join(lines)
