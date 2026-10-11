"""Statistics for comparing two planners without fooling yourself.

Copied from abyyworld/vla-evals (``src/policy_evals/stats.py``; see
VENDORED.md), where it compared robot policies. Everything down to
``required_episodes`` is unchanged. Added below it, for this project: Wilson
intervals for single success rates, a cluster bootstrap (episodes from one
instruction template are not independent), a paired sample-size estimate, and
Holm's correction for testing every category at once.

The original notes, with "policy" meaning "planner configuration" here:

The default way labs compare checkpoints is: run both, look at the two success
rates, ship the bigger one. With 50 episodes, a 6-point difference in success
rate is well inside noise, and that comparison will promote a worse model
roughly as often as it promotes a better one.

Three things here fix that:

**Paired evaluation.** Both policies face identical seeds, so each episode is a
matched pair. Pairing removes episode difficulty from the comparison entirely,
and typically shrinks the confidence interval on the difference by a large
factor versus treating the two runs as independent samples.

**Paired bootstrap over the difference**, not over each arm separately. The
question is "is B better than A", so the quantity to put an interval around is
B − A.

**McNemar's exact test** for the binary outcomes. With paired binary data, only
the *discordant* pairs — the ones where the policies disagree — carry
information. The episodes both solved and both failed tell you nothing about
which is better, and any test that counts them is throwing away power and
overstating certainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def __str__(self) -> str:
        return f"{self.point:+.4f} [{self.low:+.4f}, {self.high:+.4f}]"


def bootstrap_mean_ci(
    values: np.ndarray, n_samples: int = 10_000, seed: int = 17, alpha: float = 0.05
) -> Interval:
    """Percentile CI for a mean."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    if values.size == 1:
        v = float(values[0])
        return Interval(v, v, v)

    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_samples, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(values.mean()), float(lo), float(hi))


def paired_bootstrap_diff(
    baseline: np.ndarray,
    candidate: np.ndarray,
    n_samples: int = 10_000,
    seed: int = 17,
    alpha: float = 0.05,
) -> Interval:
    """CI for mean(candidate) − mean(baseline) over matched episodes.

    Resamples *pairs*, never the two arms independently. Resampling
    independently would discard the pairing and reintroduce the episode-difficulty
    variance that pairing exists to remove.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if baseline.shape != candidate.shape:
        raise ValueError(
            f"paired comparison needs equal lengths, got {baseline.shape} and {candidate.shape}"
        )
    return bootstrap_mean_ci(candidate - baseline, n_samples=n_samples, seed=seed, alpha=alpha)


def mcnemar_exact(baseline: np.ndarray, candidate: np.ndarray) -> tuple[int, int, float]:
    """Two-sided exact McNemar test on paired binary outcomes.

    Returns `(wins, losses, p_value)` where a *win* is an episode the candidate
    solved and the baseline did not.

    Exact binomial rather than the chi-squared approximation: benchmark suites
    routinely produce fewer than 25 discordant pairs, which is exactly where the
    approximation stops being trustworthy — and it errs toward reporting
    significance that is not there.
    """
    baseline = np.asarray(baseline).astype(bool)
    candidate = np.asarray(candidate).astype(bool)
    if baseline.shape != candidate.shape:
        raise ValueError("paired test needs equal lengths")

    wins = int(np.sum(candidate & ~baseline))
    losses = int(np.sum(~candidate & baseline))
    n = wins + losses
    if n == 0:
        # The policies never disagreed. That is not evidence they are equal, and
        # p = 1.0 is the honest answer rather than a small number.
        return wins, losses, 1.0

    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return wins, losses, float(min(1.0, 2.0 * tail))


def required_episodes(
    baseline_rate: float, detectable_difference: float, power: float = 0.8, alpha: float = 0.05
) -> int:
    """Roughly how many episodes are needed to detect a given improvement.

    A normal-approximation power calculation for two proportions. Approximate on
    purpose — its job is to answer "is 50 episodes anywhere near enough?" before
    a week is spent running a benchmark that could never have resolved the
    effect. It usually is not.
    """
    from math import sqrt

    # Inverse normal CDF via the Beasley-Springer-Moro style rational
    # approximation would be overkill; these two z values cover the standard
    # alpha/power choices and the function documents its own bluntness.
    z_alpha = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758}.get(alpha, 1.9600)
    z_power = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}.get(power, 0.8416)

    p1 = float(np.clip(baseline_rate, 1e-6, 1 - 1e-6))
    p2 = float(np.clip(baseline_rate + detectable_difference, 1e-6, 1 - 1e-6))
    p_bar = 0.5 * (p1 + p2)

    numerator = z_alpha * sqrt(2 * p_bar * (1 - p_bar)) + z_power * sqrt(
        p1 * (1 - p1) + p2 * (1 - p2)
    )
    return int(np.ceil((numerator / max(abs(p2 - p1), 1e-9)) ** 2))


# ---------------------------------------------------------------------------
# Added for language-to-robot-actions


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> Interval:
    """Wilson score interval for a single success rate.

    The normal-approximation interval collapses to zero width at 0% and 100%,
    which are exactly the rates an easy category or an oracle produces.
    """
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    z = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758}.get(alpha, 1.9600)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    # Clamp around the point: at p = 0 or 1, rounding can otherwise leave the
    # bound a hair on the wrong side of the estimate.
    return Interval(p, min(p, max(0.0, centre - half)), max(p, min(1.0, centre + half)))


def paired_cluster_bootstrap_diff(
    baseline: np.ndarray,
    candidate: np.ndarray,
    clusters: np.ndarray,
    n_samples: int = 10_000,
    seed: int = 17,
    alpha: float = 0.05,
) -> Interval:
    """CI for mean(candidate) - mean(baseline), resampling whole clusters of pairs.

    Ten seeds of one instruction template share its wording and its difficulty,
    so they are not ten independent observations. Resampling episodes treats
    them as if they were and gives an interval that is too narrow. Resampling
    templates (with all their seeds) does not.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    clusters = np.asarray(clusters)
    if not (baseline.shape == candidate.shape == clusters.shape):
        raise ValueError("baseline, candidate and clusters must align")
    diff = candidate - baseline
    labels = np.unique(clusters)
    if labels.size < 2:
        return bootstrap_mean_ci(diff, n_samples=n_samples, seed=seed, alpha=alpha)
    sums = np.array([diff[clusters == c].sum() for c in labels])
    sizes = np.array([(clusters == c).sum() for c in labels], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, labels.size, size=(n_samples, labels.size))
    draws = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(diff.mean()), float(lo), float(hi))


def required_pairs(
    discordant_rate: float, difference: float, power: float = 0.8, alpha: float = 0.05
) -> int:
    """Episodes needed for McNemar's test to detect ``difference`` (Connor, 1987).

    For matched pairs the sample size depends on how often the two conditions
    *disagree*, not on their success rates: two planners that agree on 95% of
    episodes need far more episodes to separate than their rates suggest.
    """
    from math import sqrt

    z_alpha = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758}.get(alpha, 1.9600)
    z_power = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}.get(power, 0.8416)
    d = max(abs(difference), 1e-6)
    pd = min(1.0, max(discordant_rate, d))  # disagreement can never be below the difference
    n = (z_alpha * sqrt(pd) + z_power * sqrt(max(pd - d**2, 0.0))) ** 2 / d**2
    return int(np.ceil(n))


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order.

    Testing six categories at alpha = 0.05 gives roughly a one-in-four chance
    that at least one "significant" category is noise. Holm controls that and
    is uniformly more powerful than plain Bonferroni.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()
