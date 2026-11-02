"""Intervals for pass rates and latencies. Small n is the norm here, so: Wilson score
intervals for proportions, percentile bootstrap for medians and p90s (resampling whole
sessions when observations are clustered), exact McNemar for paired pass/fail."""

from __future__ import annotations

import math
import random
from statistics import median
from typing import Callable, Sequence

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def quantile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile (numpy's default)."""
    s = sorted(xs)
    if not s:
        return math.nan
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def p90(xs: Sequence[float]) -> float:
    return quantile(xs, 0.9)


def bootstrap(
    groups: Sequence[Sequence[float]],
    stat: Callable[[Sequence[float]], float] = median,
    b: int = 4000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI. `groups` are resampled as units (sessions);
    pass one-element groups for independent observations."""
    groups = [list(g) for g in groups if len(g)]
    flat = [x for g in groups for x in g]
    if not flat:
        return (math.nan, math.nan, math.nan)
    est = stat(flat)
    if len(groups) < 2:
        return (est, math.nan, math.nan)
    rng = random.Random(seed)
    k = len(groups)
    reps = []
    for _ in range(b):
        sample = [x for _ in range(k) for x in groups[rng.randrange(k)]]
        reps.append(stat(sample))
    a = (1 - level) / 2
    return (est, quantile(reps, a), quantile(reps, 1 - a))


def ratio_of_medians(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]], n: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """median(a) / median(b) with a bootstrap CI, resampling each side's groups."""
    a = [list(g) for g in a if len(g)]
    b = [list(g) for g in b if len(g)]
    fa = [x for g in a for x in g]
    fb = [x for g in b for x in g]
    if not fa or not fb or median(fb) == 0:
        return (math.nan, math.nan, math.nan)
    est = median(fa) / median(fb)
    rng = random.Random(seed)
    reps = []
    for _ in range(n):
        sa = [x for _ in a for x in a[rng.randrange(len(a))]]
        sb = [x for _ in b for x in b[rng.randrange(len(b))]]
        mb = median(sb)
        if mb:
            reps.append(median(sa) / mb)
    return (est, quantile(reps, 0.025), quantile(reps, 0.975))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p-value from discordant pair counts (b: A passed, B failed; c: the reverse)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_diff(pairs: Sequence[tuple[bool, bool]], b: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Difference in pass rate (A - B) over matched trials, with a paired bootstrap CI."""
    if not pairs:
        return (math.nan, math.nan, math.nan)
    d = [int(x) - int(y) for x, y in pairs]
    est = sum(d) / len(d)
    rng = random.Random(seed)
    n = len(d)
    reps = [sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(b)]
    return (est, quantile(reps, 0.025), quantile(reps, 0.975))
