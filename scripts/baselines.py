"""
Baseline anomaly detection methods (Section 7.1).

All baselines receive the same N=8 per-episode throughput values and
calibration data as our episodic framework.  They return ACCEPT/REJECT/ABSTAIN.
"""

from __future__ import annotations
import math
from typing import Sequence
from robustness import ACCEPT, REJECT, ABSTAIN


def threshold_baseline(
    thrs: Sequence[float],
    T_base: float,
    gamma: float = 0.90,
) -> str:
    """Reject if any episode throughput < gamma * T_base."""
    limit = gamma * T_base
    for t in thrs:
        if t < limit:
            return REJECT
    return ACCEPT


def percentile_baseline(
    thrs: Sequence[float],
    T_base: float,
    p: float = 5.0,
    gamma: float = 0.90,
) -> str:
    """Reject if p-th percentile of throughput < gamma * T_base."""
    sorted_t = sorted(thrs)
    idx = max(0, int(math.ceil(p / 100.0 * len(sorted_t))) - 1)
    pct = sorted_t[idx]
    return REJECT if pct < gamma * T_base else ACCEPT


def ewma_baseline(
    thrs: Sequence[float],
    T_base: float,
    alpha: float = 0.2,
    L: float = 3.0,
) -> str:
    """
    Roberts (1959) EWMA control chart.
    Reject if EWMA statistic falls below the lower control limit.
    UCL/LCL = mu ± L * sigma_ewma, where sigma is estimated from baseline.
    We estimate sigma from the throughput sequence variance w.r.t. T_base.
    """
    mu = T_base
    # estimate sigma as std dev of sequence from baseline mean
    if len(thrs) < 2:
        return ABSTAIN
    var = sum((t - mu) ** 2 for t in thrs) / (len(thrs) - 1)
    sigma = math.sqrt(var) if var > 0 else 1.0
    sigma_ewma = sigma * math.sqrt(alpha / (2 - alpha))
    lcl = mu - L * sigma_ewma

    ewma = mu
    for t in thrs:
        ewma = alpha * t + (1 - alpha) * ewma
        if ewma < lcl:
            return REJECT
    return ACCEPT


def cusum_baseline(
    thrs: Sequence[float],
    T_base: float,
    k_slack: float = 0.5,
    h_threshold: float = 5.0,
) -> str:
    """
    Page (1954) one-sided CUSUM for negative drift.
    Detects when cumulative sum of (mu - X_i) exceeds threshold h.
    k_slack: allowance parameter (in units of std dev).
    h_threshold: decision threshold.
    """
    mu = T_base
    if len(thrs) < 2:
        return ABSTAIN
    var = sum((t - mu) ** 2 for t in thrs) / (len(thrs) - 1)
    sigma = math.sqrt(var) if var > 0 else 1.0

    k = k_slack * sigma
    S = 0.0
    for t in thrs:
        S = max(0.0, S + (mu - t) - k)
        if S > h_threshold * sigma:
            return REJECT
    return ACCEPT


def changepoint_baseline(
    thrs: Sequence[float],
    T_base: float,
    gamma: float = 0.90,
    min_size: int = 2,
) -> tuple[str, bool]:
    """
    Bayesian online changepoint detection (Adams & MacKay 2007) — simplified.
    We use a PELT-style energy cost detection via ruptures if available,
    falling back to a manual sliding-window approach.

    Returns (verdict, detected_changepoint).
    """
    n = len(thrs)
    if n < 4:
        return ABSTAIN, False

    try:
        import ruptures as rpt
        model = rpt.Pelt(model="rbf", min_size=min_size).fit(
            [t for t in thrs]
        )
        breakpoints = model.predict(pen=2.0)
        has_cp = len(breakpoints) > 1 or (len(breakpoints) == 1 and breakpoints[0] < n)
    except Exception:
        # Fallback: compare first half mean to second half mean
        mid = n // 2
        mean_first = sum(thrs[:mid]) / mid
        mean_second = sum(thrs[mid:]) / (n - mid)
        has_cp = (mean_first - mean_second) > 0.05 * T_base

    if has_cp:
        # Check if post-changepoint mean is below threshold
        try:
            import ruptures as rpt
            mid = breakpoints[0] if breakpoints[0] < n else n // 2
        except Exception:
            mid = n // 2
        post_mean = sum(thrs[mid:]) / max(1, n - mid)
        if post_mean < gamma * T_base:
            return REJECT, True
    return ACCEPT, has_cp
