"""
Episodic robustness framework: Section 3 and 4 of the paper.

An episode is a dict with keys:
  throughput   float  ops/s
  severity     float  ms/s  (total monitor-wait / episode_duration)
  duration     float  seconds

A contract is:
  thr_threshold  = gamma * T_base
  sev_threshold  = theta * S_base
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class Episode:
    throughput: float   # ops/s
    severity: float     # ms/s
    duration: float = 60.0


@dataclass
class Contract:
    thr_base: float     # T_base calibrated on clean baseline
    sev_base: float     # S_base calibrated on clean baseline
    gamma: float = 0.90  # throughput slack (require >= gamma * T_base)
    theta: float = 10.0  # severity slack   (require <= theta * S_base)


def episode_robustness(ep: Episode, contract: Contract) -> float:
    """
    Per-episode robustness margin rho_ep(phi, pi).

    phi = (thr >= gamma*T_base) AND (sev <= theta*S_base)
    Eq. (1) and conjunction rule Eq. (3).
    Returns min of the two component margins.
    """
    thr_thresh = contract.gamma * contract.thr_base
    sev_thresh = contract.theta * contract.sev_base

    rho_thr = ep.throughput - thr_thresh     # positive => thr OK
    rho_sev = sev_thresh - ep.severity       # positive => sev OK

    return min(rho_thr, rho_sev)             # Eq. (3) conjunction


def episodic_robustness_vector(episodes: Sequence[Episode], contract: Contract) -> list[float]:
    return [episode_robustness(ep, contract) for ep in episodes]


ACCEPT  = "ACCEPT"
REJECT  = "REJECT"
ABSTAIN = "ABSTAIN"


def order_statistic_verdict(
    robustness_margins: Sequence[float],
    k: int = 2,
) -> tuple[str, float, float, float]:
    """
    Section 4.1: order-statistic verdict procedure.

    Returns (verdict, r_k, r_N-k+1, coverage).
    """
    r = sorted(robustness_margins)
    N = len(r)
    if N < 2 * k:
        raise ValueError(f"Need N >= 2k; got N={N}, k={k}")

    r_low  = r[k - 1]          # r_(k)  (1-indexed)
    r_high = r[N - k]          # r_(N-k+1)

    cov = binomial_coverage(N, k)

    if r_low > 0:
        verdict = ACCEPT
    elif r_high < 0:
        verdict = REJECT
    else:
        verdict = ABSTAIN

    return verdict, r_low, r_high, cov


def binomial_coverage(N: int, k: int) -> float:
    """
    Theorem 1: coverage = 1 - 2 * sum_{i=0}^{k-1} C(N,i) * 2^{-N}
    """
    tail = sum(_comb(N, i) for i in range(k))
    return 1.0 - 2.0 * tail * (2 ** -N)


def _comb(n: int, k: int) -> int:
    return math.comb(n, k)


def calibrate_baseline(episodes: Sequence[Episode]) -> tuple[float, float]:
    """
    Phase 0: estimate T_base (median throughput) and S_base (median severity).
    """
    thrs = sorted(ep.throughput for ep in episodes)
    sevs = sorted(ep.severity   for ep in episodes)
    T_base = _median(thrs)
    S_base = _median(sevs)
    return T_base, S_base


def _median(vals: list[float]) -> float:
    n = len(vals)
    if n == 0:
        return 0.0
    m = n // 2
    return vals[m] if n % 2 else (vals[m - 1] + vals[m]) / 2.0
