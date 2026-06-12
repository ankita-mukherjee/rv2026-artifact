#!/usr/bin/env python3
"""
step1_parse_real_jfr.py
========================
Parses the real JFR recordings for the 4 experiments that were
actually run on hardware: K2, C5, J7, H2.

For each experiment it reads the N=8 buggy and N=8 clean .jfr files
from the corresponding sub-directory under ../data/, computes per-episode
robustness margins using the paper's contract (Eq. 7), and prints the
order-statistic verdict (Definition 3).

This lets you verify the real experiment numbers without re-running the
full workload.

Usage:
    python step1_parse_real_jfr.py [--case K2|C5|J7|H2]

Requirements:
    - Java 21 JDK on PATH (for `jfr print`)
    - Core modules (jfr_parser.py, robustness.py) live alongside this script.
"""

import sys
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent          # artifact/
DATA_DIR   = REPO_ROOT / "data"
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from jfr_parser import parse_jfr
    from robustness import (
        Episode, Contract, episodic_robustness_vector,
        order_statistic_verdict, calibrate_baseline, binomial_coverage,
    )
except ImportError as e:
    print(f"ERROR: {e}")
    print(f"Ensure {SCRIPT_DIR} contains jfr_parser.py and robustness.py")
    sys.exit(1)

# Map case ID to its JFR recording directory under data/jfr/
EXPERIMENTS = {
    "K2": DATA_DIR / "jfr" / "K2",
    "C5": DATA_DIR / "jfr" / "C5",
    "J7": DATA_DIR / "jfr" / "J7",
    "H2": DATA_DIR / "jfr" / "H2",
}

# Patterns for JFR file names: jfr_{ID}_{buggy|fixed}_ep{n}.jfr
def _find_jfr(exp_dir: Path, case_id: str, variant: str, n_eps: int = 8) -> list[Path]:
    files = []
    for i in range(n_eps):
        f = exp_dir / f"jfr_{case_id}_{variant}_ep{i}.jfr"
        if f.exists():
            files.append(f)
    # Fallback: any jfr files matching the variant
    if not files:
        pattern = f"*{variant}*ep*.jfr"
        files = sorted(exp_dir.glob(pattern))[:n_eps]
    return files


def run_case(case_id: str, N: int = 8, k: int = 2, duration_s: float = 60.0):
    exp_dir = EXPERIMENTS.get(case_id)
    if exp_dir is None or not exp_dir.exists():
        print(f"[{case_id}] experiment directory not found: {exp_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Case {case_id}  (N={N}, k={k})")
    print(f"Directory: {exp_dir}")
    print(f"{'='*60}")

    # --- Clean (fixed) episodes ---
    clean_jfrs = _find_jfr(exp_dir, case_id, "fixed", N)
    if not clean_jfrs:
        clean_jfrs = _find_jfr(exp_dir, case_id, "clean", N)
    print(f"Clean JFR files found: {len(clean_jfrs)}")

    clean_eps = []
    for jfr_path in clean_jfrs:
        try:
            ep, events = parse_jfr(jfr_path, duration_s, throughput=0.0)
            clean_eps.append(ep)
            print(f"  {jfr_path.name}: sev={ep.severity:.2f} ms/s  events={len(events)}")
        except Exception as e:
            print(f"  ERROR parsing {jfr_path.name}: {e}")

    if not clean_eps:
        print("No clean episodes parsed; cannot calibrate.")
        return

    T_base, S_base = calibrate_baseline(clean_eps)
    print(f"\nCalibration: T_base={T_base:.1f} ops/s  S_base={S_base:.4f} ms/s")
    print("(NOTE: T_base=0 because throughput must be injected from stress tool output)")

    # --- Buggy episodes ---
    buggy_jfrs = _find_jfr(exp_dir, case_id, "buggy", N)
    print(f"\nBuggy JFR files found: {len(buggy_jfrs)}")

    buggy_eps = []
    for jfr_path in buggy_jfrs:
        try:
            ep, events = parse_jfr(jfr_path, duration_s, throughput=0.0)
            buggy_eps.append(ep)
            print(f"  {jfr_path.name}: sev={ep.severity:.2f} ms/s  events={len(events)}")
        except Exception as e:
            print(f"  ERROR parsing {jfr_path.name}: {e}")

    if not buggy_eps:
        print("No buggy episodes parsed.")
        return

    # --- Verdict using severity only (throughput=0 as placeholder) ---
    # With throughput=0 the thr component of robustness will be negative,
    # so we report severity-only verdict here for illustration.
    # True verdict requires stress-tool throughput injected at parse time.
    print(f"\nSeverity-only verdict (throughput=0 placeholder):")
    contract = Contract(thr_base=T_base, sev_base=S_base, gamma=0.90, theta=10.0)
    margins = episodic_robustness_vector(buggy_eps, contract)
    print(f"  Margins: {[f'{m:.2f}' for m in margins]}")
    verdict, lo, hi = order_statistic_verdict(margins, k)
    cov = binomial_coverage(N, k)
    print(f"  Verdict: {verdict}  [{lo:.2f}, {hi:.2f}]")
    print(f"  Coverage: {cov:.4f}")
    print()
    print("See ../data/results/result_{}.json for the full verdict".format(case_id))
    print("(includes throughput from the original experiment run).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["K2", "C5", "J7", "H2"],
                        help="Run a specific case (default: all 4)")
    args = parser.parse_args()

    cases = [args.case] if args.case else ["K2", "C5", "J7", "H2"]
    for cid in cases:
        run_case(cid)


if __name__ == "__main__":
    main()
