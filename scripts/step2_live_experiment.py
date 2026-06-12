#!/usr/bin/env python3
"""
step2_live_experiment.py
========================
Runs a LIVE experiment for a single fault pair using the real system.

This script requires:
  - Java 21 JDK with `jfr` on PATH or JAVA_HOME set
  - The target project (Kafka / Cassandra / Jetty / HBase) compiled and
    accessible in REPO_ROOT/projects/<project>/
  - The project-specific workload tool (kafka-producer-perf-test, etc.)

For running all experiments, use the full orchestration script:
    python ../src/run_experiments.py --case K1

This script provides a minimal single-case runner that can be used
for spot-checking individual fault pairs.

Usage:
    python step2_live_experiment.py --case K2 [--N 8] [--k 2]

See ../src/run_experiments.py for the full multi-project orchestration.
"""

import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR   = REPO_ROOT / "src"
DATA_DIR  = REPO_ROOT / "data"

sys.path.insert(0, str(SRC_DIR))

try:
    from run_experiments import run_kafka_fault_experiment, load_corpus
except ImportError as e:
    print(f"ERROR: {e}")
    print(f"Ensure {SRC_DIR}/run_experiments.py and its dependencies exist.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run a live experiment for one fault pair."
    )
    parser.add_argument("--case",  required=True, help="Case ID, e.g. K2")
    parser.add_argument("--N",     type=int, default=8, help="Number of episodes")
    parser.add_argument("--k",     type=int, default=2, help="Rank parameter k")
    parser.add_argument("--warmup",   type=int, default=30,  help="Warmup seconds")
    parser.add_argument("--duration", type=int, default=60,  help="Episode seconds")
    args = parser.parse_args()

    try:
        corpus = load_corpus()
    except FileNotFoundError:
        corpus = []
        # Try reading from reproduce/data/
        import json
        fp = DATA_DIR / "fault_pairs_classified.json"
        if fp.exists():
            data = json.loads(fp.read_text())
            corpus = data.get("fault_pairs", [])

    faults = [f for f in corpus if f.get("id") == args.case
              or f.get("case_id") == args.case]

    if not faults:
        print(f"Case {args.case!r} not found in corpus.")
        print(f"Available cases: {[f.get('id') or f.get('case_id') for f in corpus[:10]]}")
        sys.exit(1)

    fault = faults[0]
    project = fault.get("project", "kafka")

    if project == "kafka":
        result = run_kafka_fault_experiment(
            fault, N=args.N, k=args.k,
            warmup_s=args.warmup, duration_s=args.duration,
        )
        print(f"\nVerdict: {result.get('verdict_ours', '?')}")
        lo = result.get("buggy_r_low", 0.0)
        hi = result.get("buggy_r_high", 0.0)
        print(f"Interval: [{lo:.1f}, {hi:.1f}]")
    else:
        print(f"Project {project!r} runner not implemented in this wrapper.")
        print(f"Use ../src/run_experiments.py for full project support.")
        sys.exit(1)


if __name__ == "__main__":
    main()
