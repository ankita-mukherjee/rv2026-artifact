# Artifact: Runtime Monitoring of Lock Contention in Non-Stationary JVM Executions

CASCON 2025 / RV 2026 paper artifact.

---

## Contents

```
artifact/
├── README.md                        — this file
├── scripts/                         — all runnable code + core monitor modules
│   ├── reproduce_all.py             — Reproduces all paper tables from pre-computed data
│   ├── step1_parse_real_jfr.py      — Parses real JFR recordings (data/jfr/) into verdicts
│   ├── step2_live_experiment.py     — Runs a live JVM episode (optional; needs full repo + subject)
│   ├── robustness.py                — Episode, Contract, order-statistic verdict (Sections 4–5)
│   ├── jfr_parser.py                — JFR event parser (jdk.JavaMonitorEnter, jdk.ThreadPark)
│   ├── baselines.py                 — EWMA, CUSUM, Changepoint, Threshold, Percentile baselines
│   ├── classifier.py                — Lock-smell classifier (SM, UL, LIC, LOC, OS)
│   ├── tables.py                    — Table-printing utilities
│   └── rv2026.jfc                   — JFR config: jdk.JavaMonitorEnter + jdk.ThreadPark at 0ms
└── data/
    ├── corpus/
    │   ├── fault_corpus.json        — 30 historical faults: commit SHAs, issue URLs, smell labels
    │   └── workload_config.json     — Per-subject workload parameters (Table 2)
    ├── results/
    │   └── result_*.json            — Pre-computed episode data for all 30 fault pairs
    ├── jfr/
    │   ├── K2/                      — Real JFR recordings: Kafka K2 (8 buggy + 8 clean episodes)
    │   ├── C5/                      — Real JFR recordings: Cassandra C5
    │   ├── H2/                      — Real JFR recordings: HBase H2
    │   ├── J7/                      — Real JFR recordings: Jetty J7
    │   └── overhead/                — JFR overhead recordings (all 4 subjects, 2 configs)
    └── overhead_summary.json        — Aggregated overhead measurements (Section 6.2)
```

---

## Requirements

- Python 3.9+ — stdlib only, no external packages.
- Java 21 JDK on `PATH` (provides `jfr print`) — only for `step1`/`step2`, not for `reproduce_all`.

---

## Step 0: Reproduce all paper tables from pre-computed data

No live JVM required. Reads `data/results/result_*.json` and prints every paper table.

```bash
cd artifact
python scripts/reproduce_all.py
```

Expected output (~5 s): Theorem 1 coverage, Tables 2–4, RQ3 ablation, autocorrelation
table, per-case appendix, and a final match check — all matching the paper:

```
Prec=1.00  Recall=0.77  F1=0.87  FP=0  FN=0  ABSTAIN=7  Coverage=0.9297
```

---

## Step 1: Parse the real JFR recordings

Re-derives per-episode severity and the order-statistic verdict directly from the
pre-recorded `.jfr` files for the 4 experiments run on hardware (K2, C5, J7, H2).
No subject installation required.

```bash
python scripts/step1_parse_real_jfr.py            # all 4 cases
python scripts/step1_parse_real_jfr.py --case C5  # one case
```

Reads `data/jfr/{K2,C5,H2,J7}/`. Throughput is injected at experiment time, so this
script reports the severity-only verdict; see `data/results/result_*.json` for the
full throughput-aware verdict.

---

## Step 2: Live experiment (optional)

Runs live JVM episodes against a locally installed subject using `scripts/rv2026.jfc`.
Requires the full project repository (`src/run_experiments.py`) plus a compiled subject
and its workload tool — out of scope for table reproduction.

```bash
python scripts/step2_live_experiment.py --case K2 --N 8 --k 2
```

---

## JFR Configuration

`scripts/rv2026.jfc` enables `jdk.JavaMonitorEnter` and `jdk.ThreadPark` at 0 ms
threshold (all events captured). Apply with:

```bash
-XX:+FlightRecorder -XX:StartFlightRecording=filename=out.jfr,settings=scripts/rv2026.jfc,dumponexit=true
```

JFR buffers: global 256 MB, thread-local 16 MB (prevents ring-buffer saturation under
high event volume).

---

## Fault Corpus

`data/corpus/fault_corpus.json` contains all 30 fault pairs with:
- `case_id`, `project`, `fix_sha`, `buggy_sha` — commit hashes
- `github_commit_url`, `issue_url` — issue tracker links
- `smell_type` — SM / UL / LIC / LOC / OS
- `real_jfr` — true for K2, C5, H2, J7

Gaps in case ID numbering (K3, J3, J4, C1–C4, H8) reflect issues excluded during corpus
construction (non-compilable, non-contention, or workload-incompatible); see Section 5.1
of the paper.

---

## Verdict Parameters

| Parameter        | Value     | Notes                                    |
|------------------|-----------|------------------------------------------|
| N                | 8         | Episodes per verdict                     |
| k                | 2         | Rank parameter                           |
| Coverage         | ≈ 92.97%  | Theorem 1: exact binomial                |
| Episode duration | 60 s      | Post warm-up                             |
| Warm-up discard  | 30 s      | Per episode                              |
| Calibration      | 3×MAD     | Applied to clean fixing-commit baseline  |
