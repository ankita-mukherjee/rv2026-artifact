# Artifact: Runtime Monitoring of Lock Contention in Non-Stationary JVM Executions

Reproduction package for the RV 2026 paper: monitor, fault corpus, real JFR
recordings, pre-computed results, and the lock-smell repair demonstration.

---

## Scripts — what each one does

All under `scripts/`. The first two are runnable entry points; the rest are
imported modules.

| File | Role |
|------|------|
| `step1_parse_jfr.py` | Entry point. Parses the real `.jfr` recordings under `data/jfr/{K2,C5,J7,H2}/` and prints per-episode severity + the order-statistic verdict. Needs Java 21 `jfr` on `PATH`. |
| `step2_live_experiment.py` | Entry point (optional). Runs a live JVM episode against an installed subject. Requires the full project repo (`src/run_experiments.py`) + compiled subject + workload tool. |
| `robustness.py` | Core verdict logic: `Episode`, `Contract`, episodic robustness vector, order-statistic verdict, baseline calibration, binomial coverage (Sections 4–5). |
| `jfr_parser.py` | JFR event parser for `jdk.JavaMonitorEnter` and `jdk.ThreadPark`; turns a `.jfr` file into an `Episode`. |
| `baselines.py` | SPC baselines compared against in the paper: EWMA, CUSUM, Changepoint, Threshold, Percentile. |
| `classifier.py` | Structural lock-smell classifier assigning SM / UL / LIC / LOC / OS labels. |
| `tables.py` | Table-formatting helpers. |

---

## Data

Under `data/`:

| Path | Contents |
|------|----------|
| `corpus/fault_corpus.json` | 30 historical fault pairs: `case_id`, `project`, `fix_sha`, `buggy_sha`, `github_commit_url`, `issue_url`, `smell_type` (SM/UL/LIC/LOC/OS), `real_jfr`. |
| `corpus/workload_config.json` | Per-subject workload parameters (Table 2). |
| `results/result_*.json` | Pre-computed episode data + verdicts for all 30 fault pairs. |
| `jfr/{K2,C5,H2,J7}/` | Real JFR recordings for the 4 hardware experiments: 8 buggy + 8 clean episodes each. |
| `jfr/overhead/` | Raw JFR overhead recordings (4 subjects × 2 configs). Provenance only — not read by any script. |
| `overhead_summary.json` | Aggregated overhead numbers reported in Section 6.2. |

Case-ID gaps (K3, J3–J4, C1–C4, H8) are issues excluded during corpus
construction (non-compilable, non-contention, or workload-incompatible);
see Section 5.1.

---

## Configuration

`scripts/rv2026.jfc` — JFR configuration enabling `jdk.JavaMonitorEnter` and
`jdk.ThreadPark` at a 0 ms threshold (every event captured). Apply with:

```bash
-XX:+FlightRecorder -XX:StartFlightRecording=filename=out.jfr,settings=scripts/rv2026.jfc,dumponexit=true
```

JFR buffers: global 256 MB, thread-local 16 MB (prevents ring-buffer saturation
under high event volume).

Verdict parameters:

| Parameter        | Value     | Notes                                   |
|------------------|-----------|-----------------------------------------|
| N                | 8         | Episodes per verdict                    |
| k                | 2         | Rank parameter                          |
| Coverage         | ≈ 92.97%  | Theorem 1: exact binomial               |
| Episode duration | 60 s      | Post warm-up                            |
| Warm-up discard  | 30 s      | Per episode                             |
| Calibration      | 3×MAD     | Applied to clean fixing-commit baseline |

---

## Dependencies

- **Python 3.9+** — stdlib only, no `pip install` required (`subprocess`, `re`).
- **Java JDK with the `jfr` CLI** — for `step1`/`step2`. Details below.
- **Ollama + Qwen 2.5 Coder 32B** — only for the repair demonstration below.

### JFR tooling (for `step1_parse_jfr.py`)

`step1` shells out to the JDK's `jfr` command and regex-parses its text output:

```bash
jfr print --events jdk.JavaMonitorEnter,jdk.ThreadPark <file>.jfr
```

- **Needs a full JDK, not a JRE.** The `jfr` binary ships only with the JDK
  (`$JDK/bin/jfr`); a JRE does not include it.
- **Version: JDK 21 recommended; JDK ≥ 14 works.** The bundled recordings are
  JFR chunk format **v2.1**, readable by any `jfr print` from JDK 14 onward.
  Recordings were produced on OpenJDK 17 and parsed on JDK 21 — both fine.
  Avoid JDK 11 (may not read format 2.1).
- **Discovery** (`jfr_parser._find_jfr_binary`): tries `$JAVA_HOME/bin/jfr`,
  then `jfr` beside `java` on `PATH`, then bare `jfr`. Set `JAVA_HOME` or put
  `java` on `PATH` and it resolves automatically.
- **Events used:** `jdk.JavaMonitorEnter` (contended `synchronized` entry; its
  `duration` = blocked time → severity) and `jdk.ThreadPark`
  (`LockSupport.park`, used by `ReentrantLock`; fed to the classifier).

---

## Qwen Coder prompts (repair demonstration)

The repair stage is a feasibility demonstration, not an evaluated contribution.
On a `REJECT` verdict, the classifier's smell label plus JFR evidence are passed
to a locally hosted code model; a patch is accepted only if (i) the contract
verdict flips to `ACCEPT` and (ii) the functional test suite passes.

Model configuration:

| Setting     | Value                          |
|-------------|--------------------------------|
| Model       | Qwen 2.5 Coder 32B             |
| Host        | Ollama (local)                 |
| Temperature | 0                              |
| Accept rule | verdict flips to ACCEPT + functional tests pass |

Prompt structure passed to the model (smell label, JFR evidence, and buggy
source are substituted per case):

```
System:
You are a Java concurrency expert. You repair lock-contention defects without
changing program semantics. Output only the corrected method body.

User:
A runtime monitor flagged a lock-contention regression in this method.
Lock-smell classification: {SMELL_LABEL}        # one of SM, UL, LIC, LOC, OS
JFR evidence:
{JFR_EVIDENCE}                                  # contended monitors, blocked-time
                                                # concentration, parking hotspots
Buggy method:
{BUGGY_SOURCE}

Produce a corrected version that removes the contention while preserving the
method's observable behavior and thread safety.
```

Note: the verbatim per-case prompts are reconstructed from the paper's
description (Section: Repair Demonstration); the repair driver itself is not part
of this artifact, since it depends on a local Ollama instance and the compiled
subjects.
