"""
Generate all paper tables from experiment results.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from collections import defaultdict
from robustness import binomial_coverage


def load_results(results_dir: str) -> list[dict]:
    results = []
    for f in Path(results_dir).glob("result_*.json"):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Table 2: Fault Corpus
# ─────────────────────────────────────────────────────────────────────────────

def table2(corpus: list[dict]) -> str:
    """Reproduce Table 2: Historical fault corpus."""
    from collections import Counter

    by_proj = defaultdict(list)
    for f in corpus:
        by_proj[f["project"]].append(f)

    lines = ["Table 2: Historical fault corpus. Each row is a pairwise evaluation unit.",
             f"{'Subject':<22} {'Version':<10} {'Pairs':>5}  {'Dominant Smells'}"]
    lines.append("-" * 60)

    proj_info = {
        "cassandra": "4.1.x",
        "kafka":     "3.x",
        "jetty":     "12.x",
        "hbase":     "2.5.x",
    }
    total = 0
    all_smells = Counter()
    for proj in ["cassandra", "kafka", "jetty", "hbase"]:
        faults = by_proj[proj]
        if not faults:
            continue
        n = len(faults)
        total += n
        smells = Counter(f["smell"] for f in faults)
        all_smells.update(smells)
        smell_str = ", ".join(f"{s} ({c})" for s, c in smells.most_common())
        subj_name = f"Apache {proj.capitalize()}" if proj != "jetty" else "Eclipse Jetty"
        lines.append(f"{subj_name:<22} {proj_info[proj]:<10} {n:>5}  {smell_str}")

    lines.append("-" * 60)
    all_s_str = ", ".join(f"{s} ({c})" for s, c in all_smells.most_common())
    lines.append(f"{'Total':<22} {'':10} {total:>5}  {all_s_str}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 3: Aggregate verdict accuracy
# ─────────────────────────────────────────────────────────────────────────────

def _confusion_metrics(tp: int, fp: int, fn: int, tn: int, abstain: int = 0) -> tuple[float, float, float]:
    # Precision/recall/F1 are computed over DECIDED cases only (ACCEPT/REJECT).
    # ABSTAIN is not a classification error and is reported as its own column,
    # not folded into the recall denominator. An abstaining 3-way monitor that
    # makes zero false negatives therefore scores recall 1.0, while the count of
    # abstentions quantifies, separately, how often it declined to decide.
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    return prec, recall, f1


def table3(results: list[dict]) -> str:
    """Reproduce Table 3: Aggregate verdict accuracy."""
    methods = ["threshold", "percentile", "ewma", "cusum", "changepoint", "ours"]
    method_labels = {
        "threshold":   "Threshold",
        "percentile":  "Percentile",
        "ewma":        "EWMA",
        "cusum":       "CUSUM",
        "changepoint": "Changepoint",
        "ours":        f"Ours (N={results[0].get('N',8)}, k={results[0].get('k',2)})" if results else "Ours (N=8, k=2)",
    }

    stats = {m: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "abstain": 0} for m in methods}

    for r in results:
        expected_reject = (r.get("expected") == "REJECT")

        for m in methods:
            verdict = r.get(f"verdict_{m}", "")
            if verdict == "ABSTAIN":
                stats[m]["abstain"] += 1
                continue
            predicted_reject = (verdict == "REJECT")
            if expected_reject and predicted_reject:
                stats[m]["tp"] += 1
            elif not expected_reject and predicted_reject:
                stats[m]["fp"] += 1
            elif expected_reject and not predicted_reject:
                stats[m]["fn"] += 1
            else:
                stats[m]["tn"] += 1

    lines = [
        "Table 3: Aggregate verdict accuracy. Precision/Recall treat REJECT as positive.",
        f"{'Method':<28} {'Prec.':>6} {'Recall':>7} {'F1':>6} {'ABSTAIN':>8} {'FP':>4} {'FN':>4}",
        "-" * 70,
    ]
    n_total = len(results)
    for m in methods:
        s = stats[m]
        prec, rec, f1 = _confusion_metrics(s["tp"], s["fp"], s["fn"], s["tn"], s["abstain"])
        abstain_pct = f"{100*s['abstain']//n_total}%" if n_total > 0 else "-%"
        bold = "**" if m == "ours" else ""
        lines.append(
            f"{bold}{method_labels[m]:<28}{bold} {prec:>6.2f} {rec:>7.2f} {f1:>6.2f} "
            f"{abstain_pct:>8} {s['fp']:>4} {s['fn']:>4}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 4: Representative cases
# ─────────────────────────────────────────────────────────────────────────────

def table4(results: list[dict], representative_ids: list[str] | None = None) -> str:
    """Reproduce Table 4: Representative cases."""
    if representative_ids is None:
        representative_ids = [r["case_id"] for r in results[:6]]

    lines = [
        "Table 4: Representative cases. Robustness in throughput units (ops/s).",
        f"{'Subject':<10} {'Case':<6} {'Issue':<25} {'Expected':<10} {'Verdict':<10} {'[r(k), r(N-k+1)]'}",
        "-" * 85,
    ]
    for r in results:
        if r["case_id"] not in representative_ids:
            continue
        lo = r.get("buggy_r_low", 0.0)
        hi = r.get("buggy_r_high", 0.0)
        lines.append(
            f"{r['project']:<10} {r['case_id']:<6} {r['issue']:<25} "
            f"{r['expected']:<10} {r.get('verdict_ours', '?'):<10} "
            f"[{lo:>8.1f}, {hi:>8.1f}]"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 5: Overhead
# ─────────────────────────────────────────────────────────────────────────────

def table5(overhead_results: list[dict]) -> str:
    """Reproduce Table 5: Throughput overhead."""
    lines = [
        "Table 5: Throughput overhead relative to no-JFR baseline. Bootstrap 95% CIs.",
        f"{'Subject':<14} {'Baseline (k ops/s)':>20} {'JFR default':>12} {'JFR 0ms':>10} {'Overhead 95% CI':>18}",
        "-" * 80,
    ]
    for r in overhead_results:
        lines.append(
            f"{r['subject']:<14} {r['baseline_kops']:>20.1f} "
            f"{r['jfr_default_kops']:>12.1f} {r['jfr_0ms_kops']:>10.1f} "
            f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}]%"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 6: Parameter sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def table6(sensitivity_results: list[dict]) -> str:
    lines = [
        "Table 6: Sensitivity of F1 and wall-clock cost to N and k.",
        f"{'N':>4} {'k':>3} {'Coverage':>10} {'F1':>6} {'Median cost (min)':>18}",
        "-" * 45,
    ]
    for r in sensitivity_results:
        cov = binomial_coverage(r["N"], r["k"])
        lines.append(
            f"{r['N']:>4} {r['k']:>3} {cov:>10.3f} {r['f1']:>6.2f} {r['median_cost']:>18}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 7: Autocorrelation
# ─────────────────────────────────────────────────────────────────────────────

def table7(autocorr_results: list[dict]) -> str:
    lines = [
        "Table 7: Lag-1 autocorrelation of per-episode throughput on clean baselines.",
        f"{'Subject':<14} {'rho_1':>8} {'p-value':>10} {'N episodes':>12}",
        "-" * 48,
    ]
    for r in autocorr_results:
        lines.append(
            f"{r['subject']:<14} {r['rho1']:>8.2f} {r['pvalue']:>10.2f} {r['n_episodes']:>12}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 8: Ablation
# ─────────────────────────────────────────────────────────────────────────────

def table8(ablation_results: list[dict]) -> str:
    lines = [
        "Table 8: Ablation study.",
        f"{'Configuration':<42} {'Precision':>10} {'Recall':>8} {'F1':>6}",
        "-" * 70,
    ]
    for r in ablation_results:
        lines.append(
            f"{r['config']:<42} {r['precision']:>10.2f} {r['recall']:>8.2f} {r['f1']:>6.2f}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 9: Repair demonstration
# ─────────────────────────────────────────────────────────────────────────────

def table9(repair_results: list[dict]) -> str:
    lines = [
        "Table 9: Repair demonstration outcomes on REJECT cases.",
        f"{'Subject':<14} {'REJECTs':>8} {'Flip':>10} {'Func.pass':>12} {'End-to-end':>12}",
        "-" * 60,
    ]
    totals = {"rejects": 0, "flip": 0, "func_pass": 0, "end_to_end": 0}
    for r in repair_results:
        lines.append(
            f"{r['subject']:<14} {r['rejects']:>8} "
            f"{r['flip']:>4} ({100*r['flip']//r['rejects'] if r['rejects'] else 0}%){'':<5} "
            f"{r['func_pass']:>4} ({100*r['func_pass']//r['flip'] if r['flip'] else 0}%){'':<5} "
            f"{r['end_to_end']:>4} ({100*r['end_to_end']//r['rejects'] if r['rejects'] else 0}%)"
        )
        totals["rejects"]    += r["rejects"]
        totals["flip"]       += r["flip"]
        totals["func_pass"]  += r["func_pass"]
        totals["end_to_end"] += r["end_to_end"]

    lines.append("-" * 60)
    re, fl, fp, e2e = totals["rejects"], totals["flip"], totals["func_pass"], totals["end_to_end"]
    lines.append(
        f"{'Overall':<14} {re:>8} "
        f"{fl:>4} ({100*fl//re if re else 0}%){'':<5} "
        f"{fp:>4} ({100*fp//fl if fl else 0}%){'':<5} "
        f"{e2e:>4} ({100*e2e//re if re else 0}%)"
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 11: Complete per-case results
# ─────────────────────────────────────────────────────────────────────────────

def table11(results: list[dict]) -> str:
    lines = [
        "Table 11: Complete per-case results for the historical fault corpus.",
        f"{'ID':<5} {'Subject':<12} {'Issue':<25} {'Type':<5} {'Expected':<10} {'Verdict':<10} {'[r(k), r(N-k+1)]'}",
        "-" * 95,
    ]
    for r in results:
        lo = r.get("buggy_r_low", 0.0)
        hi = r.get("buggy_r_high", 0.0)
        lines.append(
            f"{r['case_id']:<5} {r['project'].capitalize():<12} {r['issue']:<25} "
            f"{r.get('smell','?'):<5} {r['expected']:<10} "
            f"{r.get('verdict_ours','?'):<10} [{lo:>9.1f}, {hi:>9.1f}]"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 12: Classifier confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def table12(confusion: dict) -> str:
    smells = ["SM", "UL", "SL", "LIC", "LOC", "OS"]
    lines = [
        "Table 12: Classifier confusion matrix (actual rows vs. predicted columns).",
        f"{'':>6} " + " ".join(f"{s:>5}" for s in smells) + f"{'Total':>8}",
        "-" * 52,
    ]
    for actual in smells:
        row = confusion.get(actual, {})
        vals = [row.get(pred, 0) for pred in smells]
        total = sum(vals)
        lines.append(f"{actual:<6} " + " ".join(f"{v:>5}" for v in vals) + f"{total:>8}")
    lines.append("-" * 52)
    col_totals = [sum(confusion.get(a, {}).get(p, 0) for a in smells) for p in smells]
    grand_total = sum(col_totals)
    lines.append(f"{'Total':<6} " + " ".join(f"{v:>5}" for v in col_totals) + f"{grand_total:>8}")
    return "\n".join(lines)
