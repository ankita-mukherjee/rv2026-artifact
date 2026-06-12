"""
Structural lock-smell classifier (Section 6.5).

Input:  per-episode JFR event lists (from jfr_parser.py)
Output: one of {SM, UL, SL, LIC, LOC, OS, UNKNOWN}

Detection rules follow Section 6.5 of the paper; thresholds were
calibrated on the held-out HBase/Glide set.
"""

from __future__ import annotations
from collections import Counter
from jfr_parser import compute_concentration

SMELL_SM  = "SM"   # Synchronized method dominating contention
SMELL_UL  = "UL"   # Unified lock – single lock serialising independent vars
SMELL_SL  = "SL"   # Same lock reused across unrelated methods
SMELL_LIC = "LIC"  # Loop inside critical section
SMELL_LOC = "LOC"  # Loop outside critical section (repeated lock acq/rel)
SMELL_OS  = "OS"   # Overly split adjacent synchronized blocks
SMELL_UNK = "UNKNOWN"

# ── calibrated thresholds (Section 6.5) ──────────────────────────────────────
TAU_CONC   = 0.80   # SM: concentration of blocked time in one frame
TAU_VAR    = 0.15   # SM: inter-episode severity CV (< 15% → stable = SM)
TAU_SM_MIN_EPS = 6  # SM: must hold for at least 6 of N episodes
TAU_LOC_PARK_RATE = 10.0  # LOC: park events per second (high = repeated lock/unlock)
TAU_OS_BLOCKS = 3   # OS: 3+ synchronized blocks in the same method


def classify(
    episode_events: list[list[dict]],
    episode_severities: list[float],
    source_hints: dict | None = None,
) -> str:
    """
    Classify a batch of episodes.

    episode_events     : list of event-lists, one per episode
    episode_severities : list of severity (ms/s) values, one per episode
    source_hints       : optional dict with AST metadata from source analysis
    """
    N = len(episode_events)
    if N == 0:
        return SMELL_UNK

    # ── SM detection ─────────────────────────────────────────────────────────
    conc_per_episode = [compute_concentration(evts) for evts in episode_events]
    sm_episodes = 0
    dominant_frames: list[str] = []
    for conc in conc_per_episode:
        if conc:
            top_frame, top_val = max(conc.items(), key=lambda x: x[1])
            if top_val > TAU_CONC:
                sm_episodes += 1
                dominant_frames.append(top_frame)

    mean_sev = sum(episode_severities) / N if N else 0
    std_sev  = (sum((s - mean_sev) ** 2 for s in episode_severities) / N) ** 0.5
    cv_sev   = std_sev / mean_sev if mean_sev > 0 else 0.0

    if sm_episodes >= TAU_SM_MIN_EPS and cv_sev < TAU_VAR:
        return SMELL_SM

    # ── UL / SL detection ────────────────────────────────────────────────────
    # UL: many distinct frames from a single monitor class (same lock object)
    monitor_classes: list[str] = []
    for evts in episode_events:
        for e in evts:
            if e["type"] == "JavaMonitorEnter" and e.get("monitor_class"):
                monitor_classes.append(e["monitor_class"])

    if monitor_classes:
        cls_counts = Counter(monitor_classes)
        dominant_cls, dom_count = cls_counts.most_common(1)[0]
        dom_fraction = dom_count / len(monitor_classes)

        # Collect distinct top frames under this dominant lock
        frames_under_dom: set[str] = set()
        for evts in episode_events:
            for e in evts:
                if (e["type"] == "JavaMonitorEnter"
                        and e.get("monitor_class") == dominant_cls
                        and e.get("frames")):
                    frames_under_dom.add(e["frames"][0])

        if dom_fraction > 0.8:
            if len(frames_under_dom) >= 3:
                # SL: same lock, many unrelated methods
                return SMELL_SL
            elif len(frames_under_dom) >= 2:
                # UL: same lock protecting independent vars
                return SMELL_UL

    # ── LIC / LOC detection ──────────────────────────────────────────────────
    park_counts = [
        sum(1 for e in evts if e["type"] == "ThreadPark")
        for evts in episode_events
    ]
    monitor_counts = [
        sum(1 for e in evts if e["type"] == "JavaMonitorEnter")
        for evts in episode_events
    ]

    total_parks   = sum(park_counts)
    total_monitor = sum(monitor_counts)
    duration_total = N * 60.0  # seconds

    if total_parks > 0 and total_parks / duration_total > TAU_LOC_PARK_RATE:
        if total_monitor > 0 and total_parks / max(total_monitor, 1) > 5:
            return SMELL_LOC  # many parks relative to monitor-enters
        return SMELL_LIC

    # ── OS detection ─────────────────────────────────────────────────────────
    if source_hints and source_hints.get("overly_split_count", 0) >= TAU_OS_BLOCKS:
        return SMELL_OS

    return SMELL_UNK


def classify_from_source(source_info: dict) -> str:
    """
    Classify based purely on source-level AST analysis (for pre-run classification).
    """
    synchronized_methods = source_info.get("synchronized_methods", 0)
    critical_sections    = source_info.get("critical_sections", [])
    loops_in_cs          = source_info.get("loops_in_critical_section", 0)
    loops_outside_cs     = source_info.get("loops_outside_critical_section", 0)
    split_count          = source_info.get("overly_split_count", 0)
    distinct_lock_vars   = source_info.get("distinct_lock_vars", 1)

    if loops_in_cs > 0:
        return SMELL_LIC
    if loops_outside_cs > 0:
        return SMELL_LOC
    if split_count >= TAU_OS_BLOCKS:
        return SMELL_OS
    if synchronized_methods >= 2 and distinct_lock_vars == 1:
        if len(set(cs.get("lock_obj") for cs in critical_sections)) <= 1:
            return SMELL_SL
        return SMELL_UL
    if synchronized_methods >= 1:
        return SMELL_SM
    return SMELL_UNK
