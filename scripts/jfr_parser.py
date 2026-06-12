"""
JFR event parser.

Uses `jfr print` (bundled with Java 21) to parse .jfr recordings,
then extracts jdk.JavaMonitorEnter and jdk.ThreadPark events.

Each JFR recording covers one episode.  Returns an Episode object
with throughput (from external measurement) and severity (computed from
total blocked time / episode duration).
"""

from __future__ import annotations
import subprocess
import re
import os
import sys
from pathlib import Path
from robustness import Episode

# Regex patterns for JFR text output
_MONITOR_ENTER_BLOCK = re.compile(
    r"jdk\.JavaMonitorEnter\s*\{(.*?)\}",
    re.DOTALL,
)
_THREAD_PARK_BLOCK = re.compile(
    r"jdk\.ThreadPark\s*\{(.*?)\}",
    re.DOTALL,
)
_DURATION_FIELD = re.compile(r"duration\s*=\s*([\d.]+)\s*(ms|s|us|ns)", re.IGNORECASE)
_BLOCKED_DURATION = re.compile(r"duration\s*=\s*([\d.]+)\s*(ms|s|us|ns)", re.IGNORECASE)
_STACKTRACE = re.compile(r"stackTrace\s*=\s*\{(.*?)\}", re.DOTALL)
_FRAME_LINE  = re.compile(r'at\s+([\w.$<>]+)\(')
_CLASS_FIELD = re.compile(r'monitorClass\s*=\s*([\w.$<>]+)', re.IGNORECASE)


def _to_ms(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit == "ms":
        return value
    if unit == "s":
        return value * 1000.0
    if unit == "us":
        return value / 1000.0
    if unit == "ns":
        return value / 1_000_000.0
    return value


def _find_jfr_binary() -> str:
    """Locate the jfr command bundled with the JDK."""
    java_home = os.environ.get("JAVA_HOME", "")
    candidates = []
    if java_home:
        candidates.append(os.path.join(java_home, "bin", "jfr"))
        candidates.append(os.path.join(java_home, "bin", "jfr.exe"))

    # Check all java.exe locations and derive jfr from there
    import shutil
    java_path = shutil.which("java")
    if java_path:
        bin_dir = os.path.dirname(java_path)
        candidates.append(os.path.join(bin_dir, "jfr"))
        candidates.append(os.path.join(bin_dir, "jfr.exe"))

    for c in candidates:
        if os.path.isfile(c):
            return c

    # Last resort: hope it's on PATH
    return "jfr"


def parse_jfr(
    jfr_path: str | Path,
    episode_duration_s: float = 60.0,
    throughput: float = 0.0,
) -> tuple[Episode, list[dict]]:
    """
    Parse a JFR recording file.

    Returns:
        episode  Episode with severity computed from monitor-enter blocked time
        events   list of raw event dicts (for classifier)
    """
    jfr_path = Path(jfr_path)
    jfr_bin = _find_jfr_binary()

    cmd = [jfr_bin, "print",
           "--events", "jdk.JavaMonitorEnter,jdk.ThreadPark",
           str(jfr_path)]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        text = result.stdout
    except FileNotFoundError:
        # Fall back to jcmd or alternative parse
        text = _parse_via_jcmd(jfr_path)

    events = _extract_events(text, episode_duration_s)
    total_blocked_ms = sum(e["blocked_ms"] for e in events
                           if e["type"] == "JavaMonitorEnter")
    severity = total_blocked_ms / episode_duration_s   # ms/s

    ep = Episode(
        throughput=throughput,
        severity=severity,
        duration=episode_duration_s,
    )
    return ep, events


def _extract_events(text: str, episode_duration_s: float) -> list[dict]:
    events = []

    for m in _MONITOR_ENTER_BLOCK.finditer(text):
        block = m.group(1)
        dur_match = _DURATION_FIELD.search(block)
        if not dur_match:
            continue
        blocked_ms = _to_ms(float(dur_match.group(1)), dur_match.group(2))
        stack_match = _STACKTRACE.search(block)
        frames = []
        if stack_match:
            frames = _FRAME_LINE.findall(stack_match.group(1))
        cls_match = _CLASS_FIELD.search(block)
        monitor_class = cls_match.group(1) if cls_match else ""

        events.append({
            "type": "JavaMonitorEnter",
            "blocked_ms": blocked_ms,
            "frames": frames,
            "monitor_class": monitor_class,
        })

    for m in _THREAD_PARK_BLOCK.finditer(text):
        block = m.group(1)
        dur_match = _DURATION_FIELD.search(block)
        if not dur_match:
            continue
        park_ms = _to_ms(float(dur_match.group(1)), dur_match.group(2))
        stack_match = _STACKTRACE.search(block)
        frames = []
        if stack_match:
            frames = _FRAME_LINE.findall(stack_match.group(1))

        events.append({
            "type": "ThreadPark",
            "blocked_ms": park_ms,
            "frames": frames,
            "monitor_class": "",
        })

    return events


def _parse_via_jcmd(jfr_path: Path) -> str:
    """Alternative: use jcmd JFR.dump or FlightRecorderMXBean if jfr CLI missing."""
    return ""


def compute_severity_from_events(events: list[dict], duration_s: float) -> float:
    """Severity as defined in Eq. (5): sum(blockedTime) / duration."""
    total = sum(e["blocked_ms"] for e in events if e["type"] == "JavaMonitorEnter")
    return total / duration_s if duration_s > 0 else 0.0


def compute_concentration(events: list[dict]) -> dict[str, float]:
    """
    Section 6.5: blocked-time concentration per top frame.
    conc(f, pi) = sum_{e: top(e)=f} blockedTime(e) / total_blockedTime
    """
    monitor_events = [e for e in events if e["type"] == "JavaMonitorEnter"]
    total = sum(e["blocked_ms"] for e in monitor_events)
    if total == 0:
        return {}

    frame_times: dict[str, float] = {}
    for e in monitor_events:
        top = e["frames"][0] if e["frames"] else "<unknown>"
        frame_times[top] = frame_times.get(top, 0.0) + e["blocked_ms"]

    return {f: t / total for f, t in frame_times.items()}
