"""
ARM compatibility test: 5x aggregate run of benchmark_transformers.py and
benchmark_litespark.py with /usr/bin/time RSS capture.

This is a portable adaptation of the script ARM shipped in their test zip.
Behavioural differences from the original:

  * No `bash -lc` wrapper. Each subprocess uses sys.executable directly,
    which preserves the parent's Python interpreter (whatever venv /
    conda env / system Python the user invoked us with). The original's
    `bash -lc` reset PATH and broke installs that lived in venvs not on
    a login shell's PATH.
  * No hard-coded venv activation path -- this script is now portable to
    any clone of the repo on macOS or Linux, no edits required.
  * Cross-platform `time(1)` flags: `/usr/bin/time -l` on macOS, `-v` on
    Linux. Both produce a "maximum resident set size" line that the RSS
    regex below parses (the GNU variant is in kB; the Darwin variant is
    in bytes -- normalised to MB at report time).

Aggregation, prompts, RUNS, log/csv output -- all unchanged from the
ARM-supplied original.
"""

import csv
import os
import platform
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RUNS = 5
LOG_DIR = Path("benchmark_logs")
LOG_DIR.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"benchmark_repeat_{timestamp}.log"
CSV_FILE = LOG_DIR / f"benchmark_repeat_{timestamp}.csv"


def _time_prefix() -> list[str]:
    """Return the platform-appropriate `/usr/bin/time` prefix, or [] if no
    suitable time(1) is found (in which case RSS won't be parsed but tok/s
    still will be)."""
    sys_name = platform.system()
    candidates = {
        "Darwin": [["/usr/bin/time", "-l"]],
        "Linux":  [["/usr/bin/time", "-v"]],
    }.get(sys_name, [])
    for cand in candidates:
        if Path(cand[0]).exists():
            return cand
    return []


_TIME_PREFIX = _time_prefix()
_PYTHON = sys.executable
_HERE = Path(__file__).resolve().parent


COMMANDS = {
    "transformers": _TIME_PREFIX + [_PYTHON, "-u", str(_HERE / "benchmark_transformers.py")],
    "litespark":    _TIME_PREFIX + [_PYTHON, "-u", str(_HERE / "benchmark_litespark.py")],
}

TOK_RE = re.compile(r"Generated\s+(\d+)\s+tokens\s+in\s+([0-9.]+)s\s+\(([0-9.]+)\s+tok/s\)")

# macOS `/usr/bin/time -l` reports "maximum resident set size" in BYTES.
# GNU `/usr/bin/time -v` reports "Maximum resident set size (kbytes):" in KB.
RSS_DARWIN_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$", re.MULTILINE)
RSS_GNU_RE    = re.compile(r"^\s*Maximum resident set size \(kbytes\):\s+(\d+)\s*$", re.MULTILINE)


def _parse_rss_bytes(output: str) -> "int | None":
    m = RSS_DARWIN_RE.search(output)
    if m:
        return int(m.group(1))
    m = RSS_GNU_RE.search(output)
    if m:
        return int(m.group(1)) * 1024
    return None


results = {}


def log(text: str = "") -> None:
    print(text)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def median(values):
    return statistics.median(values)


def bytes_to_gb(n):
    return n / (1024 ** 3)


log(f"Benchmark started: {datetime.now().isoformat()}")
log(f"Saving log to: {LOG_FILE}")
log(f"Saving CSV to: {CSV_FILE}")
log(f"Python: {_PYTHON}")
log(f"Platform: {platform.system()} {platform.machine()}")
log(f"time(1) prefix: {_TIME_PREFIX or '(none -- RSS will be missing)'}")
log("")

rows = []

for name, cmd in COMMANDS.items():
    runs = []
    log(f"=== {name} ===")
    for i in range(1, RUNS + 1):
        # On Linux GNU time, default output goes to stderr; merge both
        # streams so the regex sees everything.
        proc = subprocess.run(cmd, capture_output=True, text=True)
        output = proc.stdout + proc.stderr

        log("")
        log(f"--- run {i} ---")
        log(output.rstrip())

        tok_match = TOK_RE.search(output)
        rss_bytes = _parse_rss_bytes(output)

        if not tok_match:
            raise RuntimeError(
                f"Could not parse token stats for {name} run {i} "
                f"(subprocess returncode={proc.returncode}). "
                f"Output:\n{output[-1000:]}"
            )
        if rss_bytes is None and _TIME_PREFIX:
            raise RuntimeError(f"Could not parse RSS for {name} run {i}")

        tokens = int(tok_match.group(1))
        seconds = float(tok_match.group(2))
        tok_s = float(tok_match.group(3))

        run = {
            "backend": name,
            "run": i,
            "tokens": tokens,
            "seconds": seconds,
            "tok_s": tok_s,
            "rss_bytes": rss_bytes if rss_bytes is not None else 0,
            "rss_gb": bytes_to_gb(rss_bytes) if rss_bytes is not None else 0.0,
        }
        runs.append(run)
        rows.append(run)

    results[name] = runs

with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["backend", "run", "tokens", "seconds", "tok_s", "rss_bytes", "rss_gb"],
    )
    writer.writeheader()
    writer.writerows(rows)

log("")
log("=== Median summary ===")
summary = {}
for name, runs in results.items():
    med_tokens = median([r["tokens"] for r in runs])
    med_seconds = median([r["seconds"] for r in runs])
    med_tok_s = median([r["tok_s"] for r in runs])
    med_rss = median([r["rss_bytes"] for r in runs])

    summary[name] = {
        "tokens": med_tokens,
        "seconds": med_seconds,
        "tok_s": med_tok_s,
        "rss_bytes": med_rss,
    }

    log(
        f"{name:12s} "
        f"tokens={med_tokens:.0f} "
        f"time={med_seconds:.2f}s "
        f"tok/s={med_tok_s:.2f} "
        f"rss={bytes_to_gb(med_rss):.2f} GB"
    )

log("")
log("=== Aggregate summary ===")
aggregate = {}
for name, runs in results.items():
    total_tokens = sum(r["tokens"] for r in runs)
    total_seconds = sum(r["seconds"] for r in runs)
    agg_tok_s = total_tokens / total_seconds if total_seconds else 0.0
    med_rss = median([r["rss_bytes"] for r in runs])

    aggregate[name] = {
        "tokens": total_tokens,
        "seconds": total_seconds,
        "tok_s": agg_tok_s,
        "rss_bytes": med_rss,
    }

    log(
        f"{name:12s} "
        f"total_tokens={total_tokens} "
        f"total_time={total_seconds:.2f}s "
        f"agg_tok/s={agg_tok_s:.2f} "
        f"median_rss={bytes_to_gb(med_rss):.2f} GB"
    )

tf = aggregate["transformers"]
ls = aggregate["litespark"]

speedup = ls["tok_s"] / tf["tok_s"] if tf["tok_s"] else float("inf")
memory_ratio = tf["rss_bytes"] / ls["rss_bytes"] if ls["rss_bytes"] else float("inf")
memory_saved_gb = bytes_to_gb(tf["rss_bytes"] - ls["rss_bytes"]) if ls["rss_bytes"] else 0.0

log("")
log("=== Aggregate comparison ===")
log(f"Litespark speedup vs transformers: {speedup:.2f}x")
log(f"Litespark memory reduction vs transformers: {memory_ratio:.2f}x")
log(f"Absolute median RSS saved: {memory_saved_gb:.2f} GB")
log("")
log(f"Done. Full log saved to: {LOG_FILE}")
log(f"CSV saved to: {CSV_FILE}")
