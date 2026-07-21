# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Print a human-readable report of <batch_dir>/batch_progress.json.

Designed for the "after the batch is done, what happened?" view — distinct
from monitor.py which reads ar_tasks/ live state. Static, fast, copy-pasteable.

Usage:
    python scripts/batch/summarize.py <batch_dir>
"""
from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from op_autoresearch.utils.console import emit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as mf

# Reach up one level (scripts/) for the shared settings accessors.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.settings import (
    classify_speedup,
    recorded_speedup,
    speedup_improved_above,
    speedup_regress_below,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dir")
    args = parser.parse_args()
    batch_dir = Path(args.batch_dir).resolve()
    if not batch_dir.is_dir():
        emit(f"batch dir not found: {batch_dir}", file=sys.stderr)
        return 2

    progress = mf.load_progress(batch_dir)
    cases = progress.get("cases", {})
    if not cases:
        emit(f"no cases recorded in {batch_dir / mf.PROGRESS_FILENAME}")
        return 1
    by_status = _group_cases(cases)
    _print_header(batch_dir, progress, by_status, len(cases))
    speedups, no_metric = _collect_speedups(by_status.get("done", []))
    _print_speedup_summary(speedups)
    _print_regressions(speedups)
    _print_missing_metrics(no_metric)
    _print_status_details(by_status)
    return 0


def _group_cases(cases: dict) -> dict[str, list[tuple[str, dict]]]:
    grouped: dict[str, list[tuple[str, dict]]] = {
        "done": [],
        "error": [],
        "skip": [],
        "pending": [],
        "running": [],
    }
    for name, case in cases.items():
        grouped.setdefault(case.get("status", "pending"), []).append((name, case))
    return grouped


def _print_header(
    batch_dir: Path,
    progress: dict,
    by_status: dict[str, list[tuple[str, dict]]],
    total: int,
) -> None:
    now = datetime.now(timezone.utc).astimezone().replace(tzinfo=None)
    emit(f"batch summary  ({now.isoformat(timespec='seconds')})")
    emit(f"batch_dir  {batch_dir}")
    emit(f"mode={progress.get('mode', '?')}")
    emit("─" * 60)
    emit(f"  total:    {total}")
    for status in ("done", "error", "skip", "pending", "running"):
        bucket = by_status.get(status, [])
        if bucket or status in ("done", "error"):
            emit(f"  {status:8s}: {len(bucket)}")
    emit()


def _collect_speedups(
    done_cases: list[tuple[str, dict]],
) -> tuple[list[tuple[str, float, float, float]], list[str]]:
    speedups: list[tuple[str, float, float, float]] = []
    no_metric: list[str] = []
    for name, case in done_cases:
        result = case.get("result") or {}
        speedup = recorded_speedup(result)
        if speedup is None:
            no_metric.append(name)
            continue
        speedups.append(
            (
                name,
                speedup,
                result.get("baseline_metric"),
                result.get("best_metric"),
            )
        )
    return speedups, no_metric


def _print_speedup_summary(
    speedups: list[tuple[str, float, float, float]],
) -> None:
    if not speedups:
        return
    values = [speedup for _, speedup, _, _ in speedups]
    upper = speedup_improved_above()
    lower = speedup_regress_below()
    labels = [classify_speedup(value) for value in values]
    emit("speedup (best_speedup geomean; higher better):")
    emit(f"  ops with metric: {len(speedups)}")
    emit(f"  median:          {statistics.median(values):.2f}x")
    emit(f"  best:            {max(values):.2f}x")
    emit(f"  worst:           {min(values):.2f}x")
    emit(f"  improved:        {labels.count('improved')}  (>{upper}x)")
    emit(f"  on-par:          {labels.count('on-par')}    ({lower}-{upper}x)")
    emit(f"  regress:         {labels.count('regress')}     (<{lower}x)")
    emit()


def _print_regressions(
    speedups: list[tuple[str, float, float, float]],
) -> None:
    regressions = [
        row
        for row in speedups
        if row[1] < speedup_regress_below()
    ]
    if not regressions:
        return
    emit(f"regressions ({len(regressions)} ops slower than baseline):")
    for name, speedup, baseline, best in sorted(
        regressions,
        key=lambda row: row[1],
    ):
        emit(
            f"  - {name}: baseline {baseline:.3f} -> "
            f"best {best:.3f}  ({speedup:.2f}x)"
        )
    emit()


def _print_missing_metrics(no_metric: list[str]) -> None:
    if not no_metric:
        return
    emit(f"done but no metric extracted ({len(no_metric)}):")
    for name in no_metric[:8]:
        emit(f"  - {name}")
    if len(no_metric) > 8:
        emit(f"  ... and {len(no_metric) - 8} more")
    emit()


def _print_status_details(
    by_status: dict[str, list[tuple[str, dict]]],
) -> None:
    _print_notices("errored ops", by_status.get("error", []), _error_note)
    _print_notices("skipped ops", by_status.get("skip", []), _plain_note)
    running = by_status.get("running", [])
    if running:
        emit(f"running (likely stale; batch died mid-op): {len(running)}")
        for name, case in running:
            emit(f"  - {name}: started_at={case.get('started_at', '?')}")
        emit()
    pending = by_status.get("pending", [])
    if pending:
        emit(f"still pending: {len(pending)}")
        for name, _ in pending[:10]:
            emit(f"  - {name}")
        if len(pending) > 10:
            emit(f"  ... and {len(pending) - 10} more")


def _print_notices(title: str, cases: list[tuple[str, dict]], formatter) -> None:
    if not cases:
        return
    emit(f"{title} ({len(cases)}):")
    for name, case in cases:
        emit(f"  - {name}: {formatter(case)}")
    emit()


def _error_note(case: dict) -> str:
    note = (case.get("note") or "(no note)")[:80]
    return f"phase={case.get('final_phase', '?')}  {note}"


def _plain_note(case: dict) -> str:
    return (case.get("note") or "(no note)")[:80]


if __name__ == "__main__":
    sys.exit(main())
