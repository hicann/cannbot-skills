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

"""Live monitor for the batch run.

    python scripts/batch/monitor.py <batch_dir>
        # auto-refreshing snapshot (default; Ctrl-C to stop)
    python scripts/batch/monitor.py <batch_dir> --dashboard
        # exec autoresearch's own dashboard.py on the active task (full TUI)

The view shows:
  - queue counts + visual progress bar
  - active task: phase, rounds, baseline/best/speedup, heartbeat age
  - active task: latest 3 history.jsonl decisions + plan.md head
  - tail of batch.log
  - speedup distribution across done ops
  - errored ops summary

For a static, copy-pasteable end-of-batch report use summarize.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from op_autoresearch.utils.console import emit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as mf

# Reach up one level (scripts/) for the shared settings accessors.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phase_machine import task_summary
from utils.settings import classify_speedup, recorded_speedup

logger = logging.getLogger(__name__)

DASHBOARD_PY = mf.repo_root() / "scripts" / "dashboard.py"


def fmt_metric(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _heartbeat_age(summary: dict) -> int | None:
    value = summary.get("last_touched")
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(value).timestamp()
    except (ValueError, TypeError) as exc:
        logger.debug("invalid task heartbeat timestamp %r: %s", value, exc)
        return None
    return int(time.time() - timestamp)


def _history_tail(task_dir: Path) -> list:
    path = task_dir / ".ar_state" / "history.jsonl"
    if not path.exists():
        return []
    try:
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [json.loads(line) for line in lines[-3:]]
    except (OSError, ValueError, TypeError):
        logger.debug("Could not read task history", exc_info=True)
        return []


def _plan_head(task_dir: Path) -> list[str]:
    path = task_dir / ".ar_state" / "plan.md"
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [line.rstrip() for line in lines if line.strip()][:12]
    except OSError:
        logger.debug("Could not read the task plan", exc_info=True)
        return []


def task_state(task_dir: Path) -> dict:
    """Snapshot of one task for the monitor TUI. Goes through
    phase_machine.task_summary so the schema has one owner; the
    monitor reads .ar_state/state.json
    directly and silently went blank on every field after both
    moved into state.json.
    """
    out: dict = {"task_dir": str(task_dir)}
    summary = task_summary(str(task_dir))
    if summary is None:
        out["phase"] = "UNKNOWN"
        return out

    out["phase"] = summary.get("phase") or "UNKNOWN"
    # Render zeros (not Nones) for the round counters so the existing
    # f-string formatting in render() doesn't show "None/None". When
    # progress hasn't been initialised yet we leave baseline/best out
    # entirely; the renderer already handles missing keys gracefully.
    out["eval_rounds"] = summary.get("eval_rounds") or 0
    out["max_rounds"] = summary.get("max_rounds") or 0
    out["consecutive_failures"] = summary.get("consecutive_failures") or 0
    out["plan_version"] = summary.get("plan_version") or 0
    # Expose baseline_outcome as `status` so render() can stay agnostic
    # of the underlying field name.
    if summary.get("baseline_outcome") is not None:
        out["status"] = summary.get("baseline_outcome")
    if summary.get("progress_initialized"):
        out["baseline_metric"] = summary.get("baseline_metric")
        out["best_metric"] = summary.get("best_metric")
        out["best_speedup"] = summary.get("best_speedup")

    # Heartbeat age — last_touched is the new single source of truth.
    heartbeat_age = _heartbeat_age(summary)
    if heartbeat_age is not None:
        out["heartbeat_age_s"] = heartbeat_age

    # history.jsonl / plan.md are external artifacts; task_summary
    # doesn't bundle them (they can be large). Read here directly,
    # gracefully.
    history_tail = _history_tail(task_dir)
    if history_tail:
        out["history_tail"] = history_tail
    plan_head = _plan_head(task_dir)
    if plan_head:
        out["plan_head"] = plan_head
    return out


def tail_lines(path: Path, n: int = 8) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            pos = size
            while pos > 0 and data.count(b"\n") <= n + 1:
                read = min(block, pos)
                pos -= read
                f.seek(pos)
                data = f.read(read) + data
            text = data.decode("utf-8", errors="replace")
        return text.splitlines()[-n:]
    except Exception:
        logger.debug("Could not read log tail %s", path, exc_info=True)
        return []


def render(
    batch_dir: Path,
    progress: dict,
    active: dict | None,
    log_tail: list[str],
) -> str:
    out: list[str] = []
    cases = progress.get("cases", {})
    _render_batch_header(out, batch_dir, progress, cases)
    _render_active_task(out, active)
    _render_log_tail(out, log_tail)
    _render_speedups(out, cases)
    _render_errors(out, cases)
    return "\n".join(out)


def _render_batch_header(
    out: list[str],
    batch_dir: Path,
    progress: dict,
    cases: dict,
) -> None:
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    out.extend([
        f"━━━ batch monitor  {now} ━━━",
        f"batch_dir  {batch_dir}",
        f"mode={progress.get('mode', '?')}",
        "",
    ])
    counts = _case_counts(cases)
    bar = (
        "█" * counts["done"]
        + "▶" * counts["running"]
        + "▒" * counts["error"]
        + "·" * counts["skip"]
        + " " * counts["pending"]
    )
    out.append(
        f"queue   total={sum(counts.values()):3d}  "
        f"done={counts['done']:3d}  error={counts['error']:3d}  "
        f"skip={counts['skip']:3d}  pending={counts['pending']:3d}  "
        f"running={counts['running']:3d}"
    )
    out.append(f"        [{bar}]")
    done_secs, running_secs = _elapsed_seconds(cases)
    total_secs = done_secs + running_secs
    out.extend([
        f"elapsed done={done_secs/60:.1f}min  "
        f"current_op={running_secs/60:.1f}min  "
        f"total={total_secs/60:.1f}min ({total_secs/3600:.2f}h)",
        "",
    ])


def _case_counts(cases: dict) -> dict[str, int]:
    counts = {"done": 0, "error": 0, "skip": 0, "pending": 0, "running": 0}
    for case in cases.values():
        status = case.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _elapsed_seconds(cases: dict) -> tuple[float, float]:
    done_secs = 0.0
    running_secs = 0.0
    now = datetime.now(timezone.utc).astimezone().replace(tzinfo=None)
    for case in cases.values():
        started_at = case.get("started_at")
        if not started_at:
            continue
        try:
            started = datetime.fromisoformat(started_at)
            if case.get("finished_at"):
                finished = datetime.fromisoformat(case["finished_at"])
                done_secs += (finished - started).total_seconds()
            elif case.get("status") == "running":
                running_secs += (now - started).total_seconds()
        except (TypeError, ValueError):
            continue
    return done_secs, running_secs


def _render_active_task(out: list[str], active: dict | None) -> None:
    if active is None:
        out.extend(["active  (no task in ar_tasks/)", ""])
        return
    out.append(f"active  {Path(active['task_dir']).name}")
    out.append(
        f"        phase={active.get('phase', '?')}  "
        f"rounds={active.get('eval_rounds', '?')}/"
        f"{active.get('max_rounds', '?')}  "
        f"failures={active.get('consecutive_failures', 0)}  "
        f"plan_v={active.get('plan_version', 0)}  "
        f"status={active.get('status', '?')}"
    )
    _render_active_metrics(out, active)
    _render_history(out, active.get("history_tail") or [])
    _render_plan(out, active.get("plan_head") or [])
    out.append("")


def _render_active_metrics(out: list[str], active: dict) -> None:
    baseline = active.get("baseline_metric")
    best = active.get("best_metric")
    speedup = recorded_speedup(active)
    if speedup is not None:
        out.append(
            f"        baseline={fmt_metric(baseline)}  "
            f"best={fmt_metric(best)}  speedup={speedup:.2f}x"
        )
    elif baseline is not None or best is not None:
        out.append(f"        baseline={baseline}  best={best}")
    heartbeat = active.get("heartbeat_age_s")
    if heartbeat is not None:
        stale = " (STALE)" if heartbeat > 300 else ""
        out.append(f"        heartbeat: {heartbeat}s ago{stale}")


def _render_history(out: list[str], history: list[dict]) -> None:
    if not history:
        return
    out.extend(["", "        history (last 3 rounds):"])
    for record in history:
        metrics = record.get("metrics") or {}
        metric = next(
            (
                f" {name}={fmt_metric(metrics[name])}"
                for name in ("latency_us", "metric")
                if name in metrics
            ),
            "",
        )
        correctness = (
            ""
            if record.get("correctness") is None
            else f" correct={record['correctness']}"
        )
        description = (record.get("description") or "")[:50]
        out.append(
            f"          R{record.get('round', '?'):>2} "
            f"{record.get('decision', '?')}{metric}{correctness}  {description}"
        )


def _render_plan(out: list[str], plan_head: list[str]) -> None:
    if not plan_head:
        return
    out.extend(["", "        plan.md head:"])
    out.extend(f"          {line[:90]}" for line in plan_head[:8])


def _render_log_tail(out: list[str], log_tail: list[str]) -> None:
    if not log_tail:
        return
    out.append("batch.log (last 6 lines):")
    out.extend(f"  {line[:100]}" for line in log_tail)
    out.append("")


def _render_speedups(out: list[str], cases: dict) -> None:
    speedups = []
    for case in cases.values():
        if case.get("status") != "done":
            continue
        speedup = recorded_speedup(case.get("result") or {})
        if speedup is not None:
            speedups.append(speedup)
    if not speedups:
        return
    labels = [classify_speedup(value) for value in speedups]
    out.append(
        f"done speedup  median={statistics.median(speedups):.2f}x  "
        f"best={max(speedups):.2f}x  worst={min(speedups):.2f}x  "
        f"(n={len(speedups)})"
    )
    out.append(
        f"              improved={labels.count('improved')}  "
        f"on-par={labels.count('on-par')}  "
        f"regress={labels.count('regress')}"
    )


def _render_errors(out: list[str], cases: dict) -> None:
    errored = [
        (name, case)
        for name, case in cases.items()
        if case.get("status") == "error"
    ]
    if not errored:
        return
    out.extend(["", f"errored ops ({len(errored)}):"])
    for name, case in errored[:5]:
        out.append(f"  - {name}: {(case.get('note') or '')[:80]}")
    if len(errored) > 5:
        out.append(f"  ... and {len(errored) - 5} more")


def clear_screen() -> None:
    os.system("cls" if sys.platform == "win32" else "clear")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_dir")
    ap.add_argument("-n", "--interval", type=int, default=15,
                    help="refresh interval in seconds (default: 15)")
    ap.add_argument("--dashboard", action="store_true",
                    help="exec autoresearch's dashboard.py on the active task")
    ap.add_argument("--task-dir", default="",
                    help="for --dashboard: explicit task_dir (default: most recent)")
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    if not batch_dir.is_dir():
        emit(f"batch dir not found: {batch_dir}", file=sys.stderr)
        return 2

    if args.dashboard:
        return _launch_dashboard(batch_dir, args.task_dir)

    _configure_console()
    return _watch_batch(batch_dir, args.interval)


def _launch_dashboard(batch_dir: Path, task_dir_arg: str) -> int:
    task_dir = (
        Path(task_dir_arg).resolve()
        if task_dir_arg
        else mf.find_running_case_task_dir(batch_dir)
    )
    if task_dir is None:
        emit(
            "no running case has a bound task_dir yet; "
            "pass --task-dir <path> to attach explicitly",
            file=sys.stderr,
        )
        return 2
    if not DASHBOARD_PY.exists():
        emit(f"dashboard.py not found at {DASHBOARD_PY}", file=sys.stderr)
        return 2
    emit(f"[monitor] launching autoresearch dashboard on {task_dir}")
    cmd = [sys.executable, str(DASHBOARD_PY), str(task_dir), "--watch", "5"]
    os.execvp(cmd[0], cmd)
    return 0


def _configure_console() -> None:
    if sys.platform != "win32":
        return
    try:
        reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
        if callable(reconfigure_stdout):
            reconfigure_stdout(encoding="utf-8", errors="replace")
    except OSError:
        logger.debug("Could not reconfigure monitor output", exc_info=True)


def _watch_batch(batch_dir: Path, interval: int) -> int:
    log_path = batch_dir / mf.LOG_FILENAME
    try:
        while True:
            body = _render_once(batch_dir, log_path)
            footer = (
                f"\n(refresh every {interval}s; Ctrl-C to stop  |  "
                "full TUI: monitor.py --dashboard  |  "
                "static report: summarize.py)"
            )
            clear_screen()
            emit(body + footer, flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        emit()
        return 0


def _render_once(batch_dir: Path, log_path: Path) -> str:
    progress = mf.load_progress(batch_dir)
    active_dir = mf.find_running_case_task_dir(batch_dir)
    active = task_state(active_dir) if active_dir else None
    return render(batch_dir, progress, active, tail_lines(log_path, n=6))


if __name__ == "__main__":
    sys.exit(main())
