#!/usr/bin/env python3
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

"""Live dashboard for autoresearch progress."""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from op_autoresearch.utils.console import emit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_machine as _pm
from utils.json_io import _read_whole_file as _shared_read_whole_file
from utils.json_io import load_jsonl as _shared_load_jsonl
from utils.settings import default_max_rounds as _default_max_rounds
from utils.settings import recorded_speedup

logger = logging.getLogger(__name__)


if sys.platform == "win32":
    import msvcrt

    def read_key_nonblocking():
        """Return a navigation key name, or ``None`` when no key is ready."""
        if not msvcrt.kbhit():
            return None
        key = msvcrt.getch()
        if key in (b"\x00", b"\xe0"):
            if not msvcrt.kbhit():
                return None
            return {
                b"H": "UP", b"P": "DOWN", b"I": "PGUP", b"Q": "PGDN",
                b"G": "HOME", b"O": "END",
            }.get(msvcrt.getch())
        return {b"\x1b": "ESC", b"q": "QUIT"}.get(key)

    def setup_keyboard():
        """Windows console input needs no additional setup."""

    def restore_keyboard():
        """Windows console input needs no restoration."""

else:
    import select
    import termios
    import tty

    _old_tty = None

    def setup_keyboard():
        global _old_tty
        _old_tty = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def restore_keyboard():
        if _old_tty:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _old_tty)

    def read_key_nonblocking():
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        key = sys.stdin.read(1)
        if key == "\x1b":
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                return "ESC"
            return {
                "[A": "UP", "[B": "DOWN", "[5": "PGUP", "[6": "PGDN",
                "[H": "HOME", "[F": "END",
            }.get(sys.stdin.read(2))
        return "QUIT" if key == "q" else None


BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"

_HIST_PREFIX_VIS = 36
_PLAN_PREFIX_VIS = 21
_read_raw = _shared_read_whole_file
load_jsonl = _shared_load_jsonl


@dataclass(frozen=True)
class RenderContext:
    task_dir: str
    progress: Optional[dict]
    history: list[dict]
    plan_text: str
    plan_mtime: Optional[float]
    history_offset: int
    history_window: Optional[int]
    term_width: int
    term_height: int

    @property
    def history_description_width(self) -> int:
        return max(10, self.term_width - _HIST_PREFIX_VIS - 2)

    @property
    def plan_description_width(self) -> int:
        return max(10, self.term_width - _PLAN_PREFIX_VIS - 2)

    @property
    def divider_width(self) -> int:
        return max(40, self.term_width - 2)


@dataclass
class WatchState:
    history_offset: int = 0
    last_render: float = 0.0
    interactive: bool = False


def fmt_metric(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def load_plan(path):
    if not os.path.exists(path):
        return "(no plan yet)", None
    return _read_raw(path), os.path.getmtime(path)


def bar(fraction, width=30):
    filled = int(fraction * width)
    return f"[{'#' * filled}{'.' * (width - filled)}]"


def _fit(text: str, available: int) -> str:
    """Fit text into a visible terminal column."""
    if available <= 0:
        return ""
    if len(text) <= available:
        return text
    if available == 1:
        return "…"
    return text[: available - 1] + "…"


def _terminal_size() -> tuple[int, int]:
    try:
        size = os.get_terminal_size()
        return size.columns, size.lines
    except OSError:
        logger.debug("Could not read terminal size", exc_info=True)
        return 100, 40


def _render_context(task_dir: str, history_offset: int,
                    history_window: Optional[int]) -> RenderContext:
    width, height = _terminal_size()
    plan_text, plan_mtime = load_plan(_pm.plan_path(task_dir))
    return RenderContext(
        task_dir=task_dir,
        progress=_pm.load_state(task_dir),
        history=load_jsonl(_pm.history_path(task_dir)),
        plan_text=plan_text,
        plan_mtime=plan_mtime,
        history_offset=history_offset,
        history_window=history_window,
        term_width=width,
        term_height=height,
    )


def _dashboard_header() -> list[str]:
    width = 62
    return [
        f"{BOLD}{CYAN}╔{'═' * width}╗{RESET}",
        f"{BOLD}{CYAN}║         AUTORESEARCH DASHBOARD                             ║{RESET}",
        f"{BOLD}{CYAN}╚{'═' * width}╝{RESET}",
    ]


def _missing_state_lines(context: RenderContext) -> list[str]:
    return [
        "",
        f"  {RED}No state.json found at "
        f"{_pm.state_record_path(context.task_dir)}{RESET}",
        "  Run /autoresearch --ref ... --op-name ... first.",
    ]


def _fresh_task_lines(progress: dict) -> list[str]:
    lines = [
        "",
        f"  {BOLD}{CYAN}Task scaffolded; baseline not yet run.{RESET}",
    ]
    owner = progress.get("owner") or {}
    if owner.get("session_id"):
        lines.append(f"  Owner session: {DIM}{owner.get('session_id')}{RESET}")
    lines.append(f"  {DIM}This dashboard will populate once baseline.py "
                 f"commits the first measurement.{RESET}")
    return lines


def _local_timestamp(raw_value) -> str:
    if not isinstance(raw_value, str):
        return str(raw_value)
    try:
        timestamp = datetime.fromisoformat(raw_value)
    except ValueError:
        return raw_value
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def _speedup_text(progress: dict) -> str:
    speedup = recorded_speedup(progress)
    if speedup is None:
        return f"{DIM}N/A{RESET}"
    source = progress.get("baseline_source")
    anchor_label = "vs ref" if source == "ref" else "vs baseline"
    improvement = (1.0 - 1.0 / speedup) * 100
    color = GREEN if speedup > 1 else RED
    return (f"{color}{speedup:.2f}x {anchor_label} "
            f"({improvement:+.1f}%){RESET}")


def _abort_banner(progress: dict) -> list[str]:
    if progress.get("baseline_outcome") != "infra_fail":
        return []
    if (progress.get("baseline_error_source") or "") == "ref":
        return [
            "",
            f"  {BOLD}{RED}ABORTED:{RESET}  {RED}REF BROKEN{RESET}  "
            "reference.py is invalid.",
            f"           {DIM}Fix the source --ref file and re-run "
            f"/autoresearch from scratch.{RESET}",
        ]
    return [
        "",
        f"  {BOLD}{YELLOW}ABORTED:{RESET}  "
        f"{YELLOW}EVAL PIPELINE BROKEN{RESET}  no per-shape data produced.",
        f"           {DIM}Check device / env / eval.timeout, then retry "
        f"baseline.py.{RESET}",
    ]


def _baseline_lines(progress: dict, outcome: Optional[str]) -> list[str]:
    baseline = progress.get("baseline_metric")
    if baseline is None:
        baseline_line = f"  {BOLD}Baseline:{RESET} {DIM}— (not measured){RESET}"
    else:
        source = progress.get("baseline_source")
        tag = (f"{DIM}(PyTorch reference){RESET}" if source == "ref"
               else f"{DIM}(source unknown){RESET}")
        baseline_line = (
            f"  {BOLD}Baseline:{RESET} {fmt_metric(baseline)}  {tag}")
    seed = progress.get("seed_metric")
    if seed is not None and seed != baseline:
        seed_line = (f"  {BOLD}Seed:{RESET}     {fmt_metric(seed)}  "
                     f"{DIM}(initial kernel){RESET}")
    elif seed is None and outcome == "kernel_fail":
        seed_line = (f"  {BOLD}Seed:{RESET}     {RED}FAILED{RESET}  "
                     f"{DIM}(kernel verify or profile failed; timing dropped)"
                     f"{RESET}")
    elif seed is None:
        seed_line = f"  {BOLD}Seed:{RESET}     {DIM}— (no timing recorded){RESET}"
    else:
        seed_line = ""
    return [baseline_line] + ([seed_line] if seed_line else [])


def _overview_lines(context: RenderContext) -> list[str]:
    progress = context.progress or {}
    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", _default_max_rounds())
    failures = progress.get("consecutive_failures", 0)
    status = ("active" if os.path.exists(_pm.plan_path(context.task_dir))
              else "no_plan")
    fraction = rounds / max_rounds if max_rounds > 0 else 0
    budget_color = GREEN if fraction < 0.5 else (YELLOW if fraction < 0.8 else RED)
    status_color = GREEN if status == "active" else YELLOW
    fail_color = RED if failures >= 3 else (YELLOW if failures else GREEN)
    lines = [
        "",
        f"  {BOLD}Task:{RESET}     {progress.get('task', '?')}",
        f"  {BOLD}Status:{RESET}   {status_color}{status}{RESET}  "
        f"(plan v{progress.get('plan_version', 0)})",
        f"  {BOLD}Updated:{RESET}  {DIM}"
        f"{_local_timestamp(progress.get('last_touched', '?'))}{RESET}",
    ]
    lines.extend(_abort_banner(progress))
    lines.extend([
        "",
        f"  {BOLD}Budget:{RESET}   {budget_color}{bar(fraction)} "
        f"{rounds}/{max_rounds}{RESET}",
    ])
    outcome = progress.get("baseline_outcome")
    lines.extend(_baseline_lines(progress, outcome))
    lines.extend([
        f"  {BOLD}Best:{RESET}     {GREEN}"
        f"{fmt_metric(progress.get('best_metric'))}{RESET}  "
        f"({_speedup_text(progress)})",
        f"  {BOLD}Commit:{RESET}   {progress.get('best_commit', '?')}",
        f"  {BOLD}Failures:{RESET} {fail_color}{failures}{RESET} consecutive"
        + (f"  {RED}⚠ DIAGNOSIS WILL TRIGGER{RESET}" if failures >= 3 else ""),
    ])
    return lines


def _history_view(context: RenderContext) -> tuple[list[dict], str]:
    total = len(context.history)
    window = context.history_window
    if window is None:
        window = max(5, context.term_height - 28)
    offset = max(0, min(context.history_offset, max(0, total - window)))
    end = total - offset
    start = max(0, end - window)
    info = ""
    if total > window:
        info = (f" [{start + 1}-{end} of {total}, "
                "↑↓ PgUp/PgDn Home/End q=quit]")
    return context.history[start:end], info


def _history_metric(metrics: dict) -> str:
    for key in ("latency_us", "score"):
        if key in metrics and metrics[key] is not None:
            return fmt_metric(metrics[key])
    return "—"


def _decision_text(decision: str) -> str:
    styles = {
        "KEEP": f"{GREEN}  KEEP  {RESET}",
        "DISCARD": f"{YELLOW}DISCARD {RESET}",
        "FAIL": f"{RED}  FAIL  {RESET}",
        "SEED": f"{CYAN}  SEED  {RESET}",
    }
    return styles.get(decision, f"{DIM}{decision:^8}{RESET}")


def _history_row(record: dict, description_width: int) -> str:
    round_value = record.get("round")
    round_text = "?" if round_value is None else str(round_value)
    description = record.get("description", "")
    if record.get("plan_item"):
        description = f"{record['plan_item']}: {description}"
    description = _fit(description, description_width)
    decision = record.get("decision", "?")
    metric = _history_metric(record.get("metrics", {}))
    return (f"  {round_text:>3}  │ {_decision_text(decision)} │ "
            f"{metric:>13} │ {description}")


def _history_lines(context: RenderContext) -> list[str]:
    history, scroll_info = _history_view(context)
    divider = "─" * context.divider_width
    lines = [
        "",
        f"  {BOLD}History{RESET}{DIM}{scroll_info}{RESET}",
        f"  {BOLD}{divider}{RESET}",
        f"  {BOLD}  #  │ Decision │ Metric        │ Description{RESET}",
        f"  {BOLD}{divider}{RESET}",
    ]
    lines.extend(
        _history_row(record, context.history_description_width)
        for record in history
    )
    lines.append(f"  {BOLD}{divider}{RESET}")
    return lines


def _plan_age(mtime: Optional[float]) -> str:
    if not mtime:
        return ""
    age_seconds = time.time() - mtime
    if age_seconds < 60:
        return f"{DIM}(updated {int(age_seconds)}s ago){RESET}"
    return f"{DIM}(updated {int(age_seconds / 60)}m ago){RESET}"


def _plan_outcome(tag: str) -> str:
    return next(
        (candidate for candidate in ("KEEP", "DISCARD", "FAIL")
         if tag.startswith(candidate)),
        "",
    )


def _plan_status(item: dict, description: str) -> tuple[str, str]:
    if item["active"]:
        return f"{CYAN}> ACTIVE {RESET}", f"{CYAN}{description}{RESET}"
    outcome = _plan_outcome(item["tag"])
    styles = {
        "KEEP": (f"{GREEN}  KEEP   {RESET}", f"{DIM}{description}{RESET}"),
        "DISCARD": (f"{YELLOW} DISCARD {RESET}", f"{DIM}{description}{RESET}"),
        "FAIL": (f"{RED}  FAIL   {RESET}", f"{DIM}{description}{RESET}"),
    }
    return styles.get(outcome, (" pending ", description))


def _plan_lines(context: RenderContext) -> list[str]:
    divider = "─" * context.divider_width
    lines = [
        "",
        f"  {BOLD}Current Plan:{RESET} {_plan_age(context.plan_mtime)}",
        f"  {BOLD}{divider}{RESET}",
        f"  {BOLD}  #   │ Status    │ Description{RESET}",
        f"  {BOLD}{divider}{RESET}",
    ]
    for item in _pm.parse_plan_text(context.plan_text):
        description = _fit(
            item["description"], context.plan_description_width)
        status, styled_description = _plan_status(item, description)
        lines.append(
            f"  {item['id']:>4}  │ {status}│ {styled_description}")
    lines.append(f"  {BOLD}{divider}{RESET}")
    return lines


def render(task_dir, history_offset=0, history_window=None):
    """Render one dashboard frame for ``task_dir``."""
    context = _render_context(task_dir, history_offset, history_window)
    lines = _dashboard_header()
    if context.progress is None:
        lines.extend(_missing_state_lines(context))
        return "\n".join(lines)
    if not context.progress.get("progress_initialized"):
        lines.extend(_fresh_task_lines(context.progress))
        return "\n".join(lines)
    lines.extend(_overview_lines(context))
    lines.extend(_history_lines(context))
    lines.extend(_plan_lines(context))
    lines.extend(["", f"  {DIM}Press Ctrl+C to stop watching{RESET}"])
    return "\n".join(lines)


def _auto_detect_task_dir() -> str:
    """Use the shared active-task discovery rule."""
    return _pm.find_active_task_dir() or ""


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("AutoResearch live dashboard. Auto-detects task if no "
                     "path is given."),
    )
    parser.add_argument(
        "task_dir", nargs="?", default=None,
        help="Path to task directory (auto-detected if omitted)",
    )
    parser.add_argument(
        "--watch", type=int, nargs="?", const=5, default=5,
        help="Refresh interval in seconds (default: 5, use 0 for one-shot)",
    )
    return parser


def _resolve_task_dir(requested: Optional[str]) -> Optional[str]:
    if requested:
        return os.path.abspath(requested)
    detected = _auto_detect_task_dir()
    if not detected:
        return None
    emit(f"Auto-detected: {detected}", file=sys.stderr)
    return detected


def _configure_windows_console() -> None:
    if sys.platform != "win32":
        return
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except (AttributeError, OSError):
        logger.debug("Could not enable Windows console mode", exc_info=True)


def _setup_interactive_keyboard() -> bool:
    try:
        interactive = sys.stdin.isatty()
    except (AttributeError, OSError):
        logger.debug("Dashboard input is not an interactive terminal", exc_info=True)
        return False
    if not interactive:
        return False
    try:
        setup_keyboard()
        return True
    except (OSError, termios.error) if sys.platform != "win32" else OSError:
        logger.debug("Could not configure dashboard keyboard", exc_info=True)
        return False


def _safe_read_key(interactive: bool):
    if not interactive:
        return None
    try:
        return read_key_nonblocking()
    except (OSError, ValueError):
        logger.debug("Could not read dashboard key", exc_info=True)
        return None


def _apply_navigation(key, offset: int) -> tuple[int, bool, bool]:
    if key in ("QUIT", "ESC"):
        return offset, True, False
    updates = {
        "UP": offset + 1,
        "DOWN": max(0, offset - 1),
        "PGUP": offset + 10,
        "PGDN": max(0, offset - 10),
        "HOME": 999999,
        "END": 0,
    }
    if key not in updates:
        return offset, False, False
    return updates[key], False, True


def _render_watch_frame(task_dir: str, state: WatchState, interval: int,
                        navigation_changed: bool) -> None:
    now = time.time()
    if not navigation_changed and now - state.last_render < interval:
        return
    emit("\033[2J\033[H", end="")
    emit(render(task_dir, history_offset=state.history_offset), flush=True)
    state.last_render = now


def _restore_interactive_keyboard(interactive: bool) -> None:
    if not interactive:
        return
    try:
        restore_keyboard()
    except (OSError, ValueError):
        logger.debug("Could not restore keyboard state", exc_info=True)


def _watch_dashboard(task_dir: str, interval: int) -> None:
    state = WatchState(interactive=_setup_interactive_keyboard())
    try:
        while True:
            key = _safe_read_key(state.interactive)
            state.history_offset, stop, changed = _apply_navigation(
                key, state.history_offset)
            if stop:
                break
            _render_watch_frame(task_dir, state, interval, changed)
            delay = 0.1 if state.interactive else max(0.5, interval / 2)
            time.sleep(delay)
    except KeyboardInterrupt:
        logger.debug("Dashboard interrupted by the user")
    finally:
        _restore_interactive_keyboard(state.interactive)
        emit(f"\n{DIM}Dashboard stopped.{RESET}")


def main() -> int:
    args = _make_parser().parse_args()
    task_dir = _resolve_task_dir(args.task_dir)
    if not task_dir:
        emit("No task found. Pass a task_dir or start /autoresearch first.")
        return 1
    _configure_windows_console()
    if args.watch and args.watch > 0:
        _watch_dashboard(task_dir, args.watch)
    else:
        emit(render(task_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
