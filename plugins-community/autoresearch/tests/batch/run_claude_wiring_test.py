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

"""Pins Claude batch wiring to the shared agent subprocess supervisor."""

import argparse
import io
import sys
import tempfile
from pathlib import Path

from wiring_support import REPO, R, emit, phase_machine, task_handle


class _DummyCM:
    def __enter__(self):
        return self

    @staticmethod
    def __exit__(*_args):
        return False


class _BrokenStdout:
    """Models an SSH/tee consumer that disappeared mid-batch."""

    @staticmethod
    def write(_text):
        raise BrokenPipeError("controlling SSH pipe closed")

    @staticmethod
    def flush():
        raise BrokenPipeError("controlling SSH pipe closed")


def _install_stubs(task_dir: Path, captured: dict, updates: list) -> None:
    def fake_stream(request):
        captured.update(
            cmd=request.cmd,
            cwd=request.cwd,
            timeout_s=request.timeout_s,
            extra_env=request.extra_env,
        )
        return 0, False

    def repo_root():
        return REPO

    def now_iso():
        return "T0"

    def update_case(_batch_dir, _op_name, **values):
        updates.append(values)

    def snapshot_task_dirs():
        return set()

    def load_progress(_batch_dir):
        return {"cases": {"myop": {"task_dir": str(task_dir)}}}

    def no_new_task_dir(*_args, **_kwargs):
        return None

    def finished_phase(_task_dir):
        return "FINISH"

    def finished_state(_task_dir):
        return {"best_metric": 1.0}

    def clear_active_task(**_kwargs):
        return True

    def open_task(*_args, **_kwargs):
        return _DummyCM()

    R.stream_subprocess = fake_stream
    R.mf.repo_root = repo_root
    R.mf.now_iso = now_iso
    R.mf.update_case = update_case
    R.mf.snapshot_task_dirs = snapshot_task_dirs
    R.mf.load_progress = load_progress
    R.mf.pick_new_task_dir = no_new_task_dir
    R.mf.read_phase = finished_phase
    R.mf.read_task_state = finished_state
    phase_machine.clear_active_task = clear_active_task
    task_handle.open_task = open_task


def _run_claude_cases():
    captured = {}
    updates = []
    with tempfile.TemporaryDirectory() as tmp:
        task_dir = Path(tmp) / "myop_123_abc123"
        task_dir.mkdir()
        _install_stubs(task_dir, captured, updates)
        args = argparse.Namespace(
            claude_bin="claude",
            model="",
            extra_claude_arg=[],
            max_rounds=5,
            eval_timeout=30,
            timeout_min=60,
        )
        case = {
            "op_name": "myop",
            "ref": "workspace/myop_ref.py",
            "kernel": "workspace/myop_kernel.py",
        }
        result = R.run_one(
            R.CaseRequest(REPO, case, args, "--devices 0", io.StringIO())
        )
        updates.clear()
        broken_log = io.StringIO()
        old_stdout = R.sys.stdout
        try:
            R.sys.stdout = _BrokenStdout()
            broken_result = R.run_one(
                R.CaseRequest(
                    REPO, case, args, "--devices 0", broken_log
                )
            )
        finally:
            R.sys.stdout = old_stdout
    return {
        "task_dir": task_dir,
        "captured": captured,
        "updates": updates,
        "result": result,
        "broken_result": broken_result,
        "broken_log": broken_log.getvalue(),
    }


def _claude_wiring_checks(run_result):
    captured = run_result["captured"]
    updates = run_result["updates"]
    return [
        (
            captured.get("cmd", [None])[0] == "claude",
            "Claude command uses shared stream supervisor",
        ),
        (
            captured.get("extra_env", {}).get("AR_BATCH_OP") == "myop",
            "shared supervisor receives batch identity env",
        ),
        (
            captured.get("timeout_s") == 3600,
            "shared supervisor receives wall-clock budget",
        ),
        (
            run_result["result"] == 0
            and run_result["broken_result"] == 0,
            "normal and closed-stdout runs both succeed",
        ),
        (
            any(
                update.get("status") == "done"
                and update.get("rc") == 0
                for update in updates
            ),
            "FINISH result persists after console EPIPE",
        ),
        (
            "launching: claude --print" in run_result["broken_log"]
            and "exited rc=0" in run_result["broken_log"],
            "durable batch log survives console EPIPE",
        ),
    ]


def _exercise_claude_wiring():
    run_result = _run_claude_cases()
    return run_result["task_dir"], _claude_wiring_checks(run_result)


def _always_false(*_args) -> bool:
    return False


def _stale_recovery_checks(task_dir: Path):
    stale_progress = {
        "cases": {
            "myop": {
                "status": "running",
                "task_dir": str(task_dir),
                "runner_pid": 999999,
                "note": "",
            }
        }
    }
    R.pid_alive = _always_false
    phase_machine.is_task_active = _always_false
    demoted, harvested = R.recover_stale_running(stale_progress)
    harvested_case = stale_progress["cases"]["myop"]
    return [
        (
            demoted == 0 and harvested == 1,
            "stale recovery harvests a completed task",
        ),
        (
            harvested_case.get("status") == "done"
            and harvested_case.get("final_phase") == "FINISH"
            and harvested_case.get("result", {}).get("best_metric") == 1.0,
            "stale recovery preserves the authoritative FINISH result",
        ),
    ]


def _report_checks(checks) -> int:
    failures = []
    for passed, label in checks:
        emit(("[ok]   " if passed else "[FAIL] ") + label)
        if not passed:
            failures.append(label)
    if failures:
        emit(f"\n{len(failures)} Claude wiring checks failed")
        return 1
    emit(f"\nAll {len(checks)} Claude shared-supervisor checks pass.")
    return 0


def main() -> int:
    task_dir, checks = _exercise_claude_wiring()
    checks.extend(_stale_recovery_checks(task_dir))
    return _report_checks(checks)


if __name__ == "__main__":
    sys.exit(main())
