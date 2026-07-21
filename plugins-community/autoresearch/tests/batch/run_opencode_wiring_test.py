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

"""Wiring test for batch run.py --agent opencode.

Proves run_one_opencode builds the right `run_loop.py` command, binds the
task_dir from run_loop's `[run_loop] task_dir=` line, and records the
manifest case — all WITHOUT spawning opencode or needing NPU. The actual
opencode↔plugin↔decide loop is proven separately (tests/opencode_door +
the live run_loop tests); here we only pin the batch-side wiring so the
Claude path and the opencode path stay in lockstep on bookkeeping.

Usage:  python tests/batch/run_opencode_wiring_test.py
"""
import argparse
import importlib.util
import io
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from wiring_support import REPO, R, emit, phase_machine, task_handle

_RL_SPEC = importlib.util.spec_from_file_location(
    "autoresearch_run_loop", REPO / ".opencode" / "run_loop.py")
RUN_LOOP = importlib.util.module_from_spec(_RL_SPEC)
_RL_SPEC.loader.exec_module(RUN_LOOP)


class _DummyCM:
    def __enter__(self):
        return self

    @staticmethod
    def __exit__(*_args):
        return False


@dataclass(frozen=True)
class _RunScenario:
    stream_lines: list[str]
    rc: int = 0
    phase: str = "FINISH"
    progress_task_dir: str | None = None


def _make_args(**over):
    base = dict(max_rounds=5, eval_timeout=30, model="", timeout_min=60,
                agent="opencode")
    base.update(over)
    return argparse.Namespace(**base)


def _install_stubs(updates, progress_task_dir, phase):
    def update_case(_batch_dir, _op_name, **values):
        updates.append(values)

    def now_iso():
        return "T0"

    def repo_root():
        return REPO

    def load_progress(_batch_dir):
        return {"cases": {"myop": {"task_dir": progress_task_dir}}}

    def read_phase(_task_dir):
        return phase

    def read_task_state(_task_dir):
        return {"best_metric": 1.0}

    def clear_active_task(**_kwargs):
        return True

    def open_task(*_args, **_kwargs):
        return _DummyCM()

    R.mf.update_case = update_case
    R.mf.now_iso = now_iso
    R.mf.repo_root = repo_root
    R.mf.load_progress = load_progress
    R.mf.read_phase = read_phase
    R.mf.read_task_state = read_task_state
    phase_machine.clear_active_task = clear_active_task
    task_handle.open_task = open_task


def _run(case, hw_arg, scenario: _RunScenario):
    """Drive run_one_opencode with all process/IO/state calls stubbed.
    Returns (captured_cmd, update_calls, result_rc).
    """
    captured = {"cmd": None, "extra_env": None}
    updates = []

    def fake_stream(request):
        captured["cmd"] = request.cmd
        captured["extra_env"] = request.extra_env
        line_cb = request.line_cb
        for ln in scenario.stream_lines:
            if line_cb:
                line_cb(ln)
        return scenario.rc, False

    # Stub the agent-neutral primitives so nothing real is spawned/written.
    R.stream_subprocess = fake_stream
    _install_stubs(updates, scenario.progress_task_dir, scenario.phase)

    request = R.CaseRequest(
        REPO,
        case,
        _make_args(),
        hw_arg,
        io.StringIO(),
    )
    rc_out = R.run_one_opencode(request)
    return captured, updates, rc_out


def _case_one_command_checks(command, environment):
    command_text = " ".join(command or [])
    return [
        (command is not None, "command was built"),
        (
            str(REPO / ".opencode" / "run_loop.py") in command_text,
            "invokes run_loop.py",
        ),
        ("--ref workspace/myop_ref.py" in command_text, "passes --ref"),
        (
            "--kernel workspace/myop_kernel.py" in command_text,
            "passes --kernel",
        ),
        ("--op-name myop" in command_text, "passes --op-name"),
        ("--max-rounds 5" in command_text, "passes --max-rounds"),
        ("--devices 0" in command_text, "forwards hw flag"),
        (
            environment.get("AR_BATCH_OP") == "myop",
            "passes batch op env",
        ),
        (
            environment.get("AR_BATCH_DIR") == str(REPO.resolve()),
            "passes batch dir env",
        ),
        ("--max-iters" not in command_text, "does not add --max-iters"),
    ]


def _case_one_result_checks(updates, result):
    return [
        (
            any(update.get("task_dir") for update in updates),
            "binds task_dir into manifest",
        ),
        (
            any(
                update.get("status") == "done"
                and update.get("final_phase") == "FINISH"
                for update in updates
            ),
            "records status=done at FINISH",
        ),
        (result == 0, "returns rc=0 on done"),
    ]


def _case_one_checks(case):
    with tempfile.TemporaryDirectory() as tmp:
        task_dir = Path(tmp) / "myop_123_abc123"
        task_dir.mkdir()
        task_path = str(task_dir.resolve())
        captured, updates, result = _run(
            case,
            "--devices 0",
            _RunScenario(
                [f"[run_loop] task_dir={task_path}\n", "some log\n"]
            ),
        )
    checks = _case_one_command_checks(
        captured["cmd"], captured["extra_env"] or {}
    )
    checks.extend(_case_one_result_checks(updates, result))
    return checks


def _case_two_checks(case):
    captured, updates, result = _run(
        case,
        "--worker-url h:1",
        _RunScenario(["nothing useful\n"], rc=2, phase="PLAN"),
    )
    command_text = " ".join(captured["cmd"] or [])
    return [
        (result == 2, "returns 2 when no task_dir is bound"),
        (
            any(update.get("status") == "error" for update in updates),
            "records error",
        ),
        (
            "--worker-url h:1" in command_text,
            "forwards worker-url hw flag",
        ),
    ]


def _case_three_checks(case):
    with tempfile.TemporaryDirectory() as tmp:
        task_dir = Path(tmp) / "myop_123_abc123"
        task_dir.mkdir()
        _, updates, result = _run(
            case,
            "--devices 0",
            _RunScenario(
                ["ordinary log\n"],
                progress_task_dir=str(task_dir),
            ),
        )
        expected_dir = str(task_dir.resolve())
    return [
        (result == 0, "returns rc=0 from progress fallback"),
        (
            any(
                update.get("task_dir") == expected_dir
                for update in updates
            ),
            "uses scaffold-recorded task_dir",
        ),
        (
            any(update.get("status") == "done" for update in updates),
            "records done from progress fallback",
        ),
    ]


def _session_log_checks():
    legacy = RUN_LOOP.SESSION_RE.search(
        "message=created id=ses_old123 slug=fixture"
    )
    current = RUN_LOOP.SESSION_RE.search(
        "service=session id=ses_new456 slug=fixture created"
    )
    return [
        (
            legacy and legacy.group(1) == "ses_old123",
            "captures legacy session log",
        ),
        (
            current and current.group(1) == "ses_new456",
            "captures current session log",
        ),
        (
            "ProviderModelNotFoundError"
            in RUN_LOOP.FATAL_PROVIDER_MARKERS,
            "classifies missing configured model as fatal",
        ),
    ]


def _record_checks(prefix, checks, failures):
    for passed, label in checks:
        full_label = f"{prefix}: {label}"
        emit(("[ok]   " if passed else "[FAIL] ") + full_label)
        if not passed:
            failures.append(full_label)


def main() -> int:
    failures = []
    case = {
        "op_name": "myop",
        "ref": "workspace/myop_ref.py",
        "kernel": "workspace/myop_kernel.py",
    }
    groups = [
        ("case1", _case_one_checks(case)),
        ("case2", _case_two_checks(case)),
        ("case3", _case_three_checks(case)),
        ("case4", _session_log_checks()),
    ]
    for prefix, checks in groups:
        _record_checks(prefix, checks, failures)
    if failures:
        emit(f"\n{len(failures)} wiring check(s) failed")
        return 1
    total = sum(len(checks) for _, checks in groups)
    emit(f"\nAll {total} opencode-batch wiring checks pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
