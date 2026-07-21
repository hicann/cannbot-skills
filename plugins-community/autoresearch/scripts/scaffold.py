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

"""
Task directory scaffolder for Claude Code autoresearch.

Zero external dependency. Creates a self-contained task directory with:
  - task.yaml (config)
  - reference.py (correctness baseline; AST-checked via utils.ref_ast.
    validate_ref before scaffold copies it. Runtime correctness is
    validated by --run-baseline whose verify routine tags error_source.)
  - kernel.py (editable seed; from --kernel file, or sibling kernel.py when
    --kernel is a multi-file DSL project directory)
  - <dsl project>/ (multi-file DSLs only, when --kernel points at that folder)
  - .ar_state/ (progress tracking)
  - .git/ (baseline commit)

Usage:
    # NOTE: --devices values below are placeholders; pass the actual free
    # device id at invocation time.

    # Local eval (arch auto-derived from the selected Ascend device):
    python scripts/scaffold.py --ref reference.py --kernel kernel.py --op-name my_op --devices <DEV>

    # Custom output directory:
    python scripts/scaffold.py --ref reference.py --kernel kernel.py \
        --op-name my_op --devices <DEV> --output-dir /tmp/tasks

Output (last line of stdout):
    {"task_dir": "/absolute/path/to/task_dir", "status": "ok"}
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml
from op_autoresearch.op.utils.task_layout import REF_FILE_DEFAULT
from op_autoresearch.utils.console import emit
from phase_machine import task_summary
from task_handle import Role, open_task
from utils.git_utils import commit_in_task
from utils.hw_detect import derive_arch, probe_hint

# ---------------------------------------------------------------------------
# Reference validation — delegated to the standalone library module so
# phase_machine.validators can call the same rule without importing this
# CLI script. The local re-export keeps callers that imported
# `scaffold.validate_ref` working.
# ---------------------------------------------------------------------------
from utils.ref_ast import validate_ref
from utils.settings import (
    default_code_checker_enabled,
    default_eval_timeout,
    default_max_rounds,
    default_metric,
    target_backend,
    target_dsl,
)

# ---------------------------------------------------------------------------
# DSL-aware scaffold dispatch: every per-DSL knob (does --kernel take a
# directory, what files beyond kernel.py are editable, what extra source
# tree gets copied into task_dir) is owned by the DSL adapter. Scaffold
# stays DSL-name-agnostic.
# ---------------------------------------------------------------------------


def _scaffold_dsl_adapter():
    from op_autoresearch.op.verifier.adapters.factory import get_dsl_adapter
    return get_dsl_adapter(target_dsl())


def _run_initial_baseline(task_dir: str) -> int:
    """Activate a newly scaffolded task, then run its baseline.

    PostToolUse cannot own this transition: ``--run-baseline`` executes
    baseline.py inside the scaffold command, before the hook can observe the
    new task.  Keep activation next to that synchronous call so every caller
    (Claude, OpenCode, batch, or a plain CLI) sees the same INIT -> BASELINE
    ordering.
    """
    with open_task(task_dir, role=Role.SUPERVISOR) as task:
        task.activate(fresh=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return subprocess.run([
        sys.executable,
        os.path.join(script_dir, "engine", "baseline.py"),
        task_dir,
    ], check=False).returncode


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------


@dataclass
class ScaffoldRequest:
    """Complete specification for one task directory."""

    ref_code: str
    kernel_code: str
    op_name: str
    desc: str = ""
    arch: str = ""
    devices: list | None = None
    max_rounds: int | None = None
    eval_timeout: int | None = None
    output_dir: str | None = None
    editable_filename: str = "kernel.py"
    editable_files: list | None = None
    kernel_project_src: str | None = None
    code_checker_enabled: bool | None = None
    ref_source_path: str | None = None
    worker_url: str = ""


@dataclass(frozen=True)
class PreparedSources:
    ref_code: str
    kernel_code: str
    kernel_project_src: str | None
    entry_filename: str
    editable_files: list[str]


class ScaffoldCliError(ValueError):
    """The CLI inputs cannot produce a valid task directory."""


def scaffold_task_dir(request: ScaffoldRequest) -> str:
    """Create a task directory and return its absolute path."""
    _resolve_scaffold_defaults(request)
    base_dir = request.output_dir or os.path.join(os.getcwd(), "ar_tasks")
    name = f"{request.op_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    task_dir = os.path.join(base_dir, name)
    os.makedirs(task_dir)

    _write(task_dir, REF_FILE_DEFAULT, request.ref_code)
    _write(task_dir, request.editable_filename, request.kernel_code)
    _scaffold_dsl_adapter().materialize_project_tree(
        task_dir,
        request.kernel_project_src,
    )
    data_files = _copy_reference_sidecars(
        task_dir,
        request.ref_source_path,
    )
    task_yaml = _build_task_yaml(request, task_dir, data_files)
    _write(
        task_dir,
        "task.yaml",
        yaml.dump(task_yaml, default_flow_style=False, allow_unicode=True),
    )
    os.makedirs(os.path.join(task_dir, ".ar_state"), exist_ok=True)
    _git_init(task_dir)
    return os.path.abspath(task_dir)


def _resolve_scaffold_defaults(request: ScaffoldRequest) -> None:
    if request.max_rounds is None:
        request.max_rounds = default_max_rounds()
    if request.eval_timeout is None:
        request.eval_timeout = default_eval_timeout()
    if request.code_checker_enabled is None:
        request.code_checker_enabled = default_code_checker_enabled()


def _copy_reference_sidecars(
    task_dir: str,
    ref_source_path: str | None,
) -> list[str]:
    if not ref_source_path:
        return []
    import shutil

    source_dir = os.path.dirname(os.path.abspath(ref_source_path))
    ref_stem = os.path.splitext(os.path.basename(ref_source_path))[0]
    copied = []
    try:
        candidates = sorted(os.listdir(source_dir))
        for filename in candidates:
            source = os.path.join(source_dir, filename)
            if not _is_reference_sidecar(filename, source, ref_stem):
                continue
            destination = _sidecar_destination(filename, ref_stem)
            shutil.copy(source, os.path.join(task_dir, destination))
            copied.append(destination)
    except OSError as exc:
        emit(f"[scaffold] WARNING: sidecar data file copy failed: {exc}")
    return copied


def _is_reference_sidecar(
    filename: str,
    source: str,
    ref_stem: str,
) -> bool:
    if filename.startswith(".") or not os.path.isfile(source):
        return False
    stem, extension = os.path.splitext(filename)
    if extension.lower() not in (".json", ".pt", ".npz"):
        return False
    return stem == ref_stem or stem.startswith(ref_stem + "_")


def _sidecar_destination(filename: str, ref_stem: str) -> str:
    stem, extension = os.path.splitext(filename)
    if stem != ref_stem:
        return filename
    return os.path.splitext(REF_FILE_DEFAULT)[0] + extension


def _build_task_yaml(
    request: ScaffoldRequest,
    task_dir: str,
    data_files: list[str],
) -> dict:
    eval_block = {"timeout": request.eval_timeout}
    num_cases = _probe_num_cases(task_dir, REF_FILE_DEFAULT)
    if num_cases and num_cases >= 1:
        eval_block["num_cases"] = num_cases
    metric = default_metric()
    task_yaml = {
        "name": request.op_name,
        "description": request.desc or f"Optimize {request.op_name}",
        "arch": request.arch or None,
        "editable_files": (
            request.editable_files
            or [request.editable_filename]
        ),
        "eval": eval_block,
        "metric": {
            "primary": metric["primary"],
            "lower_is_better": metric["lower_is_better"],
            "improvement_threshold": metric["improvement_threshold"],
        },
        "agent": {
            "ref_file": REF_FILE_DEFAULT,
            "max_rounds": request.max_rounds,
        },
        "code_checker": {
            "enabled": bool(request.code_checker_enabled),
        },
    }
    if request.devices:
        task_yaml["devices"] = list(request.devices)
    if request.worker_url:
        task_yaml["worker"] = {
            "urls": [
                url.strip()
                for url in request.worker_url.split(",")
                if url.strip()
            ],
        }
    if data_files:
        task_yaml["data_files"] = data_files
    return task_yaml


def _probe_num_cases(task_dir: str, ref_file: str):
    """Best-effort case count for task.yaml ``eval.num_cases``. Loads the
    just-written reference module. Generated multi-shape refs expose a
    literal ``CASES`` table, so count that first instead of calling
    ``get_input_groups()`` and constructing large tensors at scaffold time.
    For generic refs, delegate to ``utils.input_groups.num_cases`` so
    single / dyn_list / input_groups refs still resolve consistently.
    Returns None when the ref can't be imported here (e.g. no torch on the
    dev host); caller omits the field and eval_timeout scaling falls back
    to a runtime re-probe.
    """
    import importlib.util
    ref_path = os.path.join(task_dir, ref_file)
    if not os.path.isfile(ref_path):
        return None
    try:
        old_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec = importlib.util.spec_from_file_location("_ref_probe", ref_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            sys.dont_write_bytecode = old_dont_write_bytecode
        cases = getattr(mod, "CASES", None)
        if isinstance(cases, (list, tuple)):
            return len(cases)
        from utils.input_groups import num_cases
        return num_cases(mod)
    except Exception:
        return None


def _write(task_dir: str, rel_path: str, content: str):
    full_path = os.path.join(task_dir, rel_path)
    parent = os.path.dirname(full_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)


def _git_init(task_dir: str):
    """Initialize git repo and create baseline commit.

    The actual commit goes through git_utils.commit_in_task — same code
    path hooks use for round commits, so reliability is consistent.
    """
    subprocess.run(["git", "init"], cwd=task_dir, capture_output=True, check=True)
    ok, info = commit_in_task(task_dir, ["."], "scaffold: baseline")
    if not ok:
        raise RuntimeError(f"scaffold baseline commit failed: {info}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_arg_parser() -> argparse.ArgumentParser:
    """Construct scaffold's argparse, with no side effects.

    Extracted out of main() so parse_args.py can reuse the exact same flag
    spec without duplicating it. Single source of truth for which flags
    /autoresearch accepts and how they're typed/defaulted.
    """
    parser = argparse.ArgumentParser(
        description="Scaffold a task directory for Claude Code autoresearch",
    )
    parser.add_argument("--ref", required=True,
                        help="Path to reference.py (Model/get_inputs format)")
    parser.add_argument("--kernel", required=True,
                        help="Seed kernel .py file, or multi-file DSL project "
                             "directory with sibling kernel.py")
    parser.add_argument("--op-name", default=None,
                        help="Operator name (required)")
    # backend / framework / dsl are pinned per repo in config.yaml's
    # ``defaults`` block. arch is derived from the picked --devices via
    # the Ascend probe (npu-smi).
    parser.add_argument("--devices", default=None,
                        help="Comma-separated device IDs for local eval "
                             "(e.g. '5' or '0,1,2,3'). Required.")
    parser.add_argument("--max-rounds", type=int, default=default_max_rounds())
    parser.add_argument("--eval-timeout", type=int, default=default_eval_timeout())
    parser.add_argument("--output-dir", default=None,
                        help="Parent directory for the task (default: ./ar_tasks/)")
    parser.add_argument("--run-baseline", action="store_true",
                        help="Also run baseline eval after scaffolding")
    # Single flag, store_const so the absence of --no-code-checker yields
    # None (lets defaults.code_checker_enabled in config.yaml decide) and
    # presence yields False (pinned into task.yaml as enabled: false).
    parser.add_argument("--no-code-checker", dest="code_checker",
                        action="store_const", const=False, default=None,
                        help=("Disable the static Triton regression check "
                              "(validate_triton_impl) for this task. "
                              "Useful when the regression rules are too "
                              "strict for the chosen kernel style. Writes "
                              "`code_checker: {enabled: false}` into "
                              "task.yaml; flip the field to re-enable later."))
    parser.add_argument("--worker-url", default="",
                        help="Remote worker URL(s) (host:port, comma-separated). "
                             "Routes eval through the remote HTTP worker "
                             "instead of probing a local device.")
    return parser


def main() -> int:
    args = _make_arg_parser().parse_args()
    try:
        devices, arch = _resolve_hardware(args)
        sources = _prepare_sources(args)
    except ScaffoldCliError as exc:
        _print_json_error(str(exc))
        return 1

    emit(f"[scaffold] Creating task directory for {args.op_name}...")
    request = ScaffoldRequest(
        ref_code=sources.ref_code,
        kernel_code=sources.kernel_code,
        op_name=args.op_name,
        devices=devices,
        arch=arch,
        max_rounds=args.max_rounds,
        eval_timeout=args.eval_timeout,
        output_dir=args.output_dir,
        code_checker_enabled=args.code_checker,
        ref_source_path=args.ref,
        worker_url=args.worker_url,
        kernel_project_src=sources.kernel_project_src,
        editable_filename=sources.entry_filename,
        editable_files=sources.editable_files,
    )
    task_dir = scaffold_task_dir(request)
    _print_created_task(task_dir)
    _bind_batch_task(task_dir, args.op_name)
    if args.run_baseline:
        baseline_result = _run_requested_baseline(task_dir)
        if baseline_result != 0:
            return baseline_result
    summary = task_summary(task_dir) or {}
    emit(json.dumps({
        "task_dir": task_dir,
        "status": "ok",
        "baseline_outcome": summary.get("baseline_outcome"),
    }))
    return 0


def _resolve_hardware(args: argparse.Namespace) -> tuple[list[int], str]:
    has_remote = bool(args.worker_url and args.worker_url.strip())
    if not args.devices and not has_remote:
        raise ScaffoldCliError(
            "--devices (local eval) or --worker-url (remote worker) is required."
        )
    try:
        devices = [
            int(device.strip())
            for device in (args.devices or "").split(",")
            if device.strip()
        ]
    except ValueError as exc:
        raise ScaffoldCliError(f"invalid --devices: {args.devices}") from exc
    if has_remote:
        return devices, ""
    backend = target_backend()
    arch = derive_arch(devices[0], backend=backend)
    if not arch:
        raise ScaffoldCliError(
            f"could not derive arch from device {devices[0]} for "
            f"backend={backend!r} ({probe_hint(backend)})"
        )
    return devices, arch


def _prepare_sources(args: argparse.Namespace) -> PreparedSources:
    if not args.op_name:
        raise ScaffoldCliError("--op-name is required")
    if not os.path.isfile(args.ref):
        raise ScaffoldCliError(f"Reference file not found: {args.ref}")
    with open(args.ref, "r", encoding="utf-8") as reference_file:
        ref_code = reference_file.read()
    try:
        validate_ref(ref_code, args.ref)
    except ValueError as exc:
        raise ScaffoldCliError(str(exc)) from exc

    kernel_path = os.path.abspath(args.kernel)
    if not os.path.exists(kernel_path):
        raise ScaffoldCliError(f"--kernel path not found: {args.kernel}")
    adapter = _scaffold_dsl_adapter()
    try:
        kernel_code, project_src = adapter.read_kernel_source(
            kernel_path,
            op_name=args.op_name,
        )
    except FileNotFoundError as exc:
        raise ScaffoldCliError(str(exc)) from exc
    entry = adapter.entry_filename_template.format(op_name=args.op_name)
    editable = [entry] + list(
        adapter.list_kernel_project_files(
            project_src,
            op_name=args.op_name,
        )
    )
    return PreparedSources(
        ref_code,
        kernel_code,
        project_src,
        entry,
        editable,
    )


def _print_json_error(error: str) -> None:
    emit(json.dumps({"status": "error", "error": error}))


def _print_created_task(task_dir: str) -> None:
    emit(f"[scaffold] Task directory created: {task_dir}")
    emit("[scaffold] Files:")
    for filename in sorted(os.listdir(task_dir)):
        emit(f"  {filename}")


def _bind_batch_task(task_dir: str, op_name: str) -> None:
    batch_dir = os.environ.get("AR_BATCH_DIR")
    batch_op = os.environ.get("AR_BATCH_OP")
    if not batch_dir or batch_op != op_name:
        return
    try:
        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "batch",
            ),
        )
        import manifest as batch_manifest
        batch_manifest.update_case(
            Path(batch_dir),
            op_name,
            task_dir=os.path.abspath(task_dir),
        )
    except (ImportError, OSError, ValueError) as exc:
        emit(f"[scaffold] warning: failed to update batch task_dir: {exc}")


def _run_requested_baseline(task_dir: str) -> int:
    emit("[scaffold] Running baseline eval...")
    result = _run_initial_baseline(task_dir)
    if result == 0:
        return 0
    summary = task_summary(task_dir) or {}
    if result == 4:
        _print_baseline_failure(
            task_dir,
            "eval pipeline broken during baseline — see "
            "[baseline]/[eval] stderr above",
            _baseline_failure_hint(summary.get("baseline_error_source")),
        )
        return 4
    _print_baseline_failure(
        task_dir,
        f"baseline crashed unexpectedly (exit {result}); "
        "see [baseline]/[eval] stderr above",
        "This is not a classified outcome. Inspect the baseline / eval "
        "stderr above and file a bug if the exit code isn't in _EXIT_FOR.",
    )
    return result


def _baseline_failure_hint(error_source: object) -> str:
    if error_source == "ref":
        return (
            "The file passed via --ref is broken (import / forward / "
            "device-only bug). Fix the SOURCE file and re-run /autoresearch "
            "from scratch. The task directory is left for inspection but "
            "MUST NOT be activated — reference.py is not editable."
        )
    return (
        "INFRA_FAIL: no per-shape data — the seed kernel wasn't meaningfully "
        "exercised. Fix env (device / eval.timeout / worker / OOM) and re-run "
        "/autoresearch --resume <task_dir>. Phase stays at BASELINE."
    )


def _print_baseline_failure(task_dir: str, error: str, hint: str) -> None:
    emit(json.dumps({
        "status": "error",
        "task_dir": task_dir,
        "error": error,
        "hint": hint,
    }))


if __name__ == "__main__":
    sys.exit(main())
