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

"""Pre-flight verification for batch directories.

Tier 1 (default, no hardware): compile + best-effort import + required-symbol
check on ref.py (Model + get_inputs/get_input_groups + get_init_inputs) and
the kernel's importable wrapper (ModelNew). If a worker-owned runtime module
is unavailable on the orchestrator, import is deferred to Tier 2 and Tier 1
checks the same exports from the AST instead. Kernel.py additionally runs the
DSL-aware static check via ``op_autoresearch.op.utils.code_checker.CodeChecker``
(syntax / py_compile / import / per-DSL anti-cheat / autotune); each DSL
contributes its own ``_<dsl>ComplianceCheck`` subclass. The same tier-1
path covers Triton (``@triton.jit`` + launch), CATLASS
(``torch.ops.<ns>.*``), and AscendC source checks.
For directory-backed AscendC, tier-1 also scans editable project files
(``.cpp/.h/.asc/CMakeLists.txt``) for CANN-Bench-style fallback/D2H
anti-cheat before Tier-2 touches any worker.

Tier 2 (--full): FORMAL verify-only pass. It materializes a temporary
task directory and calls ``utils.eval_bridge.eval_kernel(...,
verify_only=True)``, so it reuses the same ``KernelVerifier`` +
``DSLAdapter`` chain as batch eval / worker correctness, minus profiling.

For multi-file DSLs (ascendc / ascendc_catlass), ``case["kernel"]`` is a directory
that gets passed to ``/autoresearch --kernel``, while
``case["kernel_module"]`` is the sibling ``kernel.py`` (or
``<op>_kernel.py``) that tier-1 imports and tier-2 verifies.

Each op runs in its own subprocess. Results: <batch_dir>/verify_results.json.

Usage:
    python scripts/batch/verify.py <batch_dir>             # Tier 1
    python scripts/batch/verify.py <batch_dir> --full      # Tier 1 + Tier 2
    python scripts/batch/verify.py <batch_dir> --full --worker-url 127.0.0.1:9111
    python scripts/batch/verify.py <batch_dir> --only op1,op2
    python scripts/batch/verify.py <batch_dir> --full --only op --case-ids 3,7
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from op_autoresearch.utils.console import emit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as mf

# Reach up one level for the shared precision module - single source of
# truth so verify.py and autoresearch's per-round eval can't drift.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.hw_detect import derive_arch, probe_hint
from utils.input_groups import num_cases_from_ref
from utils.settings import (
    batch_tier1_timeout,
    batch_tier2_timeout,
    target_backend,
    target_dsl,
)

VERIFY_RESULTS = "verify_results.json"
# Timeouts (seconds) from config.yaml `batch:`. tier2 cold JIT across ~50
# cases on triton-ascend can take minutes for kernels with many constexpr
# specializations (maxpool3d, ...); warm runs land in tens of seconds.
TIER1_TIMEOUT = batch_tier1_timeout()
TIER2_TIMEOUT = batch_tier2_timeout()


class VerificationTableRow(NamedTuple):
    op_name: str
    tier1_ref: str
    tier1_kernel: str
    sources: str
    tier2: str
    status: str
    message: str


class VerificationRequestError(RuntimeError):
    """Invalid batch verification request."""


@dataclass(frozen=True)
class Tier2Request:
    """Inputs for one formal Tier-2 verification."""

    ref_path: Path
    kernel_path: Path
    worker_url: str = ""
    device_ids: tuple[int, ...] = ()
    case_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class Tier2Setup:
    """Resolved hardware and temporary workspace for Tier-2."""

    request: Tier2Request
    op_name: str
    arch: str
    device_ids: tuple[int, ...]
    adapter: object
    temp_root: Path


@dataclass(frozen=True)
class SubprocessRequest:
    """One isolated verification subprocess."""

    tier: str
    ref: Path
    kernel: Path | None
    timeout: int
    worker_url: str = ""
    device_ids: tuple[int, ...] = ()
    case_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class VerificationOptions:
    """Batch verification policy shared by all cases."""

    full: bool = False
    only: str = ""
    worker_url: str = ""
    devices: str | int = ""
    case_ids: str | int | list[int] | None = None


# Reference must export Model + get_init_inputs + one of (get_inputs,
# get_input_groups). The "input provider" is checked separately (per
# input_groups.resolve duck-type) since either symbol satisfies it.
REF_REQUIRED = ("Model", "get_init_inputs")
REF_INPUT_PROVIDERS = ("get_inputs", "get_input_groups")
KERNEL_REQUIRED = ("ModelNew",)

CASE_FILTER_TEMPLATE = """

# --- verify.py case-id filter ---
_VERIFY_SELECTED_CASE_IDS = {__SELECTED_IDS__}
if _VERIFY_SELECTED_CASE_IDS:
    try:
        _VERIFY_FILTERED_CASES = []
        for _case_ordinal, _case in enumerate(CASES, 1):
            try:
                _case_id = int(_case.get("case_id") or _case_ordinal)
            except AttributeError:
                _case_id = _case_ordinal
            if _case_id in _VERIFY_SELECTED_CASE_IDS:
                _VERIFY_FILTERED_CASES.append(_case)
        CASES = _VERIFY_FILTERED_CASES
    except NameError:
        raise RuntimeError("verify.py --case-ids requires reference.py CASES")
    if not CASES:
        raise RuntimeError(
            f"verify.py --case-ids selected no cases: "
            f"{sorted(_VERIFY_SELECTED_CASE_IDS)}"
        )
# --- end verify.py case-id filter ---
"""


# ---------------------------------------------------------------------------
# Subprocess pool (this same file is re-invoked with --tier-runner)
# ---------------------------------------------------------------------------
def inspect_tier1(path: Path, required: tuple[str, ...]) -> dict:
    """Compile, import, inspect exports, and validate a kernel wrapper."""
    out = _empty_tier1_result(path)
    try:
        source = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        out.update(compile="FAIL", msg=f"read error: {exc}")
        return out
    if not _compile_tier1_source(source, path, out):
        return out
    module = _import_tier1_module(path, out)
    if module is None and out["import"] != "SKIP":
        return out
    missing = (
        _missing_static_exports(source, required)
        if module is None
        else _missing_exports(module, required)
    )
    if missing:
        out.update(
            exports="FAIL",
            missing=missing,
            msg=f"missing: {', '.join(missing)}",
        )
        return out
    out["exports"] = "PASS"
    if required is KERNEL_REQUIRED:
        _validate_kernel_source(source, path, out)
    return out


def _empty_tier1_result(path: Path) -> dict:
    return {
        "path": str(path),
        "compile": "skip",
        "import": "skip",
        "exports": "skip",
        "validate": "skip",
        "missing": [],
        "msg": "",
    }


def _compile_tier1_source(source: str, path: Path, out: dict) -> bool:
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        out.update(
            compile="FAIL",
            msg=f"syntax error line {exc.lineno}: {exc.msg}",
        )
        return False
    out["compile"] = "PASS"
    return True


def _import_tier1_module(path: Path, out: dict):
    import importlib.util

    try:
        spec = importlib.util.spec_from_file_location(
            f"_verify_{path.stem}",
            str(path),
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not build spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        if _is_worker_runtime_missing(exc):
            out["import"] = "SKIP"
            out["msg"] = f"worker runtime import deferred: {exc.name}"
            return None
        out["import"] = "FAIL"
        out["msg"] = f"{type(exc).__name__}: {exc}"
        return None
    except Exception as exc:
        out["import"] = "FAIL"
        out["msg"] = f"{type(exc).__name__}: {exc}"
        return None
    out["import"] = "PASS"
    return module


def _missing_exports(module, required: tuple[str, ...]) -> list[str]:
    missing = [name for name in required if not hasattr(module, name)]
    if required is REF_REQUIRED and not any(
        hasattr(module, name)
        for name in REF_INPUT_PROVIDERS
    ):
        missing.append(" or ".join(REF_INPUT_PROVIDERS))
    return missing


def _is_worker_runtime_missing(exc: ModuleNotFoundError) -> bool:
    missing = str(exc.name or "").split(".", 1)[0]
    return bool(missing and missing in _worker_runtime_modules())


def _worker_runtime_modules() -> frozenset[str]:
    from op_autoresearch.op.utils.code_checker import CodeChecker

    return CodeChecker.worker_runtime_modules(target_dsl())


def _missing_static_exports(
    source: str,
    required: tuple[str, ...],
) -> list[str]:
    names = _static_export_names(source)
    missing = [name for name in required if name not in names]
    if required is REF_REQUIRED and not any(
        name in names for name in REF_INPUT_PROVIDERS
    ):
        missing.append(" or ".join(REF_INPUT_PROVIDERS))
    return missing


def _static_export_names(source: str) -> set[str]:
    """Return names bound directly in a module without executing it."""
    names: set[str] = set()
    for node in ast.parse(source).body:
        names.update(_top_level_export_names(node))
    return names


def _top_level_export_names(node: ast.stmt) -> set[str]:
    if isinstance(
        node,
        (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
    ):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {
            alias.asname or alias.name.split(".", 1)[0]
            for alias in node.names
            if alias.name != "*"
        }
    if isinstance(node, ast.Assign):
        names: set[str] = set()
        for target in node.targets:
            names.update(_bound_names(target))
        return names
    if isinstance(node, ast.AnnAssign):
        return _bound_names(node.target)
    return set()


def _bound_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in target.elts:
            names.update(_bound_names(item))
        return names
    return set()


def _validate_kernel_source(source: str, path: Path, out: dict) -> None:
    _apply_code_check(source, path, out)


def _apply_code_check(source: str, path: Path, out: dict) -> None:
    """Run the configured DSL checker and update a Tier-1 result."""
    from op_autoresearch.op.utils.code_checker import CodeChecker

    try:
        passed, error_msg, errors = CodeChecker(
            backend=target_backend(),
            dsl=target_dsl(),
        ).check(source, task_info={"file": str(path)})
    except Exception as exc:
        out.update(
            validate="FAIL",
            msg=f"checker raised: {type(exc).__name__}: {exc}"[:160],
        )
        return
    out["validate"] = "PASS" if passed else "FAIL"
    if passed:
        return
    if errors:
        first = errors[0]
        out["msg"] = (
            f"L{first.get('line', 0)} "
            f"{first.get('error_type', '?')}: "
            f"{first.get('detail', '')}"
        )[:160]
    else:
        out["msg"] = (
            error_msg.splitlines()[0]
            if error_msg
            else "regression detected"
        )[:160]


def _tier1_static_check(path: Path) -> dict:
    """Run CodeChecker on a source file that is not necessarily Python.

    Directory-backed DSLs such as AscendC expose C++/AscendC/CMake files
    in addition to kernel.py. These files must be scanned for fallback
    compute and D2H egress before Tier-2 touches a worker.
    """
    out: dict = {"path": str(path), "compile": "skip", "import": "skip",
                 "exports": "PASS", "validate": "skip", "missing": [], "msg": ""}
    try:
        src = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as e:
        out["validate"] = "FAIL"
        out["msg"] = f"read error: {e}"
        return out

    _apply_code_check(src, path, out)
    return out


def _parse_device_ids(devices: str | int | list[int] | None) -> list[int]:
    if devices is None:
        return []
    if isinstance(devices, list):
        return [int(x) for x in devices]
    text = str(devices).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_case_ids(case_ids: str | int | list[int] | None) -> list[int]:
    if case_ids is None:
        return []
    if isinstance(case_ids, list):
        return [int(x) for x in case_ids]
    text = str(case_ids).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _tier2_subprocess_timeout(ref_path: Path,
                              case_ids: list[int] | None = None) -> int:
    """Parent wall-clock cap for the Tier-2 runner. ``_tier2_run`` hands the
    PER-SHAPE ``TIER2_TIMEOUT`` to eval_bridge, which expands it by num_cases
    before dispatch; the parent must mirror that expansion (same num_cases
    SSOT, ``input_groups.num_cases_from_ref``) or it kills a multi-shape
    verify mid-run and loses the per-case sidecar/logs. ``+ TIER1_TIMEOUT``
    covers the runner's own import/compile preamble.
    """
    num_cases = len(case_ids or []) or num_cases_from_ref(ref_path)
    return TIER2_TIMEOUT * num_cases + TIER1_TIMEOUT


def _tier2_run(request: Tier2Request) -> dict:
    """Run the formal KernelVerifier-backed path in a temporary task."""
    out: dict = {"status": "skip", "msg": "", "max_abs_diff": None}
    setup = None
    try:
        setup = _build_tier2_setup(request)
        raw = execute_tier2(setup)
    except Exception as exc:
        out.update(
            status="ERROR",
            msg=f"formal verify setup failed: {type(exc).__name__}: {exc}",
        )
        return out
    finally:
        if setup is not None:
            _cleanup_tier2(setup, out)
    return _format_tier2_result(out, raw)


def _build_tier2_setup(request: Tier2Request) -> Tier2Setup:
    import tempfile

    from op_autoresearch.op.verifier.adapters.factory import get_dsl_adapter

    op_name = _op_name_from_ref(request.ref_path)
    backend = target_backend()
    device_ids = request.device_ids or ((0,) if not request.worker_url else ())
    arch = ""
    if not request.worker_url:
        arch = derive_arch(device_ids[0], backend=backend) or ""
        if not arch:
            raise RuntimeError(
                f"could not derive local arch for backend={backend!r} "
                f"({probe_hint(backend)}); pass --worker-url for remote "
                "worker verify"
            )
    temp_root = Path(tempfile.mkdtemp(prefix=f"_batch_verify_{op_name}_"))
    return Tier2Setup(
        request=request,
        op_name=op_name,
        arch=arch,
        device_ids=tuple(device_ids),
        adapter=get_dsl_adapter(target_dsl()),
        temp_root=temp_root,
    )


def _op_name_from_ref(path: Path) -> str:
    stem = path.stem
    return stem[:-4] if stem.endswith("_ref") else stem


def _filtered_ref_code(path: Path, case_ids: tuple[int, ...]) -> str:
    source = path.read_text(encoding="utf-8-sig")
    if not case_ids:
        return source
    selected = ", ".join(str(case_id) for case_id in case_ids)
    patch = CASE_FILTER_TEMPLATE.replace(
        "{__SELECTED_IDS__}",
        f"{{{selected}}}",
    )
    return source.rstrip() + patch + "\n"


def _project_inputs(setup: Tier2Setup) -> tuple[str, list[str], Path | None]:
    adapter = setup.adapter
    entry_name = adapter.entry_filename_template.format(op_name=setup.op_name)
    editable_files = [entry_name]
    if not adapter.kernel_arg_is_directory:
        return entry_name, editable_files, None
    project_name = adapter.kernel_project_dir_name
    if not project_name:
        raise RuntimeError(
            f"{type(adapter).__name__} is directory-backed but has no "
            "kernel_project_dir_name"
        )
    project_src = setup.request.kernel_path.parent / project_name
    for relative in adapter.list_kernel_project_files(
        str(project_src),
        op_name=setup.op_name,
    ):
        if relative not in editable_files:
            editable_files.append(relative)
    return entry_name, editable_files, project_src


def _scaffold_tier2_task(setup: Tier2Setup) -> Path:
    from scaffold import ScaffoldRequest, scaffold_task_dir

    entry_name, editable_files, project_src = _project_inputs(setup)
    request = setup.request
    return Path(
        scaffold_task_dir(ScaffoldRequest(
            ref_code=_filtered_ref_code(request.ref_path, request.case_ids),
            kernel_code=request.kernel_path.read_text(encoding="utf-8-sig"),
            op_name=setup.op_name,
            arch=setup.arch,
            devices=list(setup.device_ids),
            max_rounds=1,
            eval_timeout=TIER2_TIMEOUT,
            output_dir=str(setup.temp_root),
            editable_filename=entry_name,
            editable_files=editable_files,
            kernel_project_src=str(project_src) if project_src else None,
            ref_source_path=str(request.ref_path),
            worker_url=request.worker_url,
        ))
    )


def execute_tier2(setup: Tier2Setup) -> dict:
    from task_config.loader import load_task_config
    from utils.eval_bridge import EvalRequest, eval_kernel as formal_eval

    task_dir = _scaffold_tier2_task(setup)
    config = load_task_config(str(task_dir))
    if config is None:
        raise RuntimeError("load_task_config returned None")
    return formal_eval(EvalRequest(
        task_dir=str(task_dir),
        config=config,
        device_id=list(setup.device_ids) or None,
        worker_url=setup.request.worker_url or None,
        current_step=0,
        verify_only=True,
    ))


def _cleanup_tier2(setup: Tier2Setup, out: dict) -> None:
    if os.environ.get("AR_KEEP_BATCH_VERIFY_TEMP") == "1":
        out["temp_dir"] = str(setup.temp_root)
        return
    import shutil

    shutil.rmtree(setup.temp_root, ignore_errors=True)


def _format_tier2_result(out: dict, raw: dict) -> dict:
    from utils.failure_extractor import summarize_one_line

    metrics = raw.get("metrics") or {}
    out["max_abs_diff"] = metrics.get("max_abs_diff")
    out["num_cases"] = int(metrics.get("num_cases") or 1)
    per_case = list(raw.get("per_case") or [])
    if per_case:
        out["per_case"] = per_case
    if raw.get("outcome") == "ok":
        out.update(status="PASS", msg=f"OK (n={out['num_cases']})")
        return out
    signals = raw.get("failure_signals") or {}
    failure_kinds = {
        str(item.get("failure_kind") or "")
        for item in per_case
        if isinstance(item, dict)
    }
    is_kernel_miss = (
        "kernel_miss" in failure_kinds
        or signals.get("primary") == "precision_fail"
    )
    out["status"] = "FAIL" if is_kernel_miss else "ERROR"
    out["msg"] = (
        summarize_one_line(signals)
        or str(raw.get("error") or "formal verify failed")
    )[:160]
    out["raw_output_tail"] = str(raw.get("raw_output_tail") or "")[-4000:]
    return out


def _run_tier_subprocess() -> int:
    """Subprocess entry point. Writes JSON to a sidecar path on stdout's last line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("1ref", "1kernel", "1source", "2"), required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--kernel", default="")
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--worker-url", default="")
    ap.add_argument("--device-ids", default="")
    ap.add_argument("--case-ids", default="")
    args = ap.parse_args(sys.argv[2:])  # skip the --tier-runner sentinel

    ref_path = Path(args.ref)
    kernel_path = Path(args.kernel) if args.kernel else None

    if args.tier == "1ref":
        result = inspect_tier1(ref_path, REF_REQUIRED)
    elif args.tier == "1kernel":
        result = inspect_tier1(kernel_path, KERNEL_REQUIRED)
    elif args.tier == "1source":
        result = _tier1_static_check(kernel_path)
    else:  # tier == "2"
        result = _tier2_run(Tier2Request(
            ref_path,
            kernel_path,
            args.worker_url,
            tuple(_parse_device_ids(args.device_ids)),
            tuple(_parse_case_ids(args.case_ids)),
        ))

    Path(args.sidecar).write_text(json.dumps(result), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _run_subprocess(request: SubprocessRequest) -> dict:
    sidecar = _sidecar_path(request)
    sidecar.unlink(missing_ok=True)
    cmd = _subprocess_command(request, sidecar)
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    started = time.time()
    try:
        process = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=request.timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "ERROR",
            "msg": f"timeout after {request.timeout}s",
            "elapsed_s": round(time.time() - started, 2),
        }
    elapsed = round(time.time() - started, 2)
    return _read_subprocess_result(sidecar, process, elapsed)


def _sidecar_path(request: SubprocessRequest) -> Path:
    name = f"_verify_{os.getpid()}_{request.tier}_{request.ref.stem}.json"
    return Path(os.environ.get("TMP", "/tmp")) / name


def _subprocess_command(
    request: SubprocessRequest,
    sidecar: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--tier-runner",
        "--tier",
        request.tier,
        "--ref",
        str(request.ref),
        "--sidecar",
        str(sidecar),
    ]
    if request.kernel is not None:
        cmd += ["--kernel", str(request.kernel)]
    if request.tier != "2":
        return cmd
    if request.worker_url:
        cmd += ["--worker-url", request.worker_url]
    if request.device_ids:
        cmd += ["--device-ids", ",".join(map(str, request.device_ids))]
    if request.case_ids:
        cmd += ["--case-ids", ",".join(map(str, request.case_ids))]
    return cmd


def _read_subprocess_result(
    sidecar: Path,
    process: subprocess.CompletedProcess,
    elapsed: float,
) -> dict:
    if not sidecar.exists():
        return {
            "status": "ERROR",
            "msg": f"no result; rc={process.returncode}",
            "stderr_tail": (process.stderr or process.stdout)[-400:],
            "elapsed_s": elapsed,
        }
    try:
        result = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "ERROR",
            "msg": f"parse sidecar: {exc}",
            "elapsed_s": elapsed,
        }
    finally:
        sidecar.unlink(missing_ok=True)
    result["elapsed_s"] = elapsed
    return result


def _verify_one(case: dict, options: VerificationOptions) -> dict:
    op_name = case["op_name"]
    ref_path = Path(case["ref"])
    kernel_path = Path(case.get("kernel_module") or case["kernel"])
    result = {
        "op_name": op_name,
        "tier1_ref": _run_subprocess(SubprocessRequest(
            "1ref",
            ref_path,
            None,
            TIER1_TIMEOUT,
        )),
        "tier1_kernel": _run_subprocess(SubprocessRequest(
            "1kernel",
            ref_path,
            kernel_path,
            TIER1_TIMEOUT,
        )),
        "tier1_sources": _verify_project_sources(
            case,
            op_name,
            ref_path,
            kernel_path,
        ),
        "tier2": None,
    }
    tier1_ok = (
        _tier1_pass(result["tier1_ref"])
        and _tier1_pass(result["tier1_kernel"])
        and all(
            source.get("validate") != "FAIL"
            for source in result["tier1_sources"]
        )
    )
    if options.full:
        result["tier2"] = _tier2_result(
            ref_path,
            kernel_path,
            options,
            tier1_ok,
        )
    return result


def _verify_project_sources(
    case: dict,
    op_name: str,
    ref_path: Path,
    kernel_path: Path,
) -> list[dict]:
    try:
        from op_autoresearch.op.verifier.adapters.factory import get_dsl_adapter
        adapter = get_dsl_adapter(target_dsl())
    except (ImportError, KeyError, ValueError):
        return []
    project_path = Path(case["kernel"])
    if not adapter.kernel_arg_is_directory or not project_path.is_dir():
        return []
    results = []
    for relative in adapter.list_kernel_project_files(
        str(project_path),
        op_name=op_name,
    ):
        source_path = project_path.parent / relative
        if source_path != kernel_path:
            results.append(_run_subprocess(SubprocessRequest(
                "1source",
                ref_path,
                source_path,
                TIER1_TIMEOUT,
            )))
    return results


def _tier1_pass(record: dict | None) -> bool:
    return bool(
        record
        and record.get("exports") == "PASS"
        and record.get("validate") != "FAIL"
    )


def _tier2_result(
    ref_path: Path,
    kernel_path: Path,
    options: VerificationOptions,
    tier1_ok: bool,
) -> dict:
    if not tier1_ok:
        return {
            "status": "skip",
            "msg": "tier1 failed; skipping tier2",
            "elapsed_s": 0,
        }
    device_ids = tuple(_parse_device_ids(options.devices))
    case_ids = tuple(_parse_case_ids(options.case_ids))
    return _run_subprocess(SubprocessRequest(
        "2",
        ref_path,
        kernel_path,
        _tier2_subprocess_timeout(ref_path, list(case_ids)),
        options.worker_url,
        device_ids,
        case_ids,
    ))


_CONTENT_FAIL_FIELDS = ("compile", "import", "exports", "validate")


def _summary_status(record: dict, full: bool) -> str:
    """P/F/E/S single-letter. compile/import/exports/validate failures
    all map to F (matches the per-tier table column); runtime ERROR is E.
    """
    t1r = record["tier1_ref"]
    t1k = record["tier1_kernel"]
    t1s = record.get("tier1_sources") or []
    t2 = record["tier2"]

    def _bad(t):
        return t and ("FAIL" in (t.get("compile"), t.get("import"),
                                  t.get("exports"), t.get("validate"))
                      or t.get("status") in ("FAIL", "ERROR"))

    def _content_fail(t):
        return t and any(t.get(f) == "FAIL" for f in _CONTENT_FAIL_FIELDS)

    if _bad(t1r):
        return "F" if _content_fail(t1r) else "E"
    if _bad(t1k):
        return "F" if _content_fail(t1k) else "E"
    for src in t1s:
        if _bad(src):
            return "F" if _content_fail(src) else "E"
    if full and t2:
        if t2.get("status") == "PASS":
            return "P"
        if t2.get("status") == "FAIL":
            return "F"
        if t2.get("status") == "ERROR":
            return "E"
        return "S"
    return "P"


def _print_table(results: dict, full: bool) -> None:
    rows = [
        _table_row(op_name, record, full)
        for op_name, record in results.items()
    ]
    op_width = max(8, max(len(row[0]) for row in rows))
    emit(
        f"  {'op':<{op_width}}  {'t1_ref':<6}  {'t1_kern':<7}  "
        f"{'t1_src':<6}  {'t2':<6}  {'ok':<3}  note"
    )
    emit(
        f"  {'-' * op_width}  {'-' * 6}  {'-' * 7}  {'-' * 6}  "
        f"{'-' * 6}  {'-' * 3}  {'-' * 60}"
    )
    for op_name, tier1_ref, tier1_kernel, sources, tier2, status, message in rows:
        emit(
            f"  {op_name:<{op_width}}  {tier1_ref:<6}  "
            f"{tier1_kernel:<7}  {sources:<6}  {tier2:<6}  "
            f"{status:<3}  {message}"
        )


def _table_row(op_name: str, record: dict, full: bool) -> VerificationTableRow:
    tier1_ref = record["tier1_ref"]
    tier1_kernel = record["tier1_kernel"]
    sources = record.get("tier1_sources") or []
    tier2 = record["tier2"]
    return VerificationTableRow(
        op_name,
        _tier1_column(tier1_ref, check_validation=False),
        _tier1_column(tier1_kernel, check_validation=True),
        _source_column(sources),
        tier2.get("status", "?") if full and tier2 is not None else "-",
        _summary_status(record, full),
        _informative_message(tier2, sources, tier1_kernel, tier1_ref)[:70],
    )


def _tier1_column(record: dict | None, check_validation: bool) -> str:
    if record is None:
        return "-"
    if check_validation and record.get("validate") == "FAIL":
        return "FAIL"
    if record.get("exports") == "PASS":
        return "PASS"
    if (
        record.get("exports") == "FAIL"
        or record.get("compile") == "FAIL"
        or record.get("import") == "FAIL"
    ):
        return "FAIL"
    return "ERROR"


def _source_column(sources: list[dict]) -> str:
    if not sources:
        return "-"
    return (
        "FAIL"
        if any(source.get("validate") == "FAIL" for source in sources)
        else "PASS"
    )


def _informative_message(
    tier2: dict | None,
    sources: list[dict],
    tier1_kernel: dict | None,
    tier1_ref: dict | None,
) -> str:
    message = ""
    for record in (tier2, *sources, tier1_kernel, tier1_ref):
        if not record or not record.get("msg") or record.get("msg") == "OK":
            continue
        message = record["msg"]
        content_failed = "FAIL" in (
            record.get("compile"),
            record.get("import"),
            record.get("exports"),
        )
        if content_failed or record.get("status") in ("FAIL", "ERROR"):
            break
    return message


def run_verification(
    batch_dir: Path,
    options: VerificationOptions | None = None,
) -> int:
    """Run both verification tiers and return zero only when all pass."""
    options = options or VerificationOptions()
    batch_dir = Path(batch_dir).resolve()
    cases = _load_verification_cases(batch_dir, options.only)
    _print_verification_header(batch_dir, cases, options)
    started = time.time()
    results = _run_verification_cases(cases, options)

    from utils.eval_summary import write_artifact
    result_path = write_artifact(
        batch_dir / VERIFY_RESULTS,
        {
            "full": options.full,
            "precision": "cannbench-mere-mare",
            "results": results,
        },
    )
    emit()
    _print_table(results, full=options.full)
    emit()
    return _print_verification_totals(
        results,
        options.full,
        started,
        result_path,
    )


def _load_verification_cases(batch_dir: Path, only: str) -> list[dict]:
    if not batch_dir.is_dir():
        raise VerificationRequestError(f"batch dir not found: {batch_dir}")
    try:
        manifest_path = mf.find_manifest(batch_dir)
        manifest_data = mf.load_manifest(manifest_path)
        cases = mf.resolve_cases(batch_dir, manifest_data, "ref-kernel")
    except mf.ManifestError as exc:
        raise VerificationRequestError(str(exc)) from exc
    selected = {name.strip() for name in only.split(",") if name.strip()}
    if selected:
        cases = [case for case in cases if case["op_name"] in selected]
        if not cases:
            raise VerificationRequestError("--only filtered out all ops")
    return cases


def _print_verification_header(
    batch_dir: Path,
    cases: list[dict],
    options: VerificationOptions,
) -> None:
    emit(
        f"verify  batch_dir={batch_dir}  "
        f"tier={'1+2' if options.full else '1'}  ops={len(cases)}  "
        "precision: KernelVerifier/CANN-Bench MERE/MARE for AscendC"
    )
    device_ids = _parse_device_ids(options.devices)
    if options.full and options.worker_url:
        device_text = (
            ",".join(map(str, device_ids))
            if device_ids
            else "worker-declared"
        )
        emit(f"  worker_url={options.worker_url}  devices={device_text}")
    case_ids = _parse_case_ids(options.case_ids)
    if case_ids:
        emit(f"  case_ids={','.join(map(str, case_ids))}")
    emit()


def _run_verification_cases(
    cases: list[dict],
    options: VerificationOptions,
) -> dict:
    results = {}
    for index, case in enumerate(cases, 1):
        op_name = case["op_name"]
        emit(f"  [{index:>3}/{len(cases)}] {op_name} ... ", end="", flush=True)
        record = _verify_one(case, options)
        results[op_name] = record
        emit(_summary_status(record, options.full), flush=True)
    return results


def _print_verification_totals(
    results: dict,
    full: bool,
    started: float,
    result_path: Path,
) -> int:
    statuses = [
        _summary_status(record, full)
        for record in results.values()
    ]
    failed = statuses.count("F")
    errors = statuses.count("E")
    emit(
        f"  total={len(results)}  pass={statuses.count('P')}  "
        f"fail={failed}  error={errors}  "
        f"elapsed={time.time() - started:.1f}s"
    )
    emit(f"  results: {result_path}")
    return 0 if failed == 0 and errors == 0 else 1


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--tier-runner":
        return _run_tier_subprocess()

    ap = argparse.ArgumentParser(description="Pre-flight verify for batch directories.")
    ap.add_argument("batch_dir")
    ap.add_argument("--full", action="store_true",
                    help="also run Tier 2 (formal verify-only path via KernelVerifier); "
                         "needs the same hardware /autoresearch eval would use")
    ap.add_argument("--only", default="",
                    help="comma-separated op names")
    ap.add_argument("--worker-url", default="",
                    help="remote worker URL for --full Tier 2, e.g. 127.0.0.1:9111")
    ap.add_argument("--devices", default="",
                    help="optional device id/list filter for --full Tier 2; "
                         "omitted with --worker-url lets the worker declare/allocate")
    ap.add_argument("--case-ids", default="",
                    help="optional comma-separated CANN-Bench case ids for Tier 2")
    args = ap.parse_args()

    try:
        return run_verification(
            Path(args.batch_dir),
            VerificationOptions(
                full=args.full,
                only=args.only,
                worker_url=args.worker_url,
                devices=args.devices,
                case_ids=args.case_ids,
            ),
        )
    except VerificationRequestError as exc:
        emit(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
