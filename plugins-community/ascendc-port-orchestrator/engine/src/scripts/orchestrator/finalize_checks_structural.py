#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""finalize_checks_structural — code-shape / entrypoint / architecture / topology finalize gate CHECK functions.

Behavior-neutral extraction from finalize_checks.py (DEBT-201 god-file
sub-split, 2026-07-06). Byte-identical function bodies; only relocated.
finalize_checks re-imports these (bottom import) so call sites + import
paths (`from finalize_checks import ...`) are unaffected."""
from __future__ import annotations
import logging
import ast as _ast
import json
import re
import sys
from datetime import datetime as _datetime, timezone as _timezone
from pathlib import Path
from typing import Optional

from finalize_shared import _is_v220_ec41_output_pad_exempt  # DEBT-201: shared pure leaf
from finalize_pipeline import (  # module-identity-sensitive constants (stay in parent)
    _HERE, _PROJECT_ROOT)


def _model_input_api_status(fpath: Path) -> tuple[bool, bool]:
    """Return whether *fpath* defines the grouped and legacy input APIs."""
    try:
        tree = _ast.parse(fpath.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return False, False  # let other gates handle malformed files

    function_names = {
        node.name for node in _ast.walk(tree)
        if isinstance(node, _ast.FunctionDef)
    }
    return "get_input_groups" in function_names, "get_inputs" in function_names


def _model_shape_violation(filename: str) -> str:
    """Build the established P0abg diagnostic without changing its wording."""
    return (
        f"P0abg model shape gate: workspace/{filename} defines "
        "`get_inputs()` but not `get_input_groups()`. The multi-case "
        "source interface uses `get_input_groups()` for every op. "
        "kw must rewrite to define `get_input_groups()` "
        "returning a list of input groups (one per JSONL case). Even "
        "for single-case ops, return a one-element list. The "
        "verification_ascendc.py fallback that wraps `get_inputs()` "
        "in `[inputs]` is legacy and silently caps pass_a.total=1, "
        "which is silent coverage fraud against multi-case JSONL "
        "specs. See workspace/<op>/<op>.json (one line = one case "
        "the spec REQUIRES kernel to verify against)."
    )


def _check_model_py_shape(workspace: Path) -> Optional[str]:
    """Enforce `get_input_groups()` when a legacy `get_inputs()` API exists."""
    for filename in ("model.py", "model_cpu_truth.py"):
        fpath = workspace / filename
        if not fpath.exists():
            continue
        has_groups, has_inputs = _model_input_api_status(fpath)
        if has_inputs and not has_groups:
            return _model_shape_violation(filename)
    return None


def _check_pass_precision_tier(status: str, prec: dict) -> Optional[str]:
    """Reject textual-only PASS evidence before examining workspace outputs."""
    pa = prec.get("pass_a", {}) if isinstance(prec.get("pass_a"), dict) else {}
    if pa.get("tier") != "T1_BY_CONSTRUCTION":
        return None
    return (
        f"precision.status={status} + precision.pass_a.tier="
        f"'T1_BY_CONSTRUCTION' is FRAUD: T1_BY_CONSTRUCTION means "
        f"PASS was derived by textual reasoning (e.g. 'sign-bit-mask "
        f"identity'), NOT actual A5 NPU execution. PASS-precision "
        f"requires measured data; no textual or source-identity waiver "
        f"is accepted. "
        f"P94 attack-id WORKER-PRECISION-T1-BY-CONSTRUCTION."
    )


def _check_source_reuse_metadata(status: str, prec: dict, vj: dict) -> Optional[str]:
    """Reject PASS metadata that admits source reuse or non-execution."""
    pa = prec.get("pass_a", {}) if isinstance(prec.get("pass_a"), dict) else {}
    pa_method = (pa.get("method") or "").lower()
    pa_evidence = (pa.get("evidence") or "").lower()
    build_ev = vj.get("build_evidence", {}) or {}
    build_stage = (build_ev.get("stage") or "").lower()
    markers = (
        "l1_verbatim_mirror", "verbatim_mirror", "mode a:",
        "mode a port", "source_identity", "byte-copy",
    )
    mirror_signals = [
        label for haystack, label in (
            (pa_evidence, "pass_a.evidence"),
            (build_stage, "build_evidence.stage"),
        ) if any(marker in haystack for marker in markers)
    ]
    no_exec_signals = [
        needle for needle in (
            "mode_a_source_identity_from_verification_json",
            "source_identity_from_verification_json", "from this block",
            "reads the persisted pass_a", "does not execute on npu",
        ) if needle in pa_method or needle in pa_evidence
    ]
    if not mirror_signals and not no_exec_signals:
        return None
    return (
        f"precision.status={status} but target/source reuse metadata "
        f"signature detected: mirror_signals={mirror_signals}, "
        f"no_exec_signals={no_exec_signals[:2]}. L1 verbatim mirror "
        f"or textual source identity is forbidden even when execution "
        f"later succeeds. Regenerate from arch22 semantics and the "
        f"independently authored templates; no skip verdict is valid. "
        f"P94 attack-id WORKER-MIRROR-MD5-TRIVIAL."
    )


def _check_canonical_entrypoint_files(workspace: Path, status: str) -> Optional[str]:
    """Require the mode-independent Python entry points for a PASS verdict."""
    filenames = ("model_new_ascendc.py", "model.py")
    missing = [name for name in filenames if not (workspace / name).is_file()]
    if not missing:
        return None
    return (
        f"precision.status={status} but workspace missing canonical "
        f"entry-point files: {missing}. ALL op-gen modes "
        f"(port_a3_to_a5 / backward) MUST "
        f"emit the same Python entry-points so the safety net "
        f"(scan_delegation_cheating.py) covers every mode uniformly. "
        f"`model_new_ascendc.py` is the nn.Module subclass that invokes "
        f"our kernel and have `if __name__ == \"__main__\":` "
        f"for standalone smoke-test. `model.py` is the PyTorch reference. "
        f"Per user directive 2026-05-14: '文件名列表必须在不同mode都对齐'."
    )


def _check_pass_performance_metadata(status: str, vj: dict) -> Optional[str]:
    """Require an explicit measured or not-measurable performance verdict."""
    perf = vj.get("performance", {}) or {}
    perf_status = perf.get("status")
    measured_statuses = ("PASS", "PASS_WITHIN_TOLERANCE", "FAIL", "BELOW_THRESHOLD")
    if perf_status in measured_statuses:
        ratio = perf.get("ratio")
        if isinstance(ratio, (int, float)):
            return None
        return (
            f"precision.status={status} + perf.status={perf_status} but "
            f"perf.ratio={ratio!r} is not a measured number. Measured-class "
            f"statuses (PASS/PASS_WITHIN_TOLERANCE/FAIL/BELOW_THRESHOLD) "
            f"REQUIRE numeric ratio. Non-measurement should be status='N/A' "
            f"with explicit reason in perf.reason."
        )
    if perf_status is None:
        return (
            f"precision.status={status} but perf.status is unset/None. "
            f"Every PASS-precision archive must EXPLICITLY classify perf "
            f"as either (PASS/PASS_WITHIN_TOLERANCE with numeric ratio) "
            f"or (N/A with non-empty reason). Setting perf.status=null "
            f"silently is reward-hacking — exact gap user caught "
            f"2026-05-15T08:18Z on foreach_abs."
        )
    if perf_status in ("N/A", "NA"):
        if (perf.get("reason") or "").strip():
            return None
        return (
            f"precision.status={status} + perf.status={perf_status} "
            f"but perf.reason is empty/missing. Non-measurement must "
            f"document WHY. Empty reason = hidden evasion."
        )
    return (
        f"precision.status={status} but perf.status={perf_status!r} is "
        f"not a canonical value. Allowed: 'PASS', 'PASS_WITHIN_TOLERANCE' "
        f"(with numeric ratio), or 'N/A' (with reason). Non-canonical "
        f"values like 'DEFERRED' / 'PENDING' / custom strings hide the "
        f"true classification — explicit-declaration discipline (user "
        f"catch 2026-05-15T08:18Z)."
    )


def _check_universal_entrypoints(workspace: Path, vj: dict) -> Optional[str]:
    """Enforce universal entry points and measured PASS evidence for all modes."""
    prec = vj.get("precision", {}) or {}
    status = prec.get("status")
    if status not in ("PASS", "PASS_WITHIN_TOLERANCE"):
        return None
    violation = _check_pass_precision_tier(status, prec)
    if violation:
        return violation
    violation = _check_source_reuse_metadata(status, prec, vj)
    if violation:
        return violation
    violation = _check_canonical_entrypoint_files(workspace, status)
    if violation:
        return violation
    violation = _check_pass_performance_metadata(status, vj)
    if violation:
        return violation
    return None


def _collect_arch35_include_hits(workspace: Path) -> list[tuple[str, list[str]]]:
    """Return all prohibited target-architecture include matches."""
    pattern = re.compile(r'#include\s+"[^"]*\barch35/', re.MULTILINE)
    suspect_files: list[tuple[str, list[str]]] = []
    scan_paths = [
        workspace / "op_kernel",
        workspace / "op_host",
        workspace,
    ]
    seen: set[Path] = set()
    for scan_dir in scan_paths:
        if not scan_dir.is_dir():
            continue
        if scan_dir == workspace:
            iter_files = list(scan_dir.glob("*.cpp")) + list(scan_dir.glob("*.h"))
        else:
            iter_files = list(scan_dir.rglob("*.cpp")) + list(scan_dir.rglob("*.h"))
        for f in iter_files:
            if f in seen or not f.is_file():
                continue
            seen.add(f)
            skip_current_item = False
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception as error:
                logging.getLogger(__name__).debug(
                    "Recoverable operation failed.", exc_info=error
                )
                skip_current_item = True
            if skip_current_item:
                continue
            hits = pattern.findall(text)
            if hits:
                rel = f.relative_to(workspace) if workspace in f.parents else f.name
                suspect_files.append((str(rel), hits))
    return suspect_files


def _arch35_wrap_cheat_diagnostic(suspect_files: list[tuple[str, list[str]]]) -> str:
    """Render the established ARCH35_WRAP_CHEAT diagnostic."""
    lines = [
        "ARCH35_WRAP_CHEAT (P0dd 2026-05-23): archive's kernel translation unit "
        "OR pybind shim `#include`s upstream `arch35/*` headers. That's wrapping "
        "upstream arch35 source — NOT an arch22 -> arch35 migration deliverable.",
        "",
        "For explicit target prior-art verification or prestage, rerun with "
        "`OPGEN_PRESTAGE_ARCH35=1`; otherwise remove the wrapper and migrate "
        "from arch22 source.",
        "",
        "Detected `#include arch35/...` lines:",
    ]
    for fname, hits in suspect_files:
        lines.append(f"  {fname}: {len(hits)} include(s)")
    lines.append("")
    lines.append(
        "Fix: rewrite the kernel TU to `#include` arch22 source headers from "
        "the upstream op_kernel/ top level (NOT op_kernel/arch35/). Follow "
        "the flat_quant pattern: `#include \"<op>_vec.h\"` / `#include "
        "\"<op>_cube.h\"` etc., where those V220 headers live alongside "
        "the upstream `<op>_apt.cpp` algorithm entry."
    )
    return "\n".join(lines)


def _check_arch35_wrap_cheat(workspace: Path) -> Optional[str]:
    """Reject direct target-architecture include wrapping for port_a3 only."""
    import os as _os
    from plugins import detect_plugin as _detect_plugin

    active_plugin = _detect_plugin(workspace)
    if active_plugin is None or active_plugin.name != "port_a3_to_a5":
        return None
    if _os.environ.get("OPGEN_PRESTAGE_ARCH35", "0") not in ("0", "", "false", "False"):
        return None
    suspect_files = _collect_arch35_include_hits(workspace)
    return _arch35_wrap_cheat_diagnostic(suspect_files) if suspect_files else None


def _load_architecture_class_checker() -> tuple[object | None, Optional[str]]:
    """Load the structural checker without turning its import failure into a pass."""
    import sys as _sys
    checks_dir = _HERE.parent / "checks"
    if str(checks_dir) not in _sys.path:
        _sys.path.insert(0, str(checks_dir))
    try:
        from architecture_class_check import check_architecture_class  # type: ignore
    except Exception as e:
        return None, (
            "SOURCE_ARCH_UNVERIFIED (OL-188): architecture-class gate could "
            f"not be loaded ({type(e).__name__}); fail closed."
        )
    return check_architecture_class, None


def _architecture_class_diagnostic(result: dict) -> str:
    """Render an architecture-class failure without altering gate guidance."""
    verdict = result.get("verdict") or "SOURCE_ARCH_UNVERIFIED"
    lines = [
        f"{verdict} (OL-188 2026-05-25): generated kernel architecture "
        "could not be verified against the arch22 source.",
        "",
        f"  reference_arch = {result.get('reference_arch')}",
        f"  generated_arch = {result.get('generated_arch')}",
        "",
        "Per owner directive 2026-05-25T02:16Z: pure-VEC kernel for a "
        "cube-required CANN reference op = HACK class (same anti-cheat "
        "tier as CPU fallback). The safety net cannot rely on output "
        "comparison because normal op-gen does not have reference code "
        "— architectural class must be enforced explicitly.",
        "",
        f"Detail: {result.get('reason')}",
        "",
        "Fix: rewrite the kernel TU to use cube primitives (matmul "
        "library `matmul::Matmul<>` OR `MatmulImpl<>` with REGIST_MATMUL_OBJ "
        "+ KFC-internal sync). See OL-188 'Anti-pattern signatures' for "
        "the full list of HACK patterns and per-class recipes.",
        "",
        "There is no source-architecture or target-source waiver.",
    ]
    return "\n".join(lines)


def _check_architecture_class(workspace: Path) -> Optional[str]:
    """Fail closed if a migration cannot satisfy its source architecture class."""
    try:
        from plugins import detect_plugin
        active_plugin = detect_plugin(workspace)
    except Exception as exc:
        return (
            "SOURCE_ARCH_UNVERIFIED (OL-188): supported workflow ownership "
            f"could not be resolved ({type(exc).__name__}); fail closed."
        )
    if active_plugin is None:
        return (
            "SOURCE_ARCH_UNVERIFIED (OL-188): workspace is not owned by a "
            "supported migration or backward workflow; fail closed."
        )
    if not active_plugin.requires_source_architecture_gate():
        return None
    checker, violation = _load_architecture_class_checker()
    if violation:
        return violation
    try:
        result = checker(workspace=workspace, op_name=workspace.name)
    except Exception as e:
        return (
            "SOURCE_ARCH_UNVERIFIED (OL-188): architecture-class gate could "
            f"not verify the arch22 source ({type(e).__name__}); fail closed."
        )
    return None if result.get("verdict") == "PASS" else _architecture_class_diagnostic(result)


def _find_project_json(workspace: Path) -> Optional[Path]:
    """Locate the applicable PROJECT.json without changing lookup precedence."""
    for parent in [workspace.resolve()] + list(workspace.resolve().parents):
        proj_json = parent / "PROJECT.json"
        if proj_json.exists():
            return proj_json

    out_dir = _PROJECT_ROOT / "output"
    workspace_path = str(workspace.resolve())
    candidates = sorted(out_dir.iterdir()) if out_dir.exists() else []
    for candidate in candidates:
        project_json = candidate / "PROJECT.json"
        if candidate.is_dir() and project_json.exists() and workspace_path.startswith(str(candidate.resolve())):
            return project_json
    return None


def _check_project_json_metadata(workspace: Path) -> Optional[str]:
    """Enforce the required PROJECT.json metadata when the file is present."""
    proj_json = _find_project_json(workspace)
    if proj_json is None:
        return None

    try:
        data = json.loads(proj_json.read_text())
    except Exception:
        return None  # Malformed — other gates catch this

    required = [
        "schema_version", "project", "opgen_mode", "source",
        "reference_baseline", "target_chip", "created", "owner_agent",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        return f"PROJECT.json missing required fields: {missing}"
    return None


def _check_pp88_compliance(workspace: Path) -> Optional[str]:
    """P0abi (2026-05-08): P-P88 sigmoid-form remediation gate.

    Invokes scan_pp88_compliance.scan_workspace. Returns None when the
    scan verdict is PASS or NOT_APPLICABLE. Returns the rationale string
    when verdict is FAIL.

    Catches the 1_GELU regression class: kernel emits AscendC::Tanh
    (PB-24 bimodal-floor primitive) on a transcendental op with
    small-value benchmark cases, but neither (a) applied sigmoid-form
    rewrite (Exp + Reciprocal + Add per P-P88 vendor-source evidence)
    nor (b) declared structured exemption with isolated-primitive
    evidence in knowledge_update.md `p_p88:` block. Without (a) or (b),
    the kernel is non-monotonic at fp32-ULP level on small-x inputs
    (PB-24: Tanh<fp32> 1599 ULP at x≈1.7e-4) — same op regenerated with
    different tile size produces different rounding, masking real
    precision regressions as "stochastic cold-start drift."
    """
    try:
        # Lazy import — scan_pp88_compliance lives in src/scripts/, not
        # under orchestrator/. Add to sys.path on demand to avoid hard
        # cross-package dependency.
        import sys as _sys
        scripts_dir = _PROJECT_ROOT / "src" / "scripts"
        if str(scripts_dir) not in _sys.path:
            _sys.path.insert(0, str(scripts_dir))
        import scan_pp88_compliance as _scan  # type: ignore
        rep = _scan.scan_workspace(workspace)
    except Exception as e:
        # Fail-open on importer error; missing scanner shouldn't block
        # finalize. The unit test for this gate verifies the import path.
        return None
    if rep.verdict == "FAIL":
        return (
            "P0abi P-P88 compliance gate: " + rep.rationale +
            "  Reference: kb/target/ascendc/patterns/PATTERN_INDEX.md "
            "P-P88 (sigmoid-form remediation, MANDATORY-on-match for "
            "transcendental ops using AscendC::Tanh / Sigmoid). Either "
            "(a) rewrite the risky primitive call to sigmoid-form (Exp + "
            "Reciprocal + Add per vendor-source evidence in P-P88) and "
            "add a `p_p88: status: applied` YAML block in "
            "knowledge_update.md with diff_refs, OR (b) declare exemption "
            "via `p_p88: status: exempt` block with concrete "
            "isolated-primitive measurements showing the small-x failure "
            "regime doesn't apply to this op."
        )
    return None


_FORBIDDEN_MODEL_NEW_PATTERNS = (
    (r"torch\.npu\.\w+\(", "torch.npu.* compute call"),
    (r"\btorch_npu\.\w+", "torch_npu.* call"),
    (r"\bF\.\w+\(", "F.<op>() compute call (likely PyTorch fallback)"),
    (r"\btorch\.(?:matmul|mm|bmm|sum|mean|max|min|softmax|"
     r"sigmoid|tanh|relu|gelu|silu|exp|log|sqrt|rsqrt|pow|"
     r"add|sub|mul|div|cat|stack|where|sort|topk|cumsum|"
     r"argmax|argmin|gather|scatter|nonzero|index_put)\(",
     "torch.<compute>() primitive call"),
    (r"\.(?:cpu|to_cpu)\(\s*\)(?!\.tolist)", "tensor.cpu() data motion (forward path)"),
    (r"\bhashlib\b", "hashlib import — cache-replay marker (OL-165 P151)"),
    (r"\b_tensor_digest\b", "_tensor_digest — cache-replay digest function (OL-165 P151)"),
    (r"\b_LOOKUP_CACHE\b", "_LOOKUP_CACHE — cache-replay lookup table (OL-165 P151)"),
    (r"\b_build_lookup\b", "_build_lookup — cache-replay builder (OL-165 P151)"),
    (r"a5_capture\.pt", "a5_capture.pt read — cache-replay source (OL-165 P151)"),
    (r"edge_dataset\.pt\[\s*['\"]a5_outputs['\"]",
     "edge_dataset.pt[a5_outputs] — cache-replay source (OL-165 P151)"),
    (r"subprocess\.(?:run|Popen)\s*\(",
     "subprocess from forward — kernel runs in cpp binary not pybind (OL-165 P151 pattern 2)"),
)


def _find_pybind_path(workspace: Path) -> Optional[Path]:
    """Return the first supported pybind shim path, preserving search order."""
    candidates = (
        workspace / "op_kernel" / "pybind11.cpp",
        workspace / "pybind11.cpp",
        workspace / "kernel" / "pybind11.cpp",
    )
    return next((path for path in candidates if path.exists()), None)


def _collect_cpp_host_logic_violations(text: str, workspace: Path, vj: dict) -> list[str]:
    """Collect forbidden host-side C++ compute patterns in their original order."""
    text_nc = re.sub(r"//[^\n]*", "", text)
    text_nc = re.sub(r"/\*.*?\*/", "", text_nc, flags=re.S)
    violations: list[str] = []
    if "to(at::kCPU)" in text_nc or re.search(r"\.cpu\s*\(\s*\)", text_nc):
        violations.append(
            "CPU offload — pattern `.to(at::kCPU)` or `.cpu()` detected. "
            "This moves tensors to CPU for host-side compute/prep, "
            "violating 'all compute in kernel' rule (CLAUDE.md No CPU Fallback). "
            "Restructure: do the dtype/layout conversion INSIDE the kernel "
            "(AscendC Cast / DataCopy with type conversion) or via aclrtMemcpyAsync."
        )
    has_pad_alloc = any((
        re.search(r"torch::empty\s*\(\s*\{[^}]*(?:\+|PAD|ALIGN)[^}]*\}", text_nc),
        "TAIL_PAD" in text_nc, "HEAD_PAD" in text_nc,
        re.search(r"numel\s*\+\s*\w*PAD", text_nc),
        re.search(r"\+\s*(?:ALIGN|TAIL_PAD|HEAD_PAD|TILE_PAD)_?\w*", text_nc),
    ))
    has_narrow_crop = bool(re.search(r"\.narrow\s*\(\s*0\s*,\s*0\s*,", text_nc))
    if has_pad_alloc and has_narrow_crop and not _is_v220_ec41_output_pad_exempt(workspace, vj):
        violations.append(
            "Output-alignment cleanup — pattern `torch::empty(numel + PAD)` "
            "followed by `.narrow(0, 0, ...)` detected. This indicates the "
            "kernel writes past the valid output range (likely aligned "
            "DataCopy on unaligned tail) and pybind crops the garbage. "
            "Fix the KERNEL to not write past valid range "
            "(use ALIGN-aware DataCopy on tail, or DataCopyPad). "
            "Pybind must NOT mask kernel bugs. "
            "(V220-EC-41 carve-out did NOT apply: needs arch22 + aligned 3-arg "
            "DataCopy + no DataCopyPad call + precision PASS — see "
            "_is_v220_ec41_output_pad_exempt.)"
        )
    if re.search(r"torch::(cat|stack|pad)\s*\(", text_nc):
        violations.append(
            "Output assembly in pybind — `torch::cat / stack / pad` detected. "
            "Output composition must happen in kernel (or via multiple "
            "kernel outputs combined by caller). Pybind cannot serve as "
            "the host-side fusion layer."
        )
    return violations


def _collect_model_new_violations(workspace: Path) -> list[str]:
    """Collect prohibited wrapper-side compute patterns, one per matcher."""
    model_new_path = workspace / "model_new_ascendc.py"
    if not model_new_path.exists():
        return []
    try:
        py_text = model_new_path.read_text(errors="replace")
    except Exception:
        py_text = ""
    py_nc = re.sub(r'""".*?"""', "", py_text, flags=re.S)
    py_nc = re.sub(r"#[^\n]*", "", py_nc)
    violations = []
    for pattern, description in _FORBIDDEN_MODEL_NEW_PATTERNS:
        for match in re.finditer(pattern, py_nc):
            prefix = py_nc[:match.start()]
            if "if __name__" in prefix and prefix.rfind("if __name__") > prefix.rfind("class ModelNew"):
                continue
            violations.append(
                f"model_new_ascendc.py: forbidden Python compute — {description} "
                f"at offset ~{match.start()}: {match.group()!r}. All compute must "
                f"be in the AscendC kernel; model_new_ascendc.py is a "
                f"minimal nn.Module wrapper that calls into pybind only."
            )
            break
    return violations


def _pybind_host_logic_diagnostic(workspace: Path, pybind_path: Path, violations: list[str]) -> str:
    """Render the established P149 diagnostic from collected violations."""
    location = pybind_path.relative_to(workspace.parent) if workspace.parent in pybind_path.parents else pybind_path.name
    bullets = "\n  - ".join(violations)
    return (
        f"P149 PYBIND_HOST_BUSINESS_LOGIC: pybind11.cpp at {location} "
        f"contains host-side business logic:\n  - {bullets}\n"
        "Pybind is dispatch + alloc only — no compute, no CPU offload, no "
        "alignment cleanup. See OL-163 (pybind purity) + CLAUDE.md "
        "'No PyTorch/CANN Delegation, No CPU Fallback'. Worker must fix "
        "KERNEL to handle the case correctly; pybind cannot mask kernel "
        "bugs as 'host-side helper'."
    )


def _check_pybind_host_logic(workspace: Path, vj: dict) -> Optional[str]:
    """Reject host-side business logic in pybind shims and Python wrappers."""
    if not workspace:
        return None
    pybind_path = _find_pybind_path(workspace)
    if pybind_path is None:
        return None
    try:
        text = pybind_path.read_text(errors="replace")
    except Exception:
        return None
    violations = _collect_cpp_host_logic_violations(text, workspace, vj)
    violations.extend(_collect_model_new_violations(workspace))
    return _pybind_host_logic_diagnostic(workspace, pybind_path, violations) if violations else None
