#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""scan_delegation_cheating.py — detect kernel-delegation cheats in op artifacts.

CLAUDE.md mandates: ALL computation must use AscendC primitives. The pybind /
Python wrapper layer may use torch only for tensor metadata, memory allocation,
and contiguous() on inputs. Anything else is delegation — wrapping CANN via
torch_npu / aclnn* — which scores trivially against itself and lies to the
customer about what the kernel actually does.

This scanner runs at Phase O5 (called by /aog-self-critic skill) and statically
audits these files for the delegation patterns the existing check_worker.sh
C++ blacklist was designed to catch, plus the Python-wrapper variant that the
C++ blacklist doesn't cover:

  workspace/{op}/<mode's kernel_logic_files()>  (Python compute — plugin-declared)
  workspace/{op}/<mode's kernel_cpp_dirs()>/**  (C++ host bridge + kernel — plugin-declared)

NEITHER the Python file list NOR the C++ dir list is hard-coded (DEBT-211). Each
mode's plugin declares where its compute actually lives — Python via
`kernel_logic_files()`, C++ via `kernel_cpp_dirs()`; this scanner asks. The C++
declaration closes the DIRECTORY-level recurrence of OL-160: the scanner used to
walk a hard-coded `kernel/` dir, but `port_a3_to_a5` keeps its C++ in `op_host/`
+ `op_kernel/` (no `kernel/`), so the walk read nothing and the whole mode's
host-compute delegation shipped `violations=0`. A mode that authors AscendC C++
A mode that declares nothing, or declares only absent dirs, is a HARD FAILURE
here — same fail-loud rule as the Python side. The op's own
generated aclnn C-API layer (`op_api/`) is skipped: those are API DEFINITIONS,
not host-compute delegation.
Hard-coding one entry name is what re-opened OL-160: the canonical entry was a
thin shim while compute lived elsewhere, so the scan read no real logic and
reported `violations=0`. A mode that declares nothing, or declares files that
are all absent, is a
HARD FAILURE here: scanning nothing is an error, never a pass.

Or if --archive-mode is given, scans every op under output/<project>/src/kernels/*.

Exit codes:
  0  no violations (PASS)
  1  violations found (FAIL — caller should propagate as REJECT)
  2  usage error
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Pattern catalog
# ---------------------------------------------------------------------------

# Category tag for "the scan could not read this workspace's compute"
# (DEBT-211). Distinct from a delegation hit: it means the safety net was
# blind here, which is a failure in its own right.
SCANNER_COVERAGE_CATEGORY = "scanner_coverage_gap"

PYTHON_WRAPPER_PATTERNS: list[tuple[str, str]] = [
    (r'\btorch_npu\.[a-zA-Z_][a-zA-Z_0-9]*\s*\(', "torch_npu.<api>(...) call (CANN delegation via torch_npu)"),
    (r'\baclnn[A-Z][a-zA-Z_0-9]*\s*\(', "aclnn* call (CANN built-in)"),
    (r'\baclop[A-Z][a-zA-Z_0-9]*\s*\(', "aclop* call"),
    (r'\bacl_op_[a-zA-Z_0-9]+\s*\(', "acl_op_* call"),
    (r'\bF\.(softmax|relu|silu|gelu|layer_norm|rms_norm|attention|cross_entropy|nll_loss|log_softmax)\s*\(',
        "F.<compute>() call (delegating to torch.nn.functional)"),
    (
        r'\btorch\.(softmax|matmul|bmm|sort|topk|cumsum|histc|nonzero|gather|'
        r'scatter|scatter_add|scatter_reduce|argsort|nll_loss|cross_entropy|'
        r'layer_norm|rms_norm)\s*\(',
        "torch.<compute>() call (delegating to torch top-level)"),
    (r'\.(softmax|matmul|bmm|sort|topk|cumsum|histc|nonzero|scatter_add|scatter_reduce)\s*\(',
        "tensor.<compute>() method (delegating computation)"),
    (r'subprocess\.(run|call|check_output|Popen)\s*\([^)]*aclnn',
        "subprocess invoking aclnn binary"),
]

CPP_PATTERNS: list[tuple[str, str]] = [
    # Tensor compute methods
    (r'\.sum\s*\(', "tensor.sum() compute in C++"),
    (r'\.mean\s*\(', "tensor.mean()"),
    (r'\.var\s*\(', "tensor.var()"),
    (r'\.norm\s*\(', "tensor.norm()"),
    (r'\.argmax\s*\(', "tensor.argmax()"),
    (r'\.topk\s*\(', "tensor.topk()"),
    (r'\.cumsum\s*\(', "tensor.cumsum()"),
    (r'\.exp\s*\(', "tensor.exp()"),
    (r'\.log\s*\(', "tensor.log()"),
    (r'\.sqrt\s*\(', "tensor.sqrt()"),
    (r'\.relu\s*\(', "tensor.relu()"),
    (r'\.sigmoid\s*\(', "tensor.sigmoid()"),
    (r'\.tanh\s*\(', "tensor.tanh()"),
    (r'\.softmax\s*\(', "tensor.softmax()"),
    (r'\.matmul\s*\(', "tensor.matmul()"),
    (r'\.bmm\s*\(', "tensor.bmm()"),
    # Top-level torch::* compute
    (r'\btorch::pow\b', "torch::pow"),
    (r'\btorch::matmul\b', "torch::matmul"),
    (r'\btorch::sum\b', "torch::sum"),
    (r'\btorch::mean\b', "torch::mean"),
    (r'\btorch::softmax\b', "torch::softmax"),
    (r'\btorch::cat\b', "torch::cat"),
    (r'\btorch::stack\b', "torch::stack"),
    (r'\btorch::bmm\b', "torch::bmm"),
    (r'\btorch::mm\b', "torch::mm"),
    (r'\btorch::topk\b', "torch::topk"),
    (r'\btorch::sort\b', "torch::sort"),
    (r'\btorch::cumsum\b', "torch::cumsum"),
    (r'\bat::matmul\b', "at::matmul"),
    (r'\bat::sum\b', "at::sum"),
    (r'\bat::mean\b', "at::mean"),
    # CANN built-ins
    (r'\baclnn[A-Z]\w*\s*\(', "aclnn* call (CANN built-in)"),
    (r'\baclop[A-Z]\w*\s*\(', "aclop* call"),
    (r'\bacl_op_\w+\s*\(', "acl_op_* call"),
    (r'\baclrtLaunchKernel\s*\(', "aclrtLaunchKernel (launching pre-built CANN kernel)"),
]

# Lines / files we exempt entirely
EXEMPT_LINE_PREFIX = (
    re.compile(r'^\s*#'),
    re.compile(r'^\s*//'),
    re.compile(r'^\s*"""'),
    re.compile(r'^\s*\*\s'),  # block comment continuation
    re.compile(r'^\s*\*/'),
    re.compile(r'^\s*import\s'),
    re.compile(r'^\s*from\s'),
)

# Symbols whose names contain a forbidden token but aren't real cheats
ALLOWLIST_TOKENS = (
    "aclrtMemcpyAsync",      # memory copy primitive — legit DMA
    "aclrtMalloc",
    "aclrtFree",
    "aclrtSynchronizeStream",
    "aclrtCreateStream",
    "aclrtlaunch_",          # user kernel launch (lowercase l = our kernel, distinct from aclrtLaunchKernel)
)

# ---------------------------------------------------------------------------
# Scan functions
# ---------------------------------------------------------------------------


def _strip_string_literals(line: str) -> str:
    """Remove string contents to avoid false positives in error messages / docstrings."""
    line = re.sub(r'"""[^"]*"""', '""', line)
    line = re.sub(r"'''[^']*'''", "''", line)
    line = re.sub(r'"[^"\\]*(\\.[^"\\]*)*"', '""', line)
    line = re.sub(r"'[^'\\]*(\\.[^'\\]*)*'", "''", line)
    line = re.sub(r'#.*$', '', line)        # py inline comment
    line = re.sub(r'//.*$', '', line)       # cpp inline comment
    return line


def _is_exempt(line: str) -> bool:
    return any(p.match(line) for p in EXEMPT_LINE_PREFIX)


def _has_allowlist_token(line: str) -> bool:
    return any(tok in line for tok in ALLOWLIST_TOKENS)


def scan_python_wrapper(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(errors="ignore")
    hits = []
    in_docstring = False
    in_main_block = False
    main_block_indent: int | None = None
    _MAIN_GUARD_RE = re.compile(r'^\s*if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:\s*(#.*)?$')
    for lineno, raw in enumerate(text.splitlines(), 1):
        triple = raw.count('"""') + raw.count("'''")
        if triple == 1:
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        # __main__-block waiver: code inside `if __name__ == "__main__":` runs
        # only when the file is invoked directly (`python model_new_ascendc.py`)
        # for dev smoke-testing. It is NEVER executed when the verifier imports
        # ModelNew + calls forward(), so it cannot be a delegation vector on the
        # kernel surface. Common pattern: CPU oracle (`out_cpu_truth = torch.gather(
        # x_cpu, ...)`) for human-visible bit-equality print. Surfaced by
        # This waiver is scoped to the non-import execution path.
        if _MAIN_GUARD_RE.match(raw):
            in_main_block = True
            main_block_indent = len(raw) - len(raw.lstrip())
            continue
        if in_main_block:
            stripped = raw.strip()
            # Stay in __main__ block until we hit a top-level statement at the
            # same indent level as the `if __name__` guard. Blank lines and
            # more-indented code remain inside. A line at OR BELOW guard indent
            # with non-empty content exits the block.
            if stripped:
                cur_indent = len(raw) - len(raw.lstrip())
                if main_block_indent is not None and cur_indent <= main_block_indent:
                    in_main_block = False
                    main_block_indent = None
                    # Fall through to normal scanning for this line
                else:
                    continue
            else:
                continue
        if _is_exempt(raw):
            continue
        if _has_allowlist_token(raw):
            continue
        cleaned = _strip_string_literals(raw)
        if not cleaned.strip():
            continue
        for pat, desc in PYTHON_WRAPPER_PATTERNS:
            if re.search(pat, cleaned):
                hits.append({"file": str(path), "line": lineno, "desc": desc, "text": raw.strip()[:160]})
                break
    return hits


def scan_cpp(path: Path, is_pybind: bool) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(errors="ignore")
    hits = []
    in_block_comment = False
    for lineno, raw in enumerate(text.splitlines(), 1):
        if "/*" in raw and "*/" not in raw:
            in_block_comment = True
            continue
        if in_block_comment:
            if "*/" in raw:
                in_block_comment = False
            continue
        if _is_exempt(raw):
            continue
        if _has_allowlist_token(raw):
            continue
        cleaned = _strip_string_literals(raw)
        if not cleaned.strip():
            continue
        for pat, desc in CPP_PATTERNS:
            if re.search(pat, cleaned):
                hits.append({"file": str(path), "line": lineno, "desc": desc, "text": raw.strip()[:160]})
                break
    return hits


PORT_A3_VERIFY_FILES = (
    "run_a5_verify.py",        # A5-side verification entry
    "pass_a_runner.py",        # Pass A canonical verifier
    "run_pass_b.py",           # Pass B canonical verifier
)

# Patterns BANNED in port_a3-mode verify files (per DEBT-NEW 2026-05-14,
# user catch "your reward hacking make us lost 2 days"):
# A5-side verification must invoke our built kernel via aclnn-direct C++
# runner / ctypes shim — NOT through PyTorch dispatcher (which silently
# falls back to AICPU when our .so isn't installed). The exact same
# `F.*` / `torch._foreach_*` / `torch_npu.npu_*` calls are OK in
# run_a3_reference.py (A3 has working CANN install) but FATAL in
# run_a5_verify.py / pass_a_runner.py.
PORT_A3_VERIFY_FORBIDDEN: list[tuple[str, str]] = [
    (
        r'\btorch\.nn\.functional\.[a-z_]+\s*\(',
        "torch.nn.functional.<op>() in A5 verify path — falls back to AICPU "
        "when our .so is unloaded; runs stock PyTorch instead of our kernel",
    ),
    (r'\bF\.[a-z_]+\s*\(',
        "F.<op>() in A5 verify path — same fallback risk; use aclnn-direct shim instead"),
    (r'\btorch\._foreach_[a-z]+\s*\(',
        "torch._foreach_<op>() in A5 verify path — routes through PyTorch foreach dispatcher, NOT our kernel"),
    (r'\btorch_npu\.npu_[a-zA-Z_]+\s*\(',
        "torch_npu.npu_<op>() in A5 verify path — may unbind on A5 (per OL-68); use aclnn-direct ctypes shim"),
    (r'\btorch\.gather\s*\(.*device\s*=\s*[\'"]npu',
        "torch.gather() on NPU in A5 verify path — uses stock PyTorch gather, NOT our aclnnGatherElementsV2"),
]

# CPU-as-reference anti-pattern: pass_a_runner / run_a5_verify computes
# reference via CPU `tensor.<op>()` and compares against A3 NPU output —
# this NEVER touches A5 hardware. Example: foreach_abs pass_a_runner used
# CPU `tensor.abs()` as reference. Caught as a separate sub-pattern so
# the violation message is clear.
CPU_AS_REFERENCE_ANTIPATTERN: list[tuple[str, str]] = [
    (
        r'tensor\.(abs|exp|log|sqrt|rsqrt|sigmoid|tanh|gelu|silu|softmax|sum|'
        r'mean|var|max|min|argmax|argmin|topk|sort|cumsum|gather|scatter)\s*\(',
        "CPU tensor.<op>() as 'reference' in verify path — never exercises "
        "A5 NPU let alone our kernel; verification is meaningless",
    ),
]


def scan_verify_file_with_patterns(
    path: Path, patterns: list[tuple[str, str]], category: str
) -> list[dict]:
    """Generic verify-file scan against a plugin-supplied pattern list.

    Mode-agnostic implementation (DEBT-094 phase 1). The plugin supplies
    the patterns + category; the scan logic itself knows nothing about
    which mode is using it. port_a3 calls this with its own patterns;
    future plugins can call the same function
    with their own patterns when they're wired.
    """
    if not path.exists():
        return []
    text = path.read_text(errors="ignore")
    hits = []
    in_docstring = False
    for lineno, raw in enumerate(text.splitlines(), 1):
        triple = raw.count('"""') + raw.count("'''")
        if triple == 1:
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if _is_exempt(raw):
            continue
        cleaned = _strip_string_literals(raw)
        if not cleaned.strip():
            continue
        for pat, desc in patterns:
            if re.search(pat, cleaned):
                hits.append({
                    "file": str(path), "line": lineno, "desc": desc,
                    "text": raw.strip()[:160],
                    "category": category,
                })
                break
    return hits


def scan_port_a3_verify(path: Path) -> list[dict]:
    """Back-compat wrapper. New code should use scan_verify_file_with_patterns
    via plugin dispatch. Kept for any test that imported by name.
    """
    return scan_verify_file_with_patterns(
        path,
        PORT_A3_VERIFY_FORBIDDEN + CPU_AS_REFERENCE_ANTIPATTERN,
        category="port_a3_verify_forbidden",
    )


def _get_active_plugin(ws: Path):
    """Plugin-dispatch helper (DEBT-094 phase 1). Returns the unique
    plugin matching `ws`, or None when no plugin claims it. Lazy-imports
    plugins to avoid bootstrap circular-import in environments where this
    scanner runs standalone.

    DEBT-216: the trailing `except Exception: return None` is gone. A scanner
    that cannot determine the mode must not report `violations=0` — that `0`
    is indistinguishable on stdout from a real clean scan, which is precisely
    how the plugin branch of this scan sat inert on 13 archives. `None` here
    now means "no plugin claims this workspace" and nothing else.
    """
    try:
        # plugins/ lives under src/scripts/orchestrator/
        import sys as _sys
        _orch = Path(__file__).parent / "orchestrator"
        if str(_orch) not in _sys.path:
            _sys.path.insert(0, str(_orch))
        from plugins import detect_plugin
    except ImportError:
        return None
    return detect_plugin(ws)


def _is_port_a3_workspace(ws: Path) -> bool:
    """Back-compat wrapper for legacy callers.

    DEPRECATED — new code MUST call `_get_active_plugin(ws)` and use
    the plugin protocol methods. This shim survives only so existing
    tests + tooling that imported the function by name don't break.
    Anti-regression test (Layer 7) explicitly exempts this single
    shim definition; do NOT add new mode-name-based callers."""
    plugin = _get_active_plugin(ws)
    from plugins.port_a3 import PortA3Plugin
    return isinstance(plugin, PortA3Plugin)


def _coverage_violation(ws: Path, desc: str) -> dict:
    """A scan that read no kernel logic is an ERROR, not a pass (DEBT-211).

    Emitted with its own category so an audit can tell "this workspace
    delegates to CANN" apart from "we never managed to look at this
    workspace's compute". Both are failures; conflating them is how the
    2026-05-14 incident stayed invisible for 4 archives.
    """
    return {
        "file": str(ws),
        "line": 0,
        "desc": desc,
        "text": SCANNER_COVERAGE_CATEGORY,
        "category": SCANNER_COVERAGE_CATEGORY,
    }


def scan_op_workspace(ws: Path) -> dict:
    """Scan a single workspace dir. Returns dict with violations + summary.

    Mode-agnostic core. The set of Python files holding kernel logic is
    NOT hard-coded here — it is declared by the mode's plugin via
    `kernel_logic_files()` and scanned with core's universal patterns
    (DEBT-211). Hard-coding one entry filename is what re-opened OL-160:
    a shim was scanned while the real compute lived elsewhere.

    Coverage is therefore mandatory, and every way of "scanning nothing"
    is an explicit violation rather than `violations=0`:
      - the mode declares no kernel-logic file, or
      - none of the declared files exist in the workspace.

    The C++ `kernel/` scan and the mode-specific verify-path scan are
    unchanged.
    """
    violations: list[dict] = []
    scanned_logic: list[Path] = []
    _plugin = _get_active_plugin(ws)

    # ── Kernel-logic sources: plugin-declared, never hard-coded ────────
    if _plugin is not None:
        declared = tuple(_plugin.kernel_logic_files())
        if not declared:
            violations.append(_coverage_violation(
                ws,
                f"mode '{_plugin.name}' declares no kernel_logic_files(); "
                f"the delegation scan cannot be pointed at its compute "
                f"(DEBT-211 — declare them on the plugin)",
            ))
        for name in declared:
            f = ws / name
            if f.is_file():
                scanned_logic.append(f)
                violations.extend(scan_python_wrapper(f))
        if declared and not scanned_logic:
            violations.append(_coverage_violation(
                ws,
                f"mode '{_plugin.name}' declares kernel_logic_files("
                f"{', '.join(declared)}) but none exist here; nothing was "
                f"scanned for delegation (DEBT-211 — a vacuous scan is an "
                f"error, not a pass)",
            ))
    else:
        # No plugin matched (legacy / pre-plugin archives, or a workspace
        # whose mode is genuinely unidentifiable). We cannot ask anyone
        # where the logic lives, so fall back to the historical canonical
        # entry-point. This preserves coverage for the ~39 pre-plugin
        # archives rather than mass-flagging them, but it is a FALLBACK:
        # it is only acceptable while it actually reads something.
        legacy = ws / "model_new_ascendc.py"
        if legacy.is_file():
            scanned_logic.append(legacy)
            violations.extend(scan_python_wrapper(legacy))

    # ── AscendC C++ sources: plugin-declared dirs, never a hard-coded `kernel/`
    # DEBT-211 (directory-level OL-160 recurrence): the scanner used to walk a
    # hard-coded `kernel/` dir. port_a3_to_a5 puts its C++ in `op_host/` +
    # `op_kernel/` and has NO `kernel/`, so the walk traversed nothing → the
    # whole mode's host-compute delegation shipped `violations=0` and the
    # finalize POST_WORKER_AUDIT gate was inert. Each plugin now DECLARES its
    # C++ dirs via `kernel_cpp_dirs()`; core scans those. `kernel/` is kept as
    # a backward-compat scan target so kernel/-based modes + the no-plugin
    # legacy path are unchanged.
    declared_cpp: tuple[str, ...] = ()
    if _plugin is not None:
        declared_cpp = tuple(_plugin.kernel_cpp_dirs())
    # Dedup, preserve order; always include the historical `kernel/`.
    scan_dir_names = list(dict.fromkeys([*declared_cpp, "kernel"]))

    cpp_scanned = 0
    for dname in scan_dir_names:
        d = ws / dname
        if not d.is_dir():
            continue
        for f in d.rglob("*"):
            if not f.is_file():
                continue
            # op_api/ suppression: the op's OWN generated aclnn C-API layer
            # (`aclnnXxxGetWorkspaceSize` / `aclnnXxx` DEFINITIONS) is API
            # plumbing, not host-compute delegation. Skip any file that sits
            # under an `op_api/` directory so those definitions don't mass-flag
            # as "aclnn* call". Genuine host-compute (cumsum/exp/matmul on
            # at::Tensor) in op_host/op_kernel is NOT under op_api and is still
            # caught.
            if "op_api" in f.relative_to(ws).parts:
                continue
            if f.suffix in {".cpp", ".cc", ".cxx"}:
                cpp_scanned += 1
                violations.extend(scan_cpp(f, is_pybind=("pybind11" in f.name)))
            elif f.suffix in {".h", ".hpp"}:
                cpp_scanned += 1
                violations.extend(scan_cpp(f, is_pybind=False))

    # ── C++ coverage-gap (fail-loud, DEBT-211 directory-level) ─────────────
    # For any mode that authors AscendC C++, an undeclared surface OR a
    # declaration whose dirs are all absent is a COVERAGE-GAP VIOLATION — the
    # scan must NOT return a silent 0-violation pass over a mode whose real
    # compute dir it never looked at.
    if _plugin is not None:
        if not declared_cpp:
            violations.append(_coverage_violation(
                ws,
                f"mode '{_plugin.name}' authors AscendC C++ but declares no "
                f"kernel_cpp_dirs(); the delegation scan cannot be pointed at "
                f"its host/kernel C++ (DEBT-211 directory-level — declare the "
                f"dirs on the plugin, do not leave it undeclared)",
            ))
        elif not any((ws / d).is_dir() for d in declared_cpp):
            violations.append(_coverage_violation(
                ws,
                f"mode '{_plugin.name}' declares kernel_cpp_dirs("
                f"{', '.join(declared_cpp)}) but none exist here; the mode's "
                f"C++ compute was never scanned (DEBT-211 directory-level — a "
                f"vacuous scan is an error, not a pass)",
            ))

    # Nothing at all was read — the exact shape of the silent zero.
    if _plugin is None and not scanned_logic and not cpp_scanned:
        violations.append(_coverage_violation(
            ws,
            "no plugin matched and no canonical entry-point or kernel/ "
            "source was found; this workspace's compute was never scanned "
            "(DEBT-211 — a vacuous scan is an error, not a pass)",
        ))

    # Plugin-dispatched mode-specific verify-path scan
    if _plugin is not None:
        plugin_verify_files = _plugin.verify_files()
        plugin_patterns = _plugin.forbidden_patterns()
        if plugin_verify_files and plugin_patterns:
            for verify_name in plugin_verify_files:
                violations.extend(scan_verify_file_with_patterns(
                    ws / verify_name, plugin_patterns,
                    category=_plugin.scanner_category(),
                ))

    return {
        "workspace": str(ws),
        "ok": len(violations) == 0,
        "violation_count": len(violations),
        "violations": violations,
        "scanned_logic_files": [str(p) for p in scanned_logic],
        "mode": _plugin.name if _plugin is not None else None,
        "is_port_a3": _is_port_a3_workspace(ws),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--workspace", type=Path, help="single workspace dir to audit")
    g.add_argument("--archive-mode", action="store_true", help="audit every archived op under output/")
    # DEBT-191: repo-relative default (not a hardcoded a5_ops abspath; also fixes the
    # stale pre-slim path). scan_delegation_cheating.py = src/scripts → parents[2] = repo root.
    parser.add_argument("--archive-root", type=Path, default=Path(__file__).resolve().parents[2] / "output")
    parser.add_argument("--json", action="store_true", help="emit JSON report instead of human-readable")
    parser.add_argument("--write-report", type=Path, help="also write report to this path (markdown)")
    args = parser.parse_args()

    reports: list[dict] = []
    if args.workspace:
        reports.append(scan_op_workspace(args.workspace))
    else:
        # Walk the archive: any path matching .../kernels/<op>/ is an op
        for d in sorted(args.archive_root.rglob("kernels/*")):
            if d.is_dir():
                reports.append(scan_op_workspace(d))

    total_violations = sum(r["violation_count"] for r in reports)
    bad_ops = [r for r in reports if not r["ok"]]

    if args.json:
        print(json.dumps({"reports": reports, "summary": {
            "ops_scanned": len(reports),
            "ops_with_violations": len(bad_ops),
            "total_violations": total_violations,
        }}, indent=2))
    else:
        print(f"Scanned {len(reports)} op workspaces")
        print(f"Ops with violations: {len(bad_ops)}")
        print(f"Total violations: {total_violations}\n")
        for r in bad_ops:
            print(f"❌ {r['workspace']}")
            for v in r["violations"]:
                print(f"     {v['file'].split('/')[-1]}:{v['line']}  [{v['desc']}]")
                print(f"        {v['text']}")
            print()

    if args.write_report:
        md = ["# Delegation cheating scan report\n"]
        md.append(f"- Ops scanned: {len(reports)}")
        md.append(f"- Ops with violations: {len(bad_ops)}")
        md.append(f"- Total violations: {total_violations}\n")
        if bad_ops:
            md.append("## Violations\n")
            for r in bad_ops:
                md.append(f"### {r['workspace']}\n")
                for v in r["violations"]:
                    md.append(f"- `{v['file']}:{v['line']}` — **{v['desc']}**")
                    md.append(f"  - `{v['text']}`")
                md.append("")
        else:
            md.append("**ALL CLEAN** — no delegation cheats detected.\n")
        args.write_report.write_text("\n".join(md))

    return 1 if total_violations else 0


if __name__ == "__main__":
    sys.exit(main())
