#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scan an Ascend C operator project for A5 (DAV_3510 / arch35) only constructs.

The scan is static and deliberately conservative: it reports what it finds and
maps each hit to an A2 (DAV_2201 / arch22) action. It never rewrites code and
never claims a finding is fatal without the operator context; the caller decides
with the inventory reference (references/a5-only-inventory.md).

Usage:
    python3 scan_a5_only.py --root /path/to/operator
    python3 scan_a5_only.py --root . --format json --out scan.json
    python3 scan_a5_only.py --root . --fail-on blocking

Exit codes:
    0  scan completed (regardless of findings, unless --fail-on matched)
    1  --fail-on threshold matched
    2  usage / IO error
"""

from __future__ import annotations

import argparse
import bisect
import datetime
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Tuple

SCHEMA = "ascend-op-arch-migration/scan/v1"
TOOL_VERSION = "1.0.0"

SOURCE_EXTS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".asc",
    ".inc",
    ".inl",
    ".py",
    ".json",
    ".ini",
    ".cmake",
    ".sh",
    ".txt",
    ".mk",
}
SPECIAL_NAMES = {"CMakeLists.txt", "Makefile", "makefile", "build.sh"}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "build",
    "build_out",
    "output",
    "out",
    "dist",
    "third_party",
    "thirdparty",
    "node_modules",
    "__pycache__",
    ".cache",
    ".idea",
    ".vscode",
    "cmake-build-debug",
    "cmake-build-release",
    "log",
    "logs",
    "prof",
    "profiling",
    ".venv",
    "venv",
    "site-packages",
}

MAX_FILE_BYTES = 4 * 1024 * 1024

SEVERITY_ORDER = {"blocking": 0, "decision": 1, "verify": 2, "advisory": 3}
LEVEL_RANK = {"D1": 1, "D2": 2, "D3": 3}

RESTORE_CHECKLIST = [
    (
        "int4 Mmad / 4:2 sparse: A2 supports both, A5 does not; evaluate restoring if the A5 code "
        "cast to int8 or removed sparsity."
    ),
    "L1->GM and GM->L0A/L0B paths: deleted on A5, available on A2; may remove a GM hop.",
    "SetLoadDataBoundary: available on A2, removed on A5; restore if the A5 code open-coded boundary handling.",
    "L0A fractal ZZ + FixpipeParamsV220: A2 native; do not keep NZ/Arch3510 parameter shapes.",
    "vnchwconv / TransDataTo5HD: high throughput on A2; consider reverting hand-written transpose workarounds.",
    "VSLDB / reduction tiling: A2 VSLDB throughput ~2x A5; consider reverting R=16/24 special-casing.",
    "BilinearInterpolation: native fused implementation on A2; consider reverting manual gather+mul+add loops.",
    "L2 size: A2 L2 is typically larger than 950PR; consider merging over-split tiles (verify with profiling).",
]


@dataclass(frozen=True)
class Rule:
    rule_id: str
    category: str
    severity: str  # blocking | decision | verify | advisory
    level: str  # D1 | D2 | D3
    pattern: Pattern[str]
    message: str
    a2_action: str


def _r(
    rule_id: str,
    category: str,
    severity: str,
    level: str,
    pattern: str,
    message: str,
    a2_action: str,
    flags: int = 0,
) -> Rule:
    return Rule(
        rule_id,
        category,
        severity,
        level,
        re.compile(pattern, flags),
        message,
        a2_action,
    )


RULES: List[Rule] = [
    # --- programming.regbase (D3) -------------------------------------------------
    _r(
        "RB-01",
        "programming.regbase",
        "blocking",
        "D3",
        r"__VEC_SCOPE__",
        "RegBase vector scope is not available on arch22.",
        "Rewrite the section as MemBase SIMD (LocalTensor + count-based APIs). "
        "See references/api-migration-recipes.md section 2.",
    ),
    _r(
        "RB-02",
        "programming.regbase",
        "blocking",
        "D3",
        r"\bMicroAPI\s*::",
        "MicroAPI (Register-based) calls do not exist on arch22.",
        "Map to the same-named MemBase API with count semantics; do not only rename the type.",
    ),
    _r(
        "RB-03",
        "programming.regbase",
        "blocking",
        "D3",
        r"\bRegTensor\s*<",
        "RegTensor is a RegBase-only container.",
        "Replace with LocalTensor backed by TQue/TBuf and add a UB budget entry.",
    ),
    _r(
        "RB-04",
        "programming.regbase",
        "blocking",
        "D3",
        r"\bMaskReg\b",
        "MaskReg is a RegBase-only mask register.",
        "Replace with an explicit element count plus tail handling (DataCopyPad / zero-fill).",
    ),
    _r(
        "RB-05",
        "programming.regbase",
        "blocking",
        "D3",
        r"\b(?:CreateMask|UpdateMask)\s*<",
        "Register mask construction is RegBase-only.",
        "Express the active element count directly; handle the tail block explicitly.",
    ),
    _r(
        "RB-06",
        "programming.regbase",
        "blocking",
        "D3",
        r"\bLocalMemBar\s*<",
        "LocalMemBar is RegBase synchronization.",
        "Use SetFlag/WaitFlag event pairs or PipeBarrier on arch22.",
    ),
    _r(
        "RB-07",
        "programming.regbase",
        "blocking",
        "D3",
        r"\b(?:LoadDist|StoreDist)\s*::",
        "RegBase load/store distribution modes are not part of MemBase DataCopy.",
        "Use DataCopy/DataCopyPad plus an explicit Cast when packing is needed.",
    ),
    _r(
        "RB-08",
        "programming.regbase",
        "blocking",
        "D3",
        r"\bCAST_[A-Z0-9]+_TO_[A-Z0-9]+\b",
        "CastTrait constants are RegBase-only; MemBase Cast takes a RoundMode.",
        "Map each CastTrait to the equivalent RoundMode and verify saturation semantics.",
    ),
    _r(
        "RB-09",
        "programming.regbase",
        "blocking",
        "D3",
        r"\b(?:asc_vf_call|__simd_vf__)\b",
        "VF call helpers are RegBase-only.",
        "Inline the VF body as a plain device function operating on LocalTensor.",
    ),
    # --- programming.simt (D3) ----------------------------------------------------
    _r(
        "SIMT-01",
        "programming.simt",
        "blocking",
        "D3",
        r"__simt_vf__",
        "SIMT vector function marker; arch22 has no SIMT execution model.",
        "Rewrite as SIMD, move to AICPU, or declare unsupported. "
        "See the SIMT decision tree in references/a5-only-inventory.md section 3.",
    ),
    _r(
        "SIMT-02",
        "programming.simt",
        "blocking",
        "D3",
        r"\bSimt\s*::\s*VF_CALL\b",
        "SIMT kernel launch is not available on arch22.",
        "Same as SIMT-01: choose SIMD rewrite, AICPU, or unsupported.",
    ),
    _r(
        "SIMT-03",
        "programming.simt",
        "blocking",
        "D3",
        r"\bAscendC\s*::\s*Simt\s*::",
        "AscendC::Simt namespace is A5-only.",
        "Remove and re-express thread/index logic with block id plus a tile loop.",
    ),
    _r(
        "SIMT-04",
        "programming.simt",
        "blocking",
        "D3",
        r"\basc_(?:syncthreads|threadfence\w*)\b",
        "SIMT synchronization primitives do not exist on arch22.",
        "Replace with the SIMD pipeline's event synchronization; verify ordering across the rewrite.",
    ),
    _r(
        "SIMT-05",
        "programming.simt",
        "blocking",
        "D3",
        r"\bLAUNCH_BOUND\s*\(",
        "SIMT launch-bound annotation is A5-only.",
        "Delete with the SIMT function it annotates.",
    ),
    _r(
        "SIMT-06",
        "programming.simt",
        "blocking",
        "D3",
        r"\b(?:GetThreadIdx|GetThreadNum)\b",
        "SIMT thread indexing is not available on arch22.",
        "Map thread index to block id plus an in-tile offset loop.",
    ),
    _r(
        "SIMT-07",
        "programming.simt",
        "blocking",
        "D3",
        r"\b__local_mem__\b",
        "SIMT local-memory pointer qualifier is A5-only.",
        "Use a UB LocalTensor and express the access pattern with DataCopy/Gather.",
    ),
    # --- dtype.a5 (D2, needs a decision) ------------------------------------------
    _r(
        "DT-01",
        "dtype.a5",
        "decision",
        "D2",
        r"(?i)\bhifloat8\b|\bHiFloat8\b|\bDT_HIFLOAT8\b",
        "HiFloat8 is an A5 data format; A2 has no HiF8 hardware type.",
        "Decide: dequantize to bf16/fp16, re-quantize to int8, or drop the dtype. "
        "Record accuracy impact and user sign-off.",
    ),
    _r(
        "DT-02",
        "dtype.a5",
        "decision",
        "D2",
        r"(?i)\bfp8_e[45]m[23][a-z_]*\b|\bfloat8_e[45]m[23][a-z_]*\b"
        r"|\bFLOAT8_E[45]M[23]\b|\bDT_FLOAT8_[A-Z0-9_]+\b",
        "FP8 (E4M3/E5M2) is an A5 data format; A2 has no FP8 hardware type.",
        "Decide: dequantize to bf16/fp16, quantize to int8, or drop the dtype. Never reinterpret the buffer.",
    ),
    _r(
        "DT-03",
        "dtype.a5",
        "decision",
        "D2",
        r"(?i)\bmxfp[48]\b|\bmxfloat8\b|\bDT_MXFP[A-Z0-9_]*\b",
        "MXFP block-scaled formats are A5-only.",
        "Re-design the quantization granularity (per-channel/per-token) only after user "
        "confirmation of the semantic change.",
    ),
    # --- datapath.a5 (D2) ---------------------------------------------------------
    _r(
        "DP-01",
        "datapath.a5",
        "blocking",
        "D2",
        r"\bisToUB\b|\bFixpipeConfig\b",
        "L0C->UB direct write-back does not exist on arch22.",
        "Route through GM workspace: Fixpipe to workspace, then DataCopy to UB; update GetWorkspaceSizes.",
    ),
    _r(
        "DP-02",
        "datapath.a5",
        "blocking",
        "D2",
        r"\b(?:dualDstCtrl|subBlockId)\b",
        "Dual-destination control is an A5 Fixpipe feature.",
        "Remove; use one destination per Fixpipe call on arch22.",
    ),
    _r(
        "DP-03",
        "datapath.a5",
        "blocking",
        "D2",
        r"\bSS(?:Buffer|buf)\b",
        "SSBuffer cross-core parameter channel is A5-only.",
        "Pass parameters through a GM workspace region with explicit event synchronization.",
    ),
    _r(
        "DP-04",
        "datapath.a5",
        "blocking",
        "D2",
        r"(?i)\benableMixDualMaster\b",
        "Dual-master MIX mode is A5-only.",
        "Remove the option and re-check any custom synchronization that assumed dual-master progress.",
    ),
    _r(
        "DP-05",
        "datapath.a5",
        "blocking",
        "D2",
        r"\bNDDMA\b",
        "NDDMA high-dimensional DMA is A5-only.",
        "Replace with per-dimension DataCopy/DataCopyPad loops, or pre-linearize the layout in GM.",
    ),
    _r(
        "DP-06",
        "datapath.a5",
        "blocking",
        "D2",
        r"\bCCU\b",
        "CCU compute-communication fusion is A5-only.",
        "Restructure to workspace exchange with explicit synchronization, or report the "
        "operator as out of scope for A2.",
    ),
    _r(
        "DP-07",
        "datapath.a5",
        "verify",
        "D2",
        r"\bLoadData2DParamsV2\b",
        "LoadData2DParamsV2 uses the A5 mStep/kStep field system.",
        "Check whether the target CANN exposes V2 on arch22; if not, rewrite as "
        "LoadData2DParams (repeatTimes/srcStride).",
    ),
    # --- sync.a5 (D2) -------------------------------------------------------------
    _r(
        "SY-01",
        "sync.a5",
        "blocking",
        "D2",
        r"\bCrossCore(?:Set|Wait)Flag\s*<\s*4\b",
        "Cross-core sync mode 4 is A5-only (950PR/DT).",
        "Switch to mode 2 (or SyncAll) and re-check the participating core sets for deadlock.",
    ),
    _r(
        "SY-02",
        "sync.a5",
        "blocking",
        "D2",
        r"\b(?:Get|Set)CtrlSpr\s*<",
        "SPR overflow-mode control is A5-only.",
        "Delete the call and its macro; solve the underlying overflow by algorithm or dtype instead.",
    ),
    _r(
        "SY-03",
        "sync.a5",
        "blocking",
        "D2",
        r"\bGLOBAL_OVERFLOW_MODE_CTRL\b",
        "A5 overflow-mode SPR id.",
        "Delete with the GetCtrlSpr/SetCtrlSpr pair.",
    ),
    # --- precision.a5 (D2) --------------------------------------------------------
    _r(
        "PR-01",
        "precision.a5",
        "blocking",
        "D2",
        r"\bPRECISION_[01]ULP_FTZ_(?:FALSE|TRUE)\b",
        "Subnormal software emulation is an A5 workaround; A2 supports subnormal natively.",
        "Drop the algorithm/config template argument and restore the default API; add subnormal precision cases.",
    ),
    _r(
        "PR-02",
        "precision.a5",
        "blocking",
        "D3",
        r"\b[A-Z][A-Za-z]*SpecificMode\b",
        "RegBase SpecificMode structures are A5-only.",
        "Remove with the RegBase rewrite.",
    ),
    _r(
        "PR-03",
        "precision.a5",
        "verify",
        "D2",
        r"\b(?:Exp|Ln|Sqrt|Rsqrt|Div|Reciprocal|EXP|LN|SQRT|RSQRT|DIV|RCP)Config\b",
        "XxxConfig template parameters belong to the A5 API family.",
        "Confirm against target CANN headers whether the config overload exists on arch22; if not, call the plain API.",
    ),
    # --- wiring.a5 (D1) -----------------------------------------------------------
    _r(
        "WI-01",
        "wiring.a5",
        "blocking",
        "D1",
        r"\barch35\b",
        "arch35 is the A5 kernel/tiling directory name.",
        "Move the implementation to arch22 and update includes, SUPPORT_TILING_DIR and CMake.",
    ),
    _r(
        "WI-02",
        "wiring.a5",
        "blocking",
        "D1",
        r"(?i)\bascend950\b",
        "ascend950 is the A5 soc/config string.",
        "Use ascend910b (or ascend910_93) consistently across --soc, AddConfig and config/<soc>.",
    ),
    _r(
        "WI-03",
        "wiring.a5",
        "blocking",
        "D1",
        r"__NPU_ARCH__\s*==\s*3510",
        "A5-only compile-time branch.",
        "Delete the branch, or rewrite both sides so they compile on arch22.",
    ),
    _r(
        "WI-04",
        "wiring.a5",
        "blocking",
        "D1",
        r"__CCE_AICORE__\s*==\s*310",
        "A5 compiler macro branch.",
        "Use __CCE_AICORE__ == 220 for arch22, or remove the branch.",
    ),
    _r(
        "WI-05",
        "wiring.a5",
        "blocking",
        "D1",
        r"\bdav-c310\b",
        "dav-c310 is the A5 compile token.",
        "Use dav-c220 (or let build.sh derive it from --soc).",
    ),
    _r(
        "WI-06",
        "wiring.a5",
        "blocking",
        "D1",
        r"\bIsRegbaseSocVersion\b",
        "RegBase platform detection has no meaning on arch22.",
        "Remove the branch and the UB 40KB reservation it guards.",
    ),
    _r(
        "WI-07",
        "wiring.a5",
        "blocking",
        "D1",
        r"\bSIMT_UB_SIZE_BYTE\b",
        "A5 reserves 40KB of UB for SIMT DCache.",
        "Delete the reservation; recompute the tiling against GetCoreMemSize(UB) on A2 (typically 192KB).",
    ),
    # --- perf.a5-workaround (advisory) -------------------------------------------
    _r(
        "PF-01",
        "perf.a5-workaround",
        "advisory",
        "D1",
        r"\bVSLDB\b",
        "VSLDB throughput is roughly half on A5; code may work around it.",
        "On A2, compare against the standard reduction tiling and consider reverting A5-specific special-casing.",
    ),
    _r(
        "PF-02",
        "perf.a5-workaround",
        "advisory",
        "D1",
        r"\bBilinearInterpolation\b",
        "BilinearInterpolation is natively fused on A2 but expanded on A5.",
        "If the A5 code replaced it with a manual gather+mul+add loop, re-test the API on A2.",
    ),
    _r(
        "PF-03",
        "perf.a5-workaround",
        "advisory",
        "D1",
        r"\bTransDataTo5HD\b",
        "TransDataTo5HD relies on vnchwconv, which is high throughput on A2 and slower on A5.",
        "If the A5 code hand-rolled a transpose to avoid the A5 slowdown, consider reverting to TransDataTo5HD.",
    ),
]


@dataclass
class Finding:
    rule: Rule
    file: str
    line: int
    column: int
    snippet: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "rule": self.rule.rule_id,
            "category": self.rule.category,
            "severity": self.rule.severity,
            "level": self.rule.level,
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "snippet": self.snippet,
            "message": self.rule.message,
            "a2_action": self.rule.a2_action,
        }


@dataclass
class ScanResult:
    root: str
    files_scanned: int = 0
    findings: List[Finding] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)

    def level(self) -> str:
        ranks = [LEVEL_RANK[f.rule.level] for f in self.findings]
        if not ranks:
            return "D1"
        return "D3" if max(ranks) >= 3 else ("D2" if max(ranks) == 2 else "D1")

    def counts(self) -> Dict[str, int]:
        out = {k: 0 for k in SEVERITY_ORDER}
        for f in self.findings:
            out[f.rule.severity] += 1
        return out

    def category_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for f in self.findings:
            out[f.rule.category] = out.get(f.rule.category, 0) + 1
        return dict(sorted(out.items()))


def iter_files(
    root: str, excludes: Sequence[Pattern[str]], skipped: List[Tuple[str, str]]
) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".git")
        )
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            if any(p.search(rel) for p in excludes):
                skipped.append((rel, "excluded"))
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in SOURCE_EXTS and name not in SPECIAL_NAMES:
                skipped.append((rel, "extension"))
                continue
            try:
                if os.path.getsize(full) > MAX_FILE_BYTES:
                    skipped.append((rel, "too_large"))
                    continue
            except OSError:
                skipped.append((rel, "stat_error"))
                continue
            yield full


def is_text(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(4096)
    except OSError:
        return False
    return b"\x00" not in chunk


def read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    return text.lstrip("\ufeff")


def scan(root: str, excludes: Sequence[Pattern[str]]) -> ScanResult:
    result = ScanResult(root=os.path.abspath(root))
    for path in iter_files(root, excludes, result.skipped):
        rel = os.path.relpath(path, root)
        if not is_text(path):
            result.skipped.append((rel, "binary"))
            continue
        text = read_text(path)
        if text is None:
            result.skipped.append((rel, "read_error"))
            continue
        result.files_scanned += 1
        lines = text.splitlines()
        line_starts = [0]
        for lm in re.finditer("\\n", text):
            line_starts.append(lm.end())
        seen: set = set()
        for rule in RULES:
            for m in rule.pattern.finditer(text):
                line_no = bisect.bisect_right(line_starts, m.start())
                key = (rule.rule_id, line_no)
                if key in seen:
                    continue
                seen.add(key)
                col = m.start() - (text.rfind("\n", 0, m.start()) + 1) + 1
                snippet = (
                    lines[line_no - 1].strip()[:200]
                    if 0 < line_no <= len(lines)
                    else ""
                )
                result.findings.append(Finding(rule, rel, line_no, col, snippet))
    result.findings.sort(key=lambda f: (f.file, f.line, f.rule.rule_id))
    return result


def render_markdown(result: ScanResult) -> str:
    counts = result.counts()
    cats = result.category_counts()
    level = result.level()
    out: List[str] = []
    out.append("# A5-only construct scan")
    out.append("")
    out.append(f"- root: `{result.root}`")
    out.append(
        f"- generated: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )
    out.append(f"- files scanned: {result.files_scanned}")
    out.append(f"- findings: {len(result.findings)}")
    out.append(f"- files skipped: {len(result.skipped)}")
    out.append(f"- level (initial): **{level}**")
    if result.skipped:
        out.append("")
        out.append("| skipped file | reason |")
        out.append("|---|---|")
        for rel, reason in result.skipped:
            out.append(f"| `{rel}` | {reason} |")
    out.append("")
    out.append("| severity | count |")
    out.append("|---|---|")
    for sev in ("blocking", "decision", "verify", "advisory"):
        out.append(f"| {sev} | {counts[sev]} |")
    out.append("")
    if cats:
        out.append("| category | count |")
        out.append("|---|---|")
        for cat, n in cats.items():
            out.append(f"| {cat} | {n} |")
        out.append("")
    if not result.findings:
        out.append(
            "No A5-only constructs were detected. Continue with project wiring checks "
            "(stage 0/2) — the scanner only sees literal tokens."
        )
        out.append("")
    else:
        by_cat: Dict[str, List[Finding]] = {}
        for f in result.findings:
            by_cat.setdefault(f.rule.category, []).append(f)
        for cat in sorted(by_cat):
            out.append(f"## {cat}")
            out.append("")
            out.append("| location | rule | severity | hit | A2 action |")
            out.append("|---|---|---|---|---|")
            for f in by_cat[cat]:
                loc = f"`{f.file}:{f.line}`"
                hit = f.snippet.replace("|", "\\|")[:120]
                action = f.rule.a2_action.replace("|", "\\|")
                out.append(
                    f"| {loc} | {f.rule.rule_id} | {f.rule.severity} | `{hit}` | {action} |"
                )
            out.append("")
    out.append("## A2 restore checklist (evaluate in stage 5)")
    out.append("")
    for item in RESTORE_CHECKLIST:
        out.append(f"- {item}")
    out.append("")
    out.append("## Next")
    out.append("")
    out.append(
        "1. Fill the inventory table in the analysis report (assets/migration-analysis.md.template)."
    )
    out.append("2. Close every decision item with the user before stage 2.")
    out.append(
        "3. Treat `verify` items as blocking until the target CANN headers are checked."
    )
    out.append("")
    return "\n".join(out)


def render_json(result: ScanResult) -> str:
    payload = {
        "schema": SCHEMA,
        "tool_version": TOOL_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "root": result.root,
        "summary": {
            "files_scanned": result.files_scanned,
            "findings": len(result.findings),
            "by_severity": result.counts(),
            "by_category": result.category_counts(),
            "level": result.level(),
            "files_skipped": len(result.skipped),
            "skipped": [
                {"file": rel, "reason": reason} for rel, reason in result.skipped
            ],
        },
        "findings": [f.to_dict() for f in result.findings],
        "a2_restore_checklist": RESTORE_CHECKLIST,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_excludes(raw: Sequence[str]) -> List[Pattern[str]]:
    return [re.compile(p) for p in raw]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Static scan for A5-only (DAV_3510 / arch35) constructs in an Ascend C operator project.",
    )
    parser.add_argument("--root", required=True, help="operator project root")
    parser.add_argument(
        "--format",
        choices=("md", "json", "both"),
        default="md",
        help="output format (default: md)",
    )
    parser.add_argument(
        "--out",
        help="output file; with --format both, use as a prefix (writes <prefix>.md/.json)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="regex of relative paths to skip (repeatable)",
    )
    parser.add_argument(
        "--fail-on",
        choices=("none", "blocking", "decision"),
        default="none",
        help=(
            "exit 1 when findings at or above this severity exist: "
            "blocking = blocking+verify; decision = blocking+decision+verify"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print the markdown report to stdout",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.root):
        print(f"error: --root is not a directory: {args.root}", file=sys.stderr)
        return 2

    try:
        excludes = parse_excludes(args.exclude)
    except re.error as exc:
        print(f"error: invalid --exclude regex: {exc}", file=sys.stderr)
        return 2
    result = scan(args.root, excludes)
    md = render_markdown(result)
    js = render_json(result)

    if args.out:
        if args.format == "both":
            md_path = args.out + ".md"
            js_path = args.out + ".json"
        elif args.format == "json":
            md_path, js_path = None, args.out
        else:
            md_path, js_path = args.out, None
        try:
            if md_path:
                with open(md_path, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(md)
                print(
                    f"wrote {md_path}",
                    file=sys.stderr if args.format == "json" else sys.stdout,
                )
            if js_path:
                with open(js_path, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(js)
                print(
                    f"wrote {js_path}",
                    file=sys.stderr if args.format == "json" else sys.stdout,
                )
        except OSError as exc:
            print(f"error: cannot write output: {exc}", file=sys.stderr)
            return 2

    # Keep stdout machine-readable when JSON is the requested format.
    status = sys.stderr if args.format == "json" else sys.stdout
    if not args.quiet:
        print(md if args.format in ("md", "both") else js)

    counts = result.counts()
    print(
        f"SUMMARY level={result.level()} files={result.files_scanned} skipped={len(result.skipped)} "
        f"blocking={counts['blocking']} decision={counts['decision']} "
        f"verify={counts['verify']} advisory={counts['advisory']}",
        file=status,
    )

    if args.fail_on != "none":
        if args.fail_on == "blocking" and (counts["blocking"] or counts["verify"]):
            return 1
        if args.fail_on == "decision" and (
            counts["blocking"] or counts["verify"] or counts["decision"]
        ):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
