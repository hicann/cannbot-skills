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

"""
Pre-build static analysis gate for AscendC kernel headers.

Runs BEFORE compile on the NPU lane — no CANN runtime, no NPU needed.
Catches ~75% of the debug iterations we hit in LightningIndexerGrad (P128-P134):
  - UB buffer layout overlaps / overflow
  - AiCore-only SyncAll misuse
  - Event Alloc/Free lifecycle violations
  - 32B alignment constraint violations

Usage:
  python3 pre_build_check.py <kernel_header.h> [--sync-audit] [--verbose]
  python3 pre_build_check.py <kernel_header.h> --json  # machine-readable output

Exit codes:
  0 = ALL PASS
  1 = FAIL (at least one check failed)
  2 = PARSE_ERROR (could not parse header)
"""

import logging
import re
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class UbBuffer:
    """Parsed UB buffer declaration."""
    name: str           # e.g. "gatherPingUb"
    offset: int         # bytes
    size: int           # bytes
    line: int           # source line number
    raw_expr: str = ""  # original expression text


@dataclass
class Finding:
    """A single check finding."""
    severity: str       # "ERROR" | "WARN"
    check: str          # "UB_LAYOUT" | "SYNC_AUDIT" | "EVENT_LIFECYCLE" | "ALIGNMENT"
    line: int           # 0 if not line-specific
    message: str
    suggestion: str = ""


@dataclass
class CheckResult:
    file_path: str
    total_size: int = 193536  # default UB TOTAL_SIZE
    buffers: List[UbBuffer] = field(default_factory=list)
    max_ub_size: Optional[int] = None
    findings: List[Finding] = field(default_factory=list)
    sync_all_count: int = 0
    pipe_barrier_count: int = 0
    event_alloc_count: int = 0
    event_free_count: int = 0

    @property
    def passed(self) -> bool:
        return not any(f.severity == "ERROR" for f in self.findings)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _read_kernel_header(filepath: str) -> Tuple[str, List[str]]:
    """Read a kernel header or retain the public missing-file failure."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Header not found: {filepath}")
    content = path.read_text(encoding="utf-8", errors="replace")
    return content, content.split("\n")


def _set_total_size(result: CheckResult, content: str) -> None:
    """Set the declared total UB size when the header supplies one."""
    total_match = re.search(r"TOTAL_SIZE\s*=\s*(\d+)\s*\*\s*(\d+)", content)
    if total_match:
        result.total_size = int(total_match.group(1)) * int(total_match.group(2))
        return

    total_match = re.search(r"TOTAL_SIZE\s*=\s*(\d+)", content)
    if total_match:
        result.total_size = int(total_match.group(1))


def _parse_constants(lines: List[str]) -> Dict[str, int]:
    """Collect standalone numeric constexpr values used by buffer expressions."""
    constexpr_pattern = re.compile(
        r"constexpr\s+(?:static\s+)?(?:int64_t|int32_t|uint32_t|uint64_t|uint16_t|int|long|size_t)\s+"
        r"(\w+)\s*=\s*(.+?)\s*;"
    )
    constants: Dict[str, int] = {}

    for line in lines:
        match = constexpr_pattern.search(line)
        if not match:
            continue

        name = match.group(1)
        expr = match.group(2).strip()
        if name.endswith("Offset") or name.endswith("Size"):
            continue
        try:
            constants[name] = int(expr)
        except ValueError:
            try:
                value = _eval_simple_expr(expr)
                if value is not None:
                    constants[name] = value
            except Exception as error:
                logging.getLogger(__name__).debug(
                    "Recoverable operation failed.", exc_info=error
                )
    return constants


def _parse_buffer_definitions(
    lines: List[str], constants: Dict[str, int], result: CheckResult
) -> Tuple[Dict[str, Tuple[int, int, str]], Dict[str, Tuple[int, int, str]]]:
    """Evaluate offset, size, and maximum-workspace declarations in source order."""
    offset_pattern = re.compile(
        r"constexpr\s+static\s+int64_t\s+(\w+Offset)\s*=\s*(.+?)\s*;"
    )
    size_pattern = re.compile(
        r"constexpr\s+static\s+int64_t\s+(\w+Size)\s*=\s*(.+?)\s*;"
    )
    max_ub_pattern = re.compile(
        r"constexpr\s+static\s+int64_t\s+MAX_UB_SIZE\s*=\s*(.+?)\s*;"
    )
    offsets: Dict[str, Tuple[int, int, str]] = {}
    sizes: Dict[str, Tuple[int, int, str]] = {}

    for line_number, line in enumerate(lines, start=1):
        match = offset_pattern.search(line)
        if match:
            name, expr = match.group(1), match.group(2).strip()
            offsets[name.replace("Offset", "")] = (
                _eval_expr(expr, offsets, sizes, constants), line_number, expr
            )

        match = size_pattern.search(line)
        if match:
            name, expr = match.group(1), match.group(2).strip()
            sizes[name.replace("Size", "")] = (
                _eval_expr(expr, offsets, sizes, constants), line_number, expr
            )

        match = max_ub_pattern.search(line)
        if match:
            result.max_ub_size = _eval_expr(
                match.group(1).strip(), offsets, sizes, constants
            )
    return offsets, sizes


def _append_declared_buffers(
    result: CheckResult,
    offsets: Dict[str, Tuple[int, int, str]],
    sizes: Dict[str, Tuple[int, int, str]],
) -> None:
    """Pair parsed offsets and sizes into the result's ordered buffer list."""
    buffer_names = set(offsets) & set(sizes)
    for name in sorted(buffer_names, key=lambda buffer_name: offsets[buffer_name][0]):
        offset, line, offset_expr = offsets[name]
        size, _, size_expr = sizes[name]
        result.buffers.append(UbBuffer(
            name=name,
            offset=offset,
            size=size,
            line=line,
            raw_expr=f"offset={offset_expr}, size={size_expr}",
        ))


def _count_sync_and_events(result: CheckResult, content: str) -> None:
    """Populate static sync and event call counters from the source text."""
    result.sync_all_count = len(re.findall(r"SyncAll\s*\(\)", content))
    result.pipe_barrier_count = len(
        re.findall(r"PipeBarrier\s*<\s*PIPE_ALL\s*>", content)
    )
    result.event_alloc_count = len(re.findall(r"AllocEvent\s*\(\s*(\w+)", content))
    result.event_free_count = len(re.findall(r"FreeEvent\s*\(\s*(\w+)", content))


def _append_datacopy_alignment_findings(result: CheckResult, lines: List[str]) -> None:
    """Report literal DataCopy counts that cannot meet the common alignment rule."""
    datacopy_pattern = re.compile(r"DataCopy\s*\([^,]+,\s*[^,]+,\s*(\d+)\s*\)")
    for line_number, line in enumerate(lines, start=1):
        for match in datacopy_pattern.finditer(line):
            count = int(match.group(1))
            if count % 8 != 0:
                result.findings.append(Finding(
                    severity="WARN", check="ALIGNMENT", line=line_number,
                    message=(
                        f"DataCopy simple element count={count} is not a multiple "
                        "of 8; verify 32B alignment against the actual source type."
                    ),
                    suggestion=(
                        "Align count * sizeof(T) to 32B, or use the target-supported "
                        "tail-copy primitive documented for this SoC and direction."
                    ),
                ))


def _append_tquebind_alignment_findings(result: CheckResult, lines: List[str]) -> None:
    """Report literal TQueBind strides that cannot meet the common alignment rule."""
    stride_pattern = re.compile(r"TQueBind\s*<[^>]*>\s*\(\s*[^,]*,\s*[^,]*,\s*(\d+)\s*\)")
    for line_number, line in enumerate(lines, start=1):
        for match in stride_pattern.finditer(line):
            stride = int(match.group(1))
            if stride % 8 != 0:
                result.findings.append(Finding(
                    severity="WARN", check="ALIGNMENT", line=line_number,
                    message=(
                        f"TQueBind stride={stride} elements is not a multiple of 8; "
                        "verify 32B alignment against the actual source type."
                    ),
                    suggestion="Ensure stride * sizeof(T) is 32B-aligned.",
                ))


def parse_kernel_header(filepath: str) -> CheckResult:
    """Parse a kernel header file and extract UB buffer declarations + sync/event usage."""
    content, lines = _read_kernel_header(filepath)

    result = CheckResult(file_path=filepath)
    _set_total_size(result, content)
    constants = _parse_constants(lines)
    offsets, sizes = _parse_buffer_definitions(lines, constants, result)
    _append_declared_buffers(result, offsets, sizes)
    _count_sync_and_events(result, content)
    _append_datacopy_alignment_findings(result, lines)
    _append_tquebind_alignment_findings(result, lines)
    return result


def _eval_simple_expr(expr: str) -> Optional[int]:
    """Evaluate a simple numeric expression like '128 * 4 * 8' or '2048'."""
    expr = expr.strip()
    try:
        return int(expr)
    except ValueError:
        pass
    # Try arithmetic evaluation
    safe_expr = re.sub(r'[^0-9+\-*/().%\s]', '', expr)
    if safe_expr.strip():
        try:
            return int(eval(safe_expr))
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
    return None


def _eval_expr(expr: str, offsets: Dict, sizes: Dict, constants: Dict[str, int] = None) -> int:
    """Best-effort evaluation of a constexpr expression.

    Handles patterns like:
      - <literal>
      - <CONSTANT_NAME> (e.g. LIMIT_TOPK, LIMIT_GROUPNUM)
      - <name>Offset + <name>Size  (chain reference)
      - (TOTAL_SIZE - (<offset_expr>)) / sizeof(float)
      - <num> * <num>
    """
    if constants is None:
        constants = {}

    expr = expr.strip()

    # Pure literal
    try:
        return int(expr)
    except ValueError:
        pass

    # Replace named constants (LIMIT_TOPK, LIMIT_GROUPNUM, etc.)
    for name, val in constants.items():
        # Use word-boundary replacement to avoid partial matches
        expr = re.sub(r'\b' + re.escape(name) + r'\b', f"({val})", expr)

    # Replace known offset/size references with their evaluated values
    for base, (val, _, _) in sorted(offsets.items(), key=lambda x: -len(x[0])):
        expr = re.sub(r'\b' + re.escape(base + "Offset") + r'\b', f"({val})", expr)
    for base, (val, _, _) in sorted(sizes.items(), key=lambda x: -len(x[0])):
        expr = re.sub(r'\b' + re.escape(base + "Size") + r'\b', f"({val})", expr)

    # Replace TOTAL_SIZE
    if "TOTAL_SIZE" in expr:
        expr = expr.replace("TOTAL_SIZE", "193536")

    # sizeof(float) -> 4
    expr = expr.replace("sizeof(float)", "4")
    expr = expr.replace("sizeof(half)", "2")

    # Try to eval
    try:
        # Security: only allow basic arithmetic
        safe_expr = re.sub(r'[^0-9+\-*/().%\s]', '', expr)
        if safe_expr.strip():
            return int(eval(safe_expr))
    except Exception as error:
        logging.getLogger(__name__).debug(
            "Recoverable operation failed.", exc_info=error
        )

    return 0


# ---------------------------------------------------------------------------
# Checkers
# ---------------------------------------------------------------------------

def _append_buffer_chain_findings(result: CheckResult, buffers: List[UbBuffer]) -> None:
    """Append findings for gaps and overlaps in the declared buffer chain."""
    for i in range(1, len(buffers)):
        prev = buffers[i - 1]
        curr = buffers[i]
        expected = prev.offset + prev.size
        if curr.offset != expected:
            gap = curr.offset - expected
            direction = "OVERLAP" if gap < 0 else "GAP"
            result.findings.append(Finding(
                severity="ERROR",
                check="UB_LAYOUT",
                line=curr.line,
                message=(
                    f"UB buffer chain broken at '{curr.name}': "
                    f"offset={curr.offset}, but prev='{prev.name}' ends at "
                    f"{prev.offset}+{prev.size}={expected} "
                    f"({direction} {abs(gap)} bytes)"
                ),
                suggestion=(
                    f"Change {curr.name}Offset from {curr.offset} to {expected} "
                    f"(= {prev.name}Offset + {prev.name}Size). "
                    f"This looks like a copy-paste error — the Offset base "
                    f"should be the previous buffer's end, not 0."
                ) if gap < 0 else (
                    f"Gap between buffers: {curr.name} starts at {curr.offset} "
                    f"but prev ends at {expected}. Insert missing buffer or fix offset."
                )
            ))


def _append_total_size_finding(result: CheckResult, last: UbBuffer) -> None:
    """Append an error when the final buffer exceeds the declared UB size."""
    declared_end = last.offset + last.size
    if declared_end > result.total_size:
        result.findings.append(Finding(
            severity="ERROR",
            check="UB_LAYOUT",
            line=last.line,
            message=(
                f"UB buffer overflow: declared buffers end at {declared_end} bytes, "
                f"but TOTAL_SIZE={result.total_size} bytes"
            ),
            suggestion="Reduce buffer sizes, or verify TOTAL_SIZE is correct for this SoC."
        ))


def _append_workspace_finding(result: CheckResult, last: UbBuffer) -> None:
    """Append MAX_UB_SIZE overflow or low-workspace diagnostics."""
    declared_end = last.offset + last.size
    if result.max_ub_size is not None:
        remaining = result.total_size - declared_end
        max_ub_bytes = result.max_ub_size * 4  # floats → bytes
        if max_ub_bytes > remaining:
            result.findings.append(Finding(
                severity="ERROR",
                check="UB_LAYOUT",
                line=0,
                message=(
                    f"MAX_UB_SIZE={result.max_ub_size} floats ({max_ub_bytes} bytes) "
                    f"exceeds available workspace {remaining} bytes "
                    f"({remaining//4} floats). Overflow={max_ub_bytes - remaining} bytes "
                    f"({(max_ub_bytes - remaining)//4} floats)"
                ),
                suggestion=(
                    f"MAX_UB_SIZE should be (TOTAL_SIZE - ({last.name}Offset + {last.name}Size)) / sizeof(float) "
                    f"= {remaining // 4}"
                )
            ))
    else:
        # No explicit MAX_UB_SIZE — compute and report
        remaining = result.total_size - declared_end
        if remaining < 1024:
            result.findings.append(Finding(
                severity="WARN",
                check="UB_LAYOUT",
                line=0,
                message=(
                    f"Only {remaining} bytes ({remaining//4} floats) remaining for workspace. "
                    f"Consider reducing declared buffer sizes."
                ),
                suggestion="Audit buffer sizes; reduce where possible."
            ))



def check_ub_layout(result: CheckResult) -> None:
    """Verify UB buffer chain: no overlaps, no overflow."""
    buffers = result.buffers
    if not buffers:
        return

    last = buffers[-1]
    _append_buffer_chain_findings(result, buffers)
    _append_total_size_finding(result, last)
    _append_workspace_finding(result, last)


def _is_aicore_only(content: str) -> bool:
    """Return whether code tokens omit the mixed-AiCore pipeline marker."""
    code_lines = [line for line in content.split("\n") if not line.strip().startswith("//")]
    return "MIX_AIC" not in "\n".join(code_lines)


def _append_aicore_sync_findings(result: CheckResult, content: str) -> None:
    """Append unsafe AiCore-only SyncAll findings at their source locations."""
    if result.sync_all_count == 0 or not _is_aicore_only(content):
        return

    for line_number, line in enumerate(content.split("\n"), start=1):
        if "SyncAll()" in line:
            result.findings.append(Finding(
                severity="ERROR",
                check="SYNC_AUDIT",
                line=line_number,
                message=(
                    f"SyncAll() found in AiCore-only pipeline at line {line_number}. "
                    f"SyncAll is a CROSS-CORE barrier — ALL scheduled cores must "
                    f"reach it. In AiCore-only mode, each core operates on "
                    f"independent data, so cross-core sync is unnecessary AND harmful: "
                    f"(1) zero-work cores that skip the batch loop will never reach "
                    f"SyncAll, causing working cores to hang forever; "
                    f"(2) even with all cores working, any core progressing at a "
                    f"different rate blocks all others."
                ),
                suggestion=(
                    "Replace SyncAll() with PipeBarrier<PIPE_ALL>(). "
                    "PipeBarrier<PIPE_ALL> synchronizes V/MTE1/MTE2/MTE3 within "
                    "a single core — ensuring MTE3 writes are visible to the next "
                    "pipeline stage without cross-core dependency. "
                    "Exception: SyncAll is acceptable in Pre/Post pipelines where "
                    "ALL cores share global initialization/finalization."
                )
            ))


def _append_loop_sync_findings(result: CheckResult, lines: List[str]) -> None:
    """Append findings for SyncAll calls observed inside source loops."""
    loop_depth = 0
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if re.match(r"\b(for|while)\s*\(", stripped):
            loop_depth += 1
        if stripped.startswith("}"):
            loop_depth = max(0, loop_depth - 1)
        if "SyncAll()" in stripped and loop_depth > 0:
            result.findings.append(Finding(
                severity="ERROR",
                check="SYNC_AUDIT",
                line=line_number,
                message=(
                    f"SyncAll() INSIDE a loop (depth={loop_depth}) at line {line_number}. "
                    f"This is the #1 cause of multi-core hangs in op-gen history. "
                    f"Per-iteration cross-core barriers couple independent core "
                    f"progress together and deadlock when any core diverges."
                ),
                suggestion="Replace with PipeBarrier<PIPE_ALL>() — intra-core sync only."
            ))


def check_sync_audit(result: CheckResult) -> None:
    """Verify sync primitive usage matches pipeline architecture.

    Rule: AiCore-only pipelines should use PipeBarrier<PIPE_ALL>, not SyncAll.
    SyncAll is a cross-core barrier — only needed when ALL cores must rendezvous
    (e.g., Pre-pipeline global init, Post-pipeline global finalize).
    In the main compute loop, each core works on independent data -> PipeBarrier.
    """
    content = Path(result.file_path).read_text(encoding="utf-8", errors="replace")
    lines = content.split("\n")
    _append_aicore_sync_findings(result, content)
    _append_loop_sync_findings(result, lines)


def _free_event_lines(lines: List[str]) -> List[int]:
    """Return source lines containing a FreeEvent call token."""
    return [
        line_number for line_number, line in enumerate(lines, start=1)
        if "FreeEvent" in line
    ]


def _append_loop_event_findings(
    result: CheckResult, lines: List[str], free_lines: List[int]
) -> None:
    """Append lifecycle warnings for FreeEvent calls observed in source loops."""
    in_loop = False
    loop_start = 0
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if re.match(r"\b(for|while)\s*\(", stripped):
            in_loop = True
            loop_start = line_number
        if stripped.startswith("}") and in_loop:
            for free_line in free_lines:
                if loop_start <= free_line <= line_number:
                    result.findings.append(Finding(
                        severity="WARN",
                        check="EVENT_LIFECYCLE",
                        line=free_line,
                        message=(
                            f"FreeEvent inside loop at line {free_line}. If events are "
                            f"allocated outside the loop, freeing them inside means "
                            f"Batch 2+ will hit an exhausted event pool."
                        ),
                        suggestion=(
                            "Either: (1) move FreeEvent outside the loop, or "
                            "(2) pair each FreeEvent inside the loop with a "
                            "corresponding AllocEvent at the top of the loop body."
                        )
                    ))
            in_loop = False


def check_event_lifecycle(result: CheckResult) -> None:
    """Check AllocEvent/FreeEvent pairing and loop-safety.

    Rule: Events allocated outside a loop should not be freed inside the loop
    (causes Batch 2+ deadlock when events are exhausted).
    """
    if result.event_alloc_count == 0 and result.event_free_count == 0:
        return

    content = Path(result.file_path).read_text(encoding="utf-8", errors="replace")
    lines = content.split("\n")
    _append_loop_event_findings(result, lines, _free_event_lines(lines))


def check_alignment(result: CheckResult) -> None:
    """Check 32B and 512B alignment constraints in DataCopy/TQueBind operations.

    V351 (Ascend950PR) constraints:
    - DataCopy count * sizeof(T) must be multiple of 32 for non-Pad variant
    - TQueBind stride * sizeof(T) must be multiple of 32
    """
    content = Path(result.file_path).read_text(encoding="utf-8", errors="replace")
    lines = content.split("\n")

    # Do not infer units from a bare ``blockLen`` assignment. DataCopyParams
    # measures this field in 32B blocks, while DataCopyPad extension params use
    # bytes. A global regex previously mislabeled the valid value 1 as "1 byte"
    # and suggested padding it to 32, which can create a 32x oversized transfer.

    # GM offset alignment: check expressions like <tensor>[<offset>]
    # where offset * sizeof(T) might not be 512B-aligned
    # (This is heuristic — false positives are possible)
    gm_offset_pattern = re.compile(r"(\w+GmTensor)\[(\d+)\]")
    for i, line in enumerate(lines, start=1):
        for m in gm_offset_pattern.finditer(line):
            offset = int(m.group(2))
            # Check 512B alignment for GM base offsets
            if offset > 0 and offset % 512 != 0:
                result.findings.append(Finding(
                    severity="WARN",
                    check="ALIGNMENT",
                    line=i,
                    message=(
                        f"GM tensor '{m.group(1)}' accessed at offset {offset} "
                        f"(not 512B-aligned). May cause performance degradation."
                    ),
                    suggestion="Align GM base offsets to 512B boundaries."
                ))


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def format_text(result: CheckResult) -> str:
    """Human-readable output."""
    lines = []
    lines.append(f"=== pre_build_check: {result.file_path} ===")
    lines.append("")

    # UB Layout summary
    if result.buffers:
        lines.append(f"  UB Buffers: {len(result.buffers)} declared, "
                     f"TOTAL_SIZE={result.total_size}")
        last = result.buffers[-1]
        declared_end = last.offset + last.size
        remaining = result.total_size - declared_end
        lines.append(f"  Declared: {declared_end} bytes, Remaining: {remaining} bytes "
                     f"({remaining//4} floats)")
        if result.max_ub_size:
            lines.append(f"  MAX_UB_SIZE: {result.max_ub_size} floats "
                         f"({result.max_ub_size * 4} bytes)")
        lines.append("")

    # Sync audit
    lines.append(f"  SyncAll calls: {result.sync_all_count}")
    lines.append(f"  PipeBarrier<PIPE_ALL> calls: {result.pipe_barrier_count}")
    lines.append("")

    # Event lifecycle
    lines.append(f"  AllocEvent calls: {result.event_alloc_count}")
    lines.append(f"  FreeEvent calls: {result.event_free_count}")
    lines.append("")

    # Findings
    if not result.findings:
        lines.append("  ✓ ALL CHECKS PASSED")
    else:
        errors = [f for f in result.findings if f.severity == "ERROR"]
        warns = [f for f in result.findings if f.severity == "WARN"]
        lines.append(f"  ✗ {len(errors)} ERROR(s), {len(warns)} WARNING(s)")
        lines.append("")

        for f in sorted(result.findings, key=lambda x: (0 if x.severity == "ERROR" else 1, x.check, x.line)):
            prefix = "ERROR" if f.severity == "ERROR" else "WARN"
            lines.append(f"  [{prefix}] [{f.check}] line {f.line or '?'}: {f.message}")
            if f.suggestion:
                lines.append(f"         → {f.suggestion}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_json(result: CheckResult) -> str:
    """Machine-readable JSON output."""
    return json.dumps({
        "file": result.file_path,
        "passed": result.passed,
        "total_size": result.total_size,
        "buffers": [
            {
                "name": b.name,
                "offset": b.offset,
                "size": b.size,
                "end": b.offset + b.size,
                "line": b.line,
            }
            for b in result.buffers
        ],
        "max_ub_size": result.max_ub_size,
        "declared_bytes": (result.buffers[-1].offset + result.buffers[-1].size) if result.buffers else 0,
        "remaining_bytes": (
            result.total_size - (result.buffers[-1].offset + result.buffers[-1].size)
            if result.buffers
            else result.total_size
        ),
        "sync_all_count": result.sync_all_count,
        "pipe_barrier_count": result.pipe_barrier_count,
        "event_alloc_count": result.event_alloc_count,
        "event_free_count": result.event_free_count,
        "findings": [
            {
                "severity": f.severity,
                "check": f.check,
                "line": f.line,
                "message": f.message,
                "suggestion": f.suggestion,
            }
            for f in result.findings
        ],
    }, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Pre-build static analysis gate for AscendC kernel headers"
    )
    parser.add_argument("header", help="Path to kernel header (.h) file")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--checks", nargs="*", default=["all"],
                        choices=["all", "ub", "sync", "event", "align"],
                        help="Which checks to run (default: all)")
    args = parser.parse_args()

    try:
        result = parse_kernel_header(args.header)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"PARSE_ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    run_all = "all" in args.checks or args.checks is None

    if run_all or "ub" in args.checks:
        check_ub_layout(result)
    if run_all or "sync" in args.checks:
        check_sync_audit(result)
    if run_all or "event" in args.checks:
        check_event_lifecycle(result)
    if run_all or "align" in args.checks:
        check_alignment(result)

    if args.json:
        print(format_json(result))
    else:
        print(format_text(result))
        if args.verbose and result.buffers:
            print("\n  Buffer layout:")
            for b in result.buffers:
                print(f"    {b.name:30s} [{b.offset:6d}, {b.offset + b.size:6d})  "
                      f"{b.size:6d} bytes")
            last = result.buffers[-1]
            end = last.offset + last.size
            print(f"    {'(WORKSPACE)':30s} [{end:6d}, {result.total_size:6d})  "
                  f"{result.total_size - end:6d} bytes")

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
