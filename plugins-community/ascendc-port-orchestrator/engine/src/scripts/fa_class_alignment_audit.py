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

"""FA-class alignment audit — pre-spawn brief-time check for V220 32B alignment.

Per DS round-1 fold (PR #122 review msg DISCORD_ID_REDACTED): catches V220's
#1 crash class (ADDR_MISALIGN / 507035) at brief-review time instead of
A3-test time. DS empirical: 2h debugging on 26_AvgPool3d alignment bugs
that a 30-second grep would have caught.

Greps a kernel brief markdown for AscendC operations whose `count` parameter
governs DMA / VEC alignment, then statically checks count % (32 / sizeof(dtype))
== 0. For each op extracted, classifies as ALIGNED / MISALIGNED / INDETERMINATE.

Operations checked (each can take a count parameter that must be 32B-aligned):
- DataCopy / DataCopyPad (DMA, MTE2/MTE3 alignment)
- Cast (T → T conversion)
- Add / Mul / Div / Sub / Max / Min (VEC binary)
- Duplicate / Adds / Muls (VEC unary)

Dtype detection: looks for dtype declaration in surrounding 5-line window,
or `LocalTensor<fp16>` / `LocalTensor<half>` etc style annotations on the
same line.

Per design §6 Phase 1b: fully implemented (text-only, no NPU dep, schema-frozen).

Usage:
  python3 src/scripts/fa_class_alignment_audit.py <brief.md> [--json out.json] [--quiet]
  python3 src/scripts/fa_class_alignment_audit.py --help

Exit codes:
  0 — audit complete, no misalignments
  1 — misalignments detected (count ≥ 1)
  2 — usage error / brief file not found
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

# Add src/scripts/ to path so fa_class_schemas can be imported
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from fa_class_schemas import (  # noqa: E402
    AlignmentAudit, AlignmentOperation, AlignmentVerdict,
)


# Operations the audit scans for. Each entry: regex pattern + extraction rules.
# Pattern uses Python re with case-insensitive flag.
_OPERATIONS = [
    # DMA / MTE
    ("DataCopyPad", r"\bDataCopyPad\b"),
    ("DataCopy", r"\bDataCopy\b(?!Pad)"),     # negative lookahead so we don't double-match Pad
    # VEC
    ("Cast", r"\bCast\b"),
    ("Add", r"\bAdd\b(?!s)"),                  # don't match "Adds"
    ("Adds", r"\bAdds\b"),
    ("Mul", r"\bMul\b(?!s)"),
    ("Muls", r"\bMuls\b"),
    ("Div", r"\bDiv\b(?!s)"),
    ("Divs", r"\bDivs\b"),
    ("Sub", r"\bSub\b(?!s)"),
    ("Subs", r"\bSubs\b"),
    ("Max", r"\bMax\b"),
    ("Min", r"\bMin\b"),
    ("Duplicate", r"\bDuplicate\b"),
    # Optional reduction primitives — count is row width
    ("Exp", r"\bExp\b"),
    ("Ln", r"\bLn\b"),
    ("Sqrt", r"\bSqrt\b"),
]

# 32B alignment elements_per_block lookup
_ELEMENTS_PER_32B = {
    "fp16": 16,
    "half": 16,
    "float16": 16,
    "bf16": 16,
    "bfloat16": 16,
    "fp32": 8,
    "float": 8,
    "float32": 8,
    "int32": 8,
    "int8": 32,
    "uint8": 32,
    "fp64": 4,
    "double": 4,
    "float64": 4,
}

# Dtype hint patterns (extracted from same line or nearby tensor declarations)
_DTYPE_HINTS = list(_ELEMENTS_PER_32B.keys())
_DTYPE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(h) for h in _DTYPE_HINTS) + r")\b",
    re.IGNORECASE,
)


def _extract_count_arg(line: str) -> int | None:
    """Heuristic: extract the LAST integer arg in the operation call.

    Most AscendC primitives take `count` as the trailing argument
    (e.g. `Add(dst, src0, src1, count)`, `DataCopy(dst, src, params, count)`).
    A naive 'last integer literal' grab works for the common case.

    Returns None when no integer literal found (INDETERMINATE).
    """
    # Look for integer literals followed by `)` or `,)` near end of line
    # Strip everything after first `//` comment
    code_part = line.split("//", 1)[0]
    # Strip trailing whitespace + closing brace
    code_part = code_part.rstrip(" ;\n\t")
    # Find all integer literals (decimal, not hex/identifier)
    candidates = re.findall(r"(?<![a-zA-Z_0-9])(\d+)(?![a-zA-Z_0-9.])", code_part)
    if not candidates:
        return None
    try:
        return int(candidates[-1])
    except ValueError:
        return None


def _detect_dtype(lines: list[str], line_idx: int, window: int = 5) -> str | None:
    """Look in a ±window line range for dtype hints."""
    start = max(0, line_idx - window)
    end = min(len(lines), line_idx + window + 1)
    for i in range(start, end):
        m = _DTYPE_RE.search(lines[i])
        if m:
            return m.group(0).lower()
    return None


def audit_brief(brief_path: Path) -> AlignmentAudit:
    """Audit a brief markdown for V220 32B alignment compliance."""
    if not brief_path.exists():
        raise FileNotFoundError(f"brief not found: {brief_path}")
    text = brief_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    audit = AlignmentAudit(
        brief_path=str(brief_path),
        audited_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        operations=[],
    )

    for i, line in enumerate(lines, start=1):
        for op_name, pattern in _OPERATIONS:
            if not re.search(pattern, line, re.IGNORECASE):
                continue
            # Skip if line looks like a comment / non-call mention
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("#"):
                continue
            # Truncate text
            text_excerpt = stripped[:200]

            count = _extract_count_arg(line)
            dtype = _detect_dtype(lines, i - 1, window=5)

            if count is None:
                audit.operations.append(AlignmentOperation(
                    line=i, operation=op_name, text=text_excerpt,
                    verdict=AlignmentVerdict.INDETERMINATE,
                    reason="could not extract count arg from call",
                ))
                continue

            if dtype is None:
                audit.operations.append(AlignmentOperation(
                    line=i, operation=op_name, text=text_excerpt,
                    verdict=AlignmentVerdict.INDETERMINATE,
                    count=count,
                    reason="dtype not found in ±5-line window",
                ))
                continue

            elements_per_32b = _ELEMENTS_PER_32B[dtype]
            remainder = count % elements_per_32b
            if remainder == 0:
                audit.operations.append(AlignmentOperation(
                    line=i, operation=op_name, text=text_excerpt,
                    verdict=AlignmentVerdict.ALIGNED,
                    count=count, dtype=dtype,
                    elements_per_32b=elements_per_32b,
                ))
            else:
                audit.operations.append(AlignmentOperation(
                    line=i, operation=op_name, text=text_excerpt,
                    verdict=AlignmentVerdict.MISALIGNED,
                    count=count, dtype=dtype,
                    elements_per_32b=elements_per_32b,
                    remainder=remainder,
                ))
            break  # only one operation per line

    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="FA-class alignment audit — pre-spawn brief-time V220 32B alignment check"
    )
    parser.add_argument(
        "brief_path", type=Path,
        help="Path to kernel brief markdown to audit",
    )
    parser.add_argument(
        "--json", type=Path, default=None,
        help="Write JSON output to this path (default: stdout)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only print summary line to stdout",
    )
    args = parser.parse_args(argv)

    try:
        audit = audit_brief(args.brief_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    json_output = audit.to_json()
    if args.json:
        args.json.write_text(json_output, encoding="utf-8")
        if not args.quiet:
            print(f"wrote {args.json}")
    elif not args.quiet:
        print(json_output)

    # Summary line always to stdout (non-quiet) or stderr (quiet)
    summary = (
        f"alignment audit: {len(audit.operations)} ops, "
        f"{audit.n_aligned} aligned, "
        f"{audit.n_misaligned} misaligned, "
        f"{audit.n_indeterminate} indeterminate"
    )
    if args.quiet:
        print(summary, file=sys.stderr)
    else:
        print(summary)

    return 1 if audit.n_misaligned > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
