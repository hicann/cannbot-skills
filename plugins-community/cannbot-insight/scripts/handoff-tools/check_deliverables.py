#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check deliverable files: existence, size, stub signals. Output JSON."""

import json
import logging
import os
import re
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

STUB_PATTERNS = [
    re.compile(r'throw\s+new\s+Error', re.IGNORECASE),
    re.compile(r'PLACEHOLDER:', re.IGNORECASE),
    re.compile(r'TODO:', re.IGNORECASE),
    re.compile(r'FIXME:', re.IGNORECASE),
]

ARCH_DIRS = ["arch22", "arch23", "arch24"]

DELIVERABLE_PATHS_STAGE1 = [
    "docs/LOG.md",
    "docs/REQUIREMENTS.md",
    "docs/aclnn{OpName}.md",
    "docs/spec.yaml",
    "docs/DESIGN.md",
    "docs/PLAN.md",
    "docs/TEST.md",
    "docs/SPEC_REVIEW.md",
    "docs/DESIGN_REVIEW.md",
    "docs/TEST_REVIEW.md",
]

DELIVERABLE_PATHS_STAGE2_FLAT = [
    "op_host/{op}_def.cpp",
    "op_host/{op}_infershape.cpp",
]

DELIVERABLE_PATHS_STAGE2_ARCH = [
    ("op_host/{op}_tiling.cpp", ["op_host/{arch}/{op}_tiling.cpp"]),
    ("op_kernel/{op}.cpp", ["op_kernel/{op}_{arch}.cpp", "op_kernel/{arch}/{op}.cpp"]),
    ("op_kernel/{op}.h", ["op_kernel/{arch}/{op}.h"]),
    ("op_kernel/{op}_tiling_data.h", ["op_kernel/{arch}/{op}_tiling_data.h"]),
]

DELIVERABLE_PATHS_STAGE2_COMMON = [
    "op_graph/{op}_proto.h",
    "CMakeLists.txt",
    "probe/PROBE_SUMMARY.md",
    "tests/ut/",
    "tests/st/CMakeLists.txt",
    "tests/st/test_aclnn_{op}.cpp",
]

DELIVERABLE_PATHS_STAGE3 = [
    "docs/precision-report.md",
    "docs/performance-report.md",
]

DELIVERABLE_PATHS_STAGE4 = [
    "README.md",
    "examples/",
    "docs/review-report.md",
]

ALL_DELIVERABLE_TEMPLATES = (
    DELIVERABLE_PATHS_STAGE1
    + DELIVERABLE_PATHS_STAGE2_FLAT
    + DELIVERABLE_PATHS_STAGE2_COMMON
    + DELIVERABLE_PATHS_STAGE3
    + DELIVERABLE_PATHS_STAGE4
)


def _emit_json(obj):
    logging.info(json.dumps(obj, ensure_ascii=False, indent=2))


def detect_stub_signals(filepath):
    try:
        content = open(filepath, "r", encoding="utf-8", errors="ignore").read()
        matches = []
        for pattern in STUB_PATTERNS:
            found = pattern.findall(content)
            if found:
                matches.extend(found[:3])
        return matches
    except Exception:
        return []


def resolve_path(template, op_dir, op_name):
    p = template.replace("{op}", op_name).replace("{OpName}", op_name.title().replace("_", ""))
    return os.path.join(op_dir, p)


def _format_arch_template(alt_template, op_name, arch):
    p = alt_template.replace("{op}", op_name)
    p = p.replace("{arch}", arch)
    p = p.replace("{OpName}", op_name.title().replace("_", ""))
    return p


def resolve_arch_paths(template, arch_alternatives, op_dir, op_name):
    flat = resolve_path(template, op_dir, op_name)
    if os.path.exists(flat):
        return flat, flat

    for alt_template in arch_alternatives:
        for arch in ARCH_DIRS:
            p = _format_arch_template(alt_template, op_name, arch)
            resolved = os.path.join(op_dir, p)
            if os.path.exists(resolved):
                return flat, resolved

    return flat, flat


def _probe(path):
    if os.path.isfile(path):
        size = os.path.getsize(path)
        if size > 0:
            return size, detect_stub_signals(path)
        return size, []
    if os.path.isdir(path):
        return -1, []
    return 0, []


def check_deliverables(op_dir, op_name):
    results = []

    for template in ALL_DELIVERABLE_TEMPLATES:
        resolved = resolve_path(template, op_dir, op_name)
        exists = os.path.exists(resolved)
        size, stub = _probe(resolved) if exists else (0, [])
        results.append({
            "template": template,
            "resolved_path": resolved,
            "exists": exists,
            "size": size,
            "stub_signals": stub,
        })

    for flat_template, arch_alternatives in DELIVERABLE_PATHS_STAGE2_ARCH:
        flat_path, actual_path = resolve_arch_paths(flat_template, arch_alternatives, op_dir, op_name)
        exists = os.path.exists(actual_path)
        size, stub = _probe(actual_path) if exists else (0, [])
        results.append({
            "template": flat_template,
            "resolved_path": flat_path,
            "actual_path": actual_path if actual_path != flat_path else None,
            "exists": exists,
            "size": size,
            "stub_signals": stub,
        })

    return results


def main():
    if len(sys.argv) < 3:
        _emit_json({"error": "Usage: check_deliverables.py <docs_dir_or_operator_dir> <operator_name>"})
        sys.exit(1)

    target_path = sys.argv[1]
    op_name = sys.argv[2]

    if target_path.endswith("/docs"):
        op_dir = os.path.dirname(target_path)
    else:
        op_dir = target_path

    results = check_deliverables(op_dir, op_name)

    summary = {
        "operator_name": op_name,
        "operator_dir": op_dir,
        "total_deliverables": len(results),
        "existing_count": sum(1 for r in results if r["exists"]),
        "missing_count": sum(1 for r in results if not r["exists"]),
        "empty_count": sum(1 for r in results if r["exists"] and r["size"] == 0),
        "stub_count": sum(1 for r in results if r["stub_signals"]),
        "deliverables": results,
    }

    _emit_json(summary)


if __name__ == "__main__":
    main()
