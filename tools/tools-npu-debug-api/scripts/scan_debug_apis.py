#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Scan AscendC source files for supported and excluded Kernel debug APIs.

This is a lightweight source check. It does not compile code or infer a
complete programming mode; its purpose is to catch obvious API-scope and
header mistakes before a user runs a kernel. The report field "supported"
means covered by this skill, not verified support on a target platform.
Comments, string literals and inactive preprocessor branches are not filtered;
header checks do not resolve transitive includes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List


SUPPORTED_PATTERNS = (
    ("printf", re.compile(r"\b(?:AscendC::)?(?:printf|PRINTF)\s*\(")),
    ("assert", re.compile(r"\b(?:assert|ascendc_assert)\s*\(")),
    ("__trap", re.compile(r"\b__trap\s*\(")),
    ("asc_dump", re.compile(r"\basc_dump(?:_(?:gm|ubuf|l1buf|cbuf|reg))?\s*(?:<[^;{}()]*>)?\s*\(")),
    ("clock", re.compile(r"\bclock\s*\(")),
    ("asc_time_stamp", re.compile(r"\basc_time_stamp\s*\(")),
)

EXCLUDED_APIS = {
    "asc_prof_start",
    "asc_prof_stop",
    "asc_mark_stamp",
    "TRACE_START",
    "TRACE_STOP",
}

EXCLUDED_PATTERNS = {
    api: re.compile(
        r"\b" + re.escape(api) + r"(?:\s*<[^;{}()]*>)?\s*\("
    )
    for api in EXCLUDED_APIS
}

REQUIRED_HEADERS = {
    "printf": "utils/debug/asc_printf.h",
    "assert": "utils/debug/asc_assert.h",
    "__trap": "utils/debug/asc_assert.h",
    "asc_dump": "utils/debug/asc_dump.h",
    "clock": "utils/debug/asc_time.h",
    "asc_time_stamp": "utils/debug/asc_time.h",
}


def _line_numbers(text: str, pattern: re.Pattern[str]) -> List[int]:
    return [index for index, line in enumerate(text.splitlines(), start=1) if pattern.search(line)]


def scan_text(text: str, filename: str = "<memory>") -> Dict[str, object]:
    """Return a stable report for one source text."""

    supported: List[str] = []
    matches: Dict[str, List[int]] = {}
    for api, pattern in SUPPORTED_PATTERNS:
        lines = _line_numbers(text, pattern)
        if lines:
            supported.append(api)
            matches[api] = lines

    excluded: List[str] = []
    excluded_matches: Dict[str, List[int]] = {}
    for api in sorted(EXCLUDED_APIS):
        lines = _line_numbers(text, EXCLUDED_PATTERNS[api])
        if lines:
            excluded.append(api)
            excluded_matches[api] = lines

    headers = set(re.findall(r'#include\s*[<"]([^>"]+)[>"]', text))
    # kernel_operator.h is the umbrella header used by the repository's
    # examples and exposes the public debug entry points.
    missing_headers = {} if "kernel_operator.h" in headers else {
        api: REQUIRED_HEADERS[api]
        for api in supported
        if REQUIRED_HEADERS[api] not in headers
    }

    return {
        "files": [filename],
        "supported": supported,
        "excluded": excluded,
        "matches": matches,
        "excluded_matches": excluded_matches,
        "missing_headers": missing_headers,
    }


def scan_file(path: Path) -> Dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: source is not valid UTF-8 ({exc})") from exc
    return scan_text(text, str(path))


def _merge_reports(reports: Iterable[Dict[str, object]]) -> Dict[str, object]:
    report_list = list(reports)
    supported: List[str] = []
    excluded: List[str] = []
    for report in report_list:
        for api in report["supported"]:  # type: ignore[index]
            if api not in supported:
                supported.append(api)
        for api in report["excluded"]:  # type: ignore[index]
            if api not in excluded:
                excluded.append(api)
    return {
        "files": report_list,
        "supported": supported,
        "excluded": sorted(excluded),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="AscendC source files to scan")
    parser.add_argument("--json", action="store_true", help="write a machine-readable JSON report")
    parser.add_argument(
        "--fail-on-excluded",
        action="store_true",
        help="return 1 for a textual profiling API match (not a platform support check)",
    )
    args = parser.parse_args(argv)

    if args.paths:
        try:
            report = _merge_reports(scan_file(path) for path in args.paths)
        except (OSError, ValueError) as exc:
            print(f"scan_debug_apis: {exc}", file=sys.stderr)
            return 2
    else:
        report = _merge_reports([scan_text(sys.stdin.read(), "<stdin>")])

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for file_report in report["files"]:
            path = file_report["files"][0]
            supported = ", ".join(file_report["supported"]) or "none"
            excluded = ", ".join(file_report["excluded"]) or "none"
            print(f"{path}: supported={supported}; excluded={excluded}")
            missing = file_report["missing_headers"]
            for api, header in missing.items():
                print(f"  warning: {api} has no directly visible #include \"{header}\"")

    return 1 if args.fail_on_excluded and report["excluded"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
