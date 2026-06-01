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
"""Check kernel catalog consistency against repository files and generated index."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_agent_index import parse_catalog

ROOT = TOOLS_DIR.parent.parent
DEFAULT_CATALOG = ROOT / "agent" / "references" / "examples" / "kernel-catalog.md"
DEFAULT_INDEX = ROOT / "agent" / "index" / "kernels.json"
DEFAULT_KERNEL_DIR = ROOT / "agent" / "example" / "kernels"
REQUIRED_SCALAR_FIELDS = ["topology"]
REQUIRED_LIST_FIELDS = ["study_for", "do_not_copy_when"]
REQUIRED_TEXT_FIELDS = ["formula"]


def _warning(code: str, path: str, message: str, field: str = "") -> Dict[str, Any]:
    payload = {
        "code": code,
        "path": path,
        "message": message,
    }
    if field:
        payload["field"] = field
    return payload


def _kernel_files(root: Path, kernel_dir: Path) -> List[str]:
    return sorted(str(path.relative_to(root)) for path in kernel_dir.glob("*.py") if path.is_file())


def _load_json(path: Path) -> Tuple[Dict[str, Any], str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except json.JSONDecodeError as exc:
        return {}, f"Invalid JSON: {exc}"


def analyze_repo(
    root: Path = ROOT, catalog_path: Path = DEFAULT_CATALOG,
    index_path: Path = DEFAULT_INDEX, kernel_dir: Path = DEFAULT_KERNEL_DIR,
) -> Dict[str, Any]:
    warnings: List[Dict[str, Any]] = []

    if not catalog_path.exists():
        return {
            "warning_count": 1,
            "warnings": [_warning("missing-catalog", str(catalog_path), "Kernel catalog file does not exist.")],
            "entry_count": 0,
            "catalog_paths": [],
            "kernel_files": [],
        }

    payload = parse_catalog(catalog_path, kind="kernel", root=root)
    entries = payload["entries"]
    catalog_paths: List[str] = []
    seen_paths = set()

    for entry in entries:
        path = entry["path"]
        catalog_paths.append(path)

        if path in seen_paths:
            warnings.append(_warning(
                "duplicate-entry-path", path,
                "Catalog contains the same kernel path more than once.",
            ))
        else:
            seen_paths.add(path)

        file_path = root / path
        if not file_path.exists():
            warnings.append(_warning("missing-file", path, "Catalog entry points to a file that does not exist."))

        for field in REQUIRED_SCALAR_FIELDS:
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                warnings.append(_warning(
                    "missing-field", path,
                    "Required scalar field is missing or empty.", field=field,
                ))

        for field in REQUIRED_TEXT_FIELDS:
            value = entry.get(field)
            if isinstance(value, str):
                if not value.strip():
                    warnings.append(_warning(
                        "missing-field", path,
                        "Required text field is missing or empty.", field=field,
                    ))
            elif isinstance(value, list):
                if not value or any(not isinstance(item, str) or not item.strip() for item in value):
                    warnings.append(_warning(
                        "missing-field", path,
                        "Required text field is missing or empty.", field=field,
                    ))
            else:
                warnings.append(_warning(
                    "missing-field", path,
                    "Required text field is missing or empty.", field=field,
                ))

        for field in REQUIRED_LIST_FIELDS:
            value = entry.get(field)
            if not isinstance(value, list) or not value:
                warnings.append(_warning(
                    "missing-field", path,
                    "Required list field is missing or empty.", field=field,
                ))
            elif any(not isinstance(item, str) or not item.strip() for item in value):
                warnings.append(_warning(
                    "malformed-field", path,
                    "List field contains an empty or non-string item.", field=field,
                ))

    kernel_files = _kernel_files(root, kernel_dir)
    missing_from_catalog = sorted(set(kernel_files) - set(catalog_paths))
    for path in missing_from_catalog:
        warnings.append(_warning(
            "uncataloged-kernel", path,
            "Kernel file exists under kernels/ but is missing from kernel-catalog.md.",
        ))

    if not index_path.exists():
        warnings.append(_warning(
            "missing-index", str(index_path.relative_to(root)),
            "Generated kernel index file is missing. Run `python3 tools/build_agent_index.py`.",
        ))
    else:
        index_payload, json_error = _load_json(index_path)
        if json_error:
            warnings.append(_warning("invalid-index-json", str(index_path.relative_to(root)), json_error))
        else:
            expected_entries = payload["entries"]
            actual_entries = index_payload.get("entries")
            if actual_entries != expected_entries:
                warnings.append(
                    _warning(
                        "stale-index",
                        str(index_path.relative_to(root)),
                        "Generated kernel index does not match the current catalog."
                        " Run `python3 tools/build_agent_index.py`.",
                    )
                )

    return {
        "warning_count": len(warnings),
        "warnings": warnings,
        "entry_count": len(entries),
        "catalog_paths": catalog_paths,
        "kernel_files": kernel_files,
    }


def print_text(result: Dict[str, Any]) -> None:
    if not result["warnings"]:
        print("No warnings found.")
        return

    for warning in result["warnings"]:
        suffix = f" | field={warning['field']}" if "field" in warning else ""
        print(f"- [{warning['code']}] {warning['path']}{suffix}")
        print(f"  {warning['message']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check kernel catalog consistency against repository files and generated index.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--fail-on-warning", action="store_true", help="Return exit code 1 if any warning is found.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    result = analyze_repo()

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_text(result)

    if args.fail_on_warning and result["warning_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
