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
"""Build machine-readable agent indexes from the human-readable example catalogs."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
KERNEL_CATALOG = ROOT / "agent" / "references" / "examples" / "kernel-catalog.md"
TOOL_CATALOG = ROOT / "agent" / "references" / "examples" / "tool-catalog.md"
KERNEL_INDEX = ROOT / "agent" / "index" / "kernels.json"
TOOL_INDEX = ROOT / "agent" / "index" / "tools.json"

HEADING_RE = re.compile(r"^###\s+`([^`]+)`\s*$")
FIELD_RE = re.compile(r"^-\s+([A-Za-z0-9_ /-]+):(?:\s*(.*))?$")
LIST_ITEM_RE = re.compile(r"^\s+-\s+(.*\S)\s*$")
INLINE_CODE_RE = re.compile(r"^`([^`]+)`$")


def _normalize_field_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _clean_scalar(value: str) -> str:
    value = value.strip()
    match = INLINE_CODE_RE.match(value)
    if match:
        return match.group(1)
    return value


def _finalize_entry(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if entry is None:
        return None
    entry["name"] = Path(entry["path"]).name
    return entry


def parse_catalog(path: Path, kind: str, root: Path = ROOT) -> Dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    entries: List[Dict[str, Any]] = []
    current_category: Optional[str] = None
    current_entry: Optional[Dict[str, Any]] = None
    current_field: Optional[str] = None

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("## ") and not stripped.startswith("### "):
            finalized = _finalize_entry(current_entry)
            if finalized is not None:
                entries.append(finalized)
            current_entry = None
            current_field = None
            current_category = stripped[3:].strip()
            continue

        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            finalized = _finalize_entry(current_entry)
            if finalized is not None:
                entries.append(finalized)
            record_path = heading_match.group(1)
            current_entry = {
                "kind": kind,
                "path": record_path,
                "category": current_category,
            }
            current_field = None
            continue

        if current_entry is None:
            continue

        field_match = FIELD_RE.match(stripped)
        if field_match:
            field_name = _normalize_field_name(field_match.group(1))
            raw_value = field_match.group(2) or ""
            if raw_value:
                current_entry[field_name] = _clean_scalar(raw_value)
            else:
                current_entry[field_name] = []
            current_field = field_name
            continue

        list_item_match = LIST_ITEM_RE.match(line)
        if list_item_match and current_field:
            item_value = list_item_match.group(1).strip()
            existing_value = current_entry.get(current_field)
            if isinstance(existing_value, list):
                existing_value.append(item_value)
            elif existing_value:
                current_entry[current_field] = [existing_value, item_value]
            else:
                current_entry[current_field] = [item_value]
            continue

        if stripped and current_field:
            existing_value = current_entry.get(current_field)
            if isinstance(existing_value, list):
                if existing_value:
                    existing_value[-1] = existing_value[-1] + " " + stripped
                else:
                    existing_value.append(stripped)
            elif existing_value:
                current_entry[current_field] = str(existing_value) + " " + stripped
            else:
                current_entry[current_field] = stripped

    finalized = _finalize_entry(current_entry)
    if finalized is not None:
        entries.append(finalized)

    for index, entry in enumerate(entries):
        entry["order"] = index

    try:
        source_path = str(path.relative_to(root))
    except ValueError:
        source_path = str(path)

    return {
        "schema_version": 1,
        "generated_by": "tools/build_agent_index.py",
        "source": source_path,
        "entry_count": len(entries),
        "entries": entries,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def build_indexes() -> None:
    kernel_payload = parse_catalog(KERNEL_CATALOG, kind="kernel")
    tool_payload = parse_catalog(TOOL_CATALOG, kind="tool")
    write_json(KERNEL_INDEX, kernel_payload)
    write_json(TOOL_INDEX, tool_payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build machine-readable agent indexes from markdown catalogs.")
    parser.parse_args()
    build_indexes()


if __name__ == "__main__":
    main()
