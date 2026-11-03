# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Verify that definition.json embeds the current reference.py exactly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(source: Path, definition_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        reference = source.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"cannot read source: {source}: {exc}"]
    if not reference.strip():
        errors.append("reference.py must not be empty")
    else:
        try:
            compile(reference, str(source), "exec")
        except SyntaxError as exc:
            errors.append(
                f"reference.py has invalid Python syntax at line {exc.lineno}: {exc.msg}"
            )
    try:
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read definition JSON: {definition_path}: {exc}"]
    if not isinstance(definition, dict):
        return ["definition must be a JSON object"]
    for field, expected_type in (("name", str), ("inputs", list), ("outputs", list)):
        if field not in definition:
            errors.append(f"definition missing required field: {field}")
        elif not isinstance(definition[field], expected_type):
            errors.append(f"definition.{field} has invalid type")
    embedded = definition.get("reference")
    if not isinstance(embedded, str):
        errors.append("definition.reference must be a string")
    elif embedded != reference:
        errors.append(
            "definition.reference differs from reference.py; regenerate definition.json"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--definition", type=Path, required=True)
    args = parser.parse_args()
    errors = validate(args.source, args.definition)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("OK: definition.reference matches reference.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
