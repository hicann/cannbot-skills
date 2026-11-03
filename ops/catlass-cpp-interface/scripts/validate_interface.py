# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Validate that the interface-stage artifact is complete enough to freeze."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


UNRESOLVED = re.compile(r"(?im)(?:^|\s)(?:TODO|TBD|待填写|待补充|待确认)(?=\s|$|[:：])")
REQUIRED_SECTIONS = ("接口签名", "支持范围", "边界行为")


def _section(text: str, title: str) -> str | None:
    match = re.search(rf"(?ms)^##\s+{re.escape(title)}\s*$\n?(.*?)(?=^##\s+|\Z)", text)
    return match.group(1) if match else None


def validate_interface(text: str) -> list[str]:
    errors: list[str] = []
    if not text.strip():
        errors.append("docs/api.md must not be empty")
    if UNRESOLVED.search(text):
        errors.append("docs/api.md contains an unresolved placeholder")
    for title in REQUIRED_SECTIONS:
        body = _section(text, title)
        if body is None:
            errors.append(f"docs/api.md missing required section: {title}")
        elif not body.strip():
            errors.append(f"docs/api.md section must not be empty: {title}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True, type=Path)
    args = parser.parse_args()

    if not args.api.is_file():
        print(f"FAIL\n- interface file not found: {args.api}")
        return 1

    try:
        text = args.api.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"FAIL\n- cannot read interface file: {exc}")
        return 1

    errors = validate_interface(text)
    if errors:
        print("FAIL")
        print("\n".join(f"- {error}" for error in errors))
        return 1

    print("PASS: interface artifact is complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
