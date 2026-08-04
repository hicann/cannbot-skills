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
"""Write or verify a SHA-256 sidecar for a handoff artifact.

The digest intentionally lives outside the artifact it describes.  Embedding a
whole-file digest back into that same file would make the value impossible to
stabilize.  The JSON sidecar is portable and can be verified without trusting
the handoff text that points to it.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path


SCHEMA = "ge-fusion-pass-handoff-digest/v1"
OUTPUT_LOGGER = logging.getLogger(f"{__name__}.stdout")
OUTPUT_LOGGER.setLevel(logging.INFO)
OUTPUT_LOGGER.propagate = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_record(input_path: Path, artifact: str) -> dict:
    return {
        "schema": SCHEMA,
        "algorithm": "sha256",
        "artifact": artifact,
        "sha256": sha256_file(input_path),
        "size_bytes": input_path.stat().st_size,
    }


def load_record(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read digest sidecar {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"digest sidecar must contain a JSON object: {path}")
    return value


def write_record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def emit_json(value: dict) -> None:
    """Write one machine-readable JSON record to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    OUTPUT_LOGGER.handlers = [handler]
    OUTPUT_LOGGER.info("%s", json.dumps(value, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="要摘要的交接文件")
    parser.add_argument("--out", required=True, help="摘要 sidecar JSON 路径")
    parser.add_argument("--artifact", help="sidecar 中记录的稳定 artifact 标识；默认输入文件名")
    parser.add_argument("--check", action="store_true", help="只校验现有 sidecar，不写文件")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.out)
    if not input_path.is_file():
        raise SystemExit(f"handoff artifact does not exist: {input_path}")
    artifact = args.artifact or input_path.name
    record = expected_record(input_path, artifact)

    if args.check:
        try:
            observed = load_record(output_path)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        mismatches = [key for key, value in record.items() if observed.get(key) != value]
        if mismatches:
            emit_json({
                "status": "FAILED",
                "sidecar": str(output_path),
                "mismatches": mismatches,
            })
            return 1
        emit_json({
            "status": "PASSED",
            "sidecar": str(output_path),
            "sha256": record["sha256"],
        })
        return 0

    write_record(output_path, record)
    emit_json({
        "status": "PASSED",
        "sidecar": str(output_path),
        "sha256": record["sha256"],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
