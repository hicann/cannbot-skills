# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Canonical logical-op metadata for a fixed source-only migration snapshot."""
from __future__ import annotations

import json
import re
from pathlib import Path

SOURCE_STAGE_DIR = ".source_arch22"
SOURCE_STAGE_MANIFEST = ".source_stage_manifest.json"
SOURCE_STAGE_SCHEMA = "source_stage/v1"
_SAFE_OP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def is_safe_op_name(value: object) -> bool:
    """Return whether ``value`` is a safe single path component for an op."""
    return isinstance(value, str) and _SAFE_OP_NAME.fullmatch(value) is not None


def require_safe_op_name(value: object) -> str:
    """Return a validated op name, rejecting path components and empty names."""
    if not is_safe_op_name(value):
        raise ValueError(f"invalid op name: {value!r}")
    return value


def resolve_logical_op_name(op_dir: Path) -> str:
    """Resolve an op's original name when ``op_dir`` is a fixed snapshot.

    Ordinary ops retain their directory basename.  The constant snapshot name
    cannot identify the op, so fixed snapshots require a regular, schema-bound
    manifest and fail closed when that metadata is missing or invalid.
    """
    if op_dir.name != SOURCE_STAGE_DIR:
        return op_dir.name

    manifest = op_dir / SOURCE_STAGE_MANIFEST
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("source-stage manifest is missing, non-regular, or a symlink")
    try:
        payload = json.loads(manifest.read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"source-stage manifest is unreadable: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SOURCE_STAGE_SCHEMA:
        raise ValueError("unsupported source-stage manifest schema")
    return require_safe_op_name(payload.get("op"))
