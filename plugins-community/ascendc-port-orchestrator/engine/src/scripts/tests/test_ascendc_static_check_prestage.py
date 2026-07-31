# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ascendc_static_check as checker  # noqa: E402


def _write_manifest(workspace: Path, rel_path: str, body: bytes) -> None:
    (workspace / ".upstream_prestaged.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "staged_files": {rel_path: hashlib.sha256(body).hexdigest()},
            }
        )
    )


def test_intact_prestaged_file_is_exempt(tmp_path: Path) -> None:
    workspace = tmp_path / "op"
    kernel = workspace / "op_kernel"
    kernel.mkdir(parents=True)
    body = b"// provenance-tracked upstream target helper\n"
    staged = kernel / "helper.h"
    staged.write_bytes(body)
    _write_manifest(workspace, "op_kernel/helper.h", body)

    assert checker.collect_files(str(kernel)) == []


def test_modified_prestaged_file_is_checked(tmp_path: Path) -> None:
    workspace = tmp_path / "op"
    kernel = workspace / "op_kernel"
    kernel.mkdir(parents=True)
    original = b"// original\n"
    staged = kernel / "helper.h"
    staged.write_bytes(original)
    _write_manifest(workspace, "op_kernel/helper.h", original)
    staged.write_text("// modified after prestage\n")

    assert checker.collect_files(str(kernel)) == [str(staged)]


def test_manifest_path_escape_never_creates_an_exemption(tmp_path: Path) -> None:
    workspace = tmp_path / "op"
    kernel = workspace / "op_kernel"
    kernel.mkdir(parents=True)
    outside = tmp_path / "outside.h"
    outside.write_text("// outside\n")
    _write_manifest(workspace, "../outside.h", outside.read_bytes())

    assert str(outside) not in getattr(checker, '_load_prestaged_exempt_set')(str(kernel))
