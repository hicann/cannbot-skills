# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Durable arch22-to-arch35 evidence propagation tests."""
from __future__ import annotations

import json

import a3_ref_validate
import finalize_pipeline  # noqa: F401 - initializes parent before dispatch re-export
import finalize_dispatch
from a3_ref_common import O25A3Report


def _seed_state(workspace):
    detection = {
        "arch": "arch22",
        "supported": True,
        "method": "explicit_source_dir",
        "confidence": "high",
        "evidence": ["op_kernel/arch22/op.cpp"],
        "analyzed_paths": ["op_kernel/arch22/op.cpp"],
    }
    (workspace / ".opgen_state.json").write_text(
        json.dumps(
            {
                "opgen_mode": "port_a3_to_a5",
                "source_arch": "arch22",
                "target_arch": "arch35",
                "source_arch_detection": detection,
            }
        )
    )
    return detection


def test_reference_report_carries_migration_detection(tmp_path):
    workspace = tmp_path / "op"
    workspace.mkdir()
    detection = _seed_state(workspace)

    path = getattr(a3_ref_validate, '_write_a3_reference_runnable_json')(
        workspace, O25A3Report(verdict="READY")
    )
    migration = json.loads(path.read_text())["migration"]
    assert migration == {
        "source_arch": "arch22",
        "target_arch": "arch35",
        "source_arch_detection": detection,
    }


def test_finalize_verification_carries_migration_detection(tmp_path):
    workspace = tmp_path / "op"
    workspace.mkdir()
    detection = _seed_state(workspace)
    verification = workspace / "verification.json"
    verification.write_text(json.dumps({"precision": {"status": "PASS"}}))

    getattr(finalize_dispatch, '_inject_migration_metadata')(workspace)

    payload = json.loads(verification.read_text())
    assert payload["migration"] == {
        "source_arch": "arch22",
        "target_arch": "arch35",
        "source_arch_detection": detection,
    }


def test_non_migration_state_does_not_gain_migration_metadata(tmp_path):
    workspace = tmp_path / "op"
    workspace.mkdir()
    (workspace / ".opgen_state.json").write_text(
        json.dumps({"opgen_mode": "backward"})
    )
    verification = workspace / "verification.json"
    verification.write_text("{}")

    getattr(finalize_dispatch, '_inject_migration_metadata')(workspace)

    assert "migration" not in json.loads(verification.read_text())
