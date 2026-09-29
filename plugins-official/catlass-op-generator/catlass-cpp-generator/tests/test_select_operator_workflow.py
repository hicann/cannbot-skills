# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/select_operator_workflow.py"
SPEC = importlib.util.spec_from_file_location("selector", SCRIPT)
selector = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(selector)


def test_new_project_waits_for_mathematical_classification(tmp_path: Path) -> None:
    result = selector.select(tmp_path, "catlass_probe", None)
    assert result["route"] == "pending"


def test_explicit_pending_waits_for_more_mathematical_information(tmp_path: Path) -> None:
    assert selector.select(tmp_path, "catlass_probe", "pending")["route"] == "pending"


def test_intake_classified_linear_request_routes_independent_of_name(tmp_path: Path) -> None:
    result = selector.select(tmp_path, "catlass_custom_scan", "linear_attention")
    assert result["route"] == "linear_attention"
    assert "intake classified" in result["reason"]


def test_intake_classified_legacy_request_routes_without_name_heuristics(tmp_path: Path) -> None:
    assert selector.select(tmp_path, "catlass_gdn", "legacy")["route"] == "legacy"


def test_model_names_are_not_accepted_as_algorithm_classification(tmp_path: Path) -> None:
    for family in ("gdn", "KDA", "rwkv", "linear-attention"):
        result = selector.select(tmp_path, "catlass_probe", family)
        assert result["route"] == "blocked"
        assert "unsupported algorithm family" in result["reason"]


def test_existing_project_without_marker_always_stays_legacy(tmp_path: Path) -> None:
    (tmp_path / "operators/catlass_gdn").mkdir(parents=True)
    result = selector.select(tmp_path, "catlass_gdn", "linear_attention")
    assert result["route"] == "legacy"
    assert "existing project" in result["reason"]


def test_existing_valid_marker_restores_new_workflow(tmp_path: Path) -> None:
    marker = tmp_path / "operators/catlass_gdn/docs/workflow.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "workflow_id": "catlass-linear-attention-v1",
                "algorithm_family": "linear_attention",
            }
        ),
        encoding="utf-8",
    )
    assert selector.select(tmp_path, "catlass_gdn", None)["route"] == "linear_attention"


def test_invalid_or_foreign_marker_is_blocked(tmp_path: Path) -> None:
    marker = tmp_path / "operators/catlass_gdn/docs/workflow.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{", encoding="utf-8")
    assert selector.select(tmp_path, "catlass_gdn", None)["route"] == "blocked"
    marker.write_text(json.dumps({"workflow_id": "other"}), encoding="utf-8")
    assert selector.select(tmp_path, "catlass_gdn", None)["route"] == "blocked"


def test_symlink_marker_is_blocked(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps(
            {
                "workflow_id": "catlass-linear-attention-v1",
                "algorithm_family": "linear_attention",
            }
        ),
        encoding="utf-8",
    )
    marker = tmp_path / "operators/catlass_gdn/docs/workflow.json"
    marker.parent.mkdir(parents=True)
    try:
        marker.symlink_to(outside)
    except OSError:
        return
    assert selector.select(tmp_path, "catlass_gdn", None)["route"] == "blocked"


def test_unsafe_operator_name_is_blocked(tmp_path: Path) -> None:
    assert selector.select(tmp_path, "../catlass_bad", "linear_attention")["route"] == "blocked"


def test_cli_works_from_unrelated_directory_with_space_in_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace with spaces"
    unrelated = tmp_path / "unrelated cwd"
    workspace.mkdir()
    unrelated.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--workspace",
            str(workspace),
            "--operator-name",
            "catlass_probe",
            "--algorithm-family",
            "linear_attention",
        ],
        cwd=unrelated,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert json.loads(result.stdout)["route"] == "linear_attention"
