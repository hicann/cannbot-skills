# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression tests for project_metadata module (DEBT-NEW 2026-05-15).

User directive: "你需要在output的项目目录里记录项目信息，用于识别使用的模式。
以及其他选哟记录的信息" — output project dirs must declare their op-gen
mode + reference baseline + source info so safety net can apply mode-
appropriate checks instead of guessing from file shape.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_metadata import (
    SCHEMA_VERSION,
    PROJECT_METADATA_FILENAME,
    VALID_OPGEN_MODES,
    VALID_REFERENCE_BASELINES,
    read_project_metadata,
    write_project_metadata,
    detect_mode_for_workspace,
)


def test_schema_version_is_int():
    assert isinstance(SCHEMA_VERSION, int)
    assert SCHEMA_VERSION >= 1


def test_valid_modes_includes_all_known():
    """Every op-gen mode in production must appear in VALID_OPGEN_MODES."""
    assert VALID_OPGEN_MODES == {"port_a3_to_a5", "backward"}


def test_write_then_read_roundtrip(tmp_path):
    project = tmp_path / "proj"
    fp = write_project_metadata(
        project,
        opgen_mode="port_a3_to_a5",
        source_type="cann_ops_nn",
        source_path="/path/to/ops-nn",
        reference_baseline="a3_cann",
        target_chip="Ascend950PR",
    )
    assert fp.is_file()
    meta = read_project_metadata(project)
    assert meta is not None
    assert meta["opgen_mode"] == "port_a3_to_a5"
    assert meta["reference_baseline"] == "a3_cann"
    assert meta["schema_version"] == SCHEMA_VERSION


def test_write_rejects_invalid_mode(tmp_path):
    project = tmp_path / "proj"
    try:
        write_project_metadata(
            project,
            opgen_mode="bogus_mode",
            source_type="x",
            source_path="x",
            reference_baseline="a3_cann",
            target_chip="x",
        )
        assert False, "should have raised"
    except ValueError as e:
        assert "opgen_mode" in str(e)


def test_write_rejects_invalid_baseline(tmp_path):
    project = tmp_path / "proj"
    try:
        write_project_metadata(
            project,
            opgen_mode="backward",
            source_type="x",
            source_path="x",
            reference_baseline="bogus_baseline",
            target_chip="x",
        )
        assert False, "should have raised"
    except ValueError as e:
        assert "reference_baseline" in str(e)


def test_write_refuses_overwrite_by_default(tmp_path):
    project = tmp_path / "proj"
    write_project_metadata(
        project, opgen_mode="backward", source_type="x", source_path="x",
        reference_baseline="cpu_fp64_autograd", target_chip="x",
    )
    try:
        write_project_metadata(
            project, opgen_mode="backward", source_type="x", source_path="x",
            reference_baseline="cpu_fp64_autograd", target_chip="x",
        )
        assert False, "should have raised"
    except ValueError as e:
        assert "already exists" in str(e)


def test_read_returns_none_when_missing(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    assert read_project_metadata(project) is None


def test_detect_mode_for_archived_workspace(tmp_path):
    """Archive path: output/<project>/src/kernels/<op>/ → reads project meta."""
    project = tmp_path / "myproj"
    write_project_metadata(
        project,
        opgen_mode="port_a3_to_a5",
        source_type="cann_ops_nn",
        source_path="/x",
        reference_baseline="a3_cann",
        target_chip="Ascend950PR",
    )
    op_ws = project / "src" / "kernels" / "my_op"
    op_ws.mkdir(parents=True)
    assert detect_mode_for_workspace(op_ws) == "port_a3_to_a5"


def test_detect_mode_falls_back_to_opgen_state(tmp_path):
    """Non-archive workspace: reads .opgen_state.json."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text(
        json.dumps({"op": "x", "opgen_mode": "backward"})
    )
    assert detect_mode_for_workspace(ws) == "backward"


def test_detect_mode_returns_none_when_no_signal(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    assert detect_mode_for_workspace(ws) is None
