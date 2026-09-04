# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Contract tests for the TileLang2AscendC source and candidate profile."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import _reorg_paths  # noqa: F401
import pytest

from npubench import npubench_target
import tilelang2ascendc_source as tile_source
# Bind the protected validator once instead of reaching into the module for it
# at every call site; production callers bind it the same way (see
# finalize_checks_structural) and the symbol is never monkeypatched.
from npubench.npubench_target import _validate_candidate_for_controlled_build


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "3_Add"
    _write(
        root,
        "model_new_ascendc.py",
        "class ModelNew(torch.nn.Module):\n"
        "    def forward(self, x, y):\n"
        "        output = torch.ops.npu.add(x, y)\n"
        "        return output\n",
    )
    _write(
        root,
        "kernel/register.cpp",
        '#include <pybind11/pybind11.h>\n'
        'TORCH_LIBRARY_FRAGMENT(npu, m) { m.def("add(Tensor x, Tensor y) -> Tensor"); }\n'
        'TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) { m.impl("add", &Add); }\n'
        'PYBIND11_MODULE(_add_ext, m) { m.doc() = "add extension"; }\n',
    )
    _write(
        root,
        "kernel/CMakeLists.txt",
        "cmake_minimum_required(VERSION 3.16)\n"
        "add_library(add SHARED op_host/add.cpp op_kernel/add.cpp register.cpp)\n",
    )
    _write(
        root,
        "kernel/op_host/add.cpp",
        "void Add() { EXEC_KERNEL_CMD(add_kernel); }\n",
    )
    _write(
        root,
        "kernel/op_kernel/add.cpp",
        "__global__ __aicore__ void add_kernel() { AscendC::DataCopy(); }\n",
    )
    return root


def _state(workspace: Path, stage: tile_source.Tilelang2AscendcSourceStage) -> dict:
    state = {
        "op": stage.op,
        "source_kind": tile_source.TILELANG2ASCENDC_SOURCE_KIND,
        "port_source": tile_source.tilelang2ascendc_state_block(stage),
        "source_arch": "arch35",
        "target_arch": "arch35",
        "port_a3_source": str(stage.root),
        "source_stage_manifest": str(stage.manifest),
        "source_stage_digest": stage.digest,
        "source_stage_file_count": stage.file_count,
        "graybox_source_dir": str(stage.root),
    }
    (workspace / ".opgen_state.json").write_text(json.dumps(state), encoding="utf-8")
    return state


def _authored_suffix(path: Path) -> bytes:
    """Return the authored-candidate marker appropriate for ``path``'s language."""
    if path.name == "CMakeLists.txt":
        return b"\nset(AUTHORED_CANDIDATE 1)\n"
    if path.suffix == ".py":
        return b"\n# authored candidate\n"
    return b"\nint authored_candidate_marker = 0;\n"


def _no_suffix(path: Path) -> bytes:
    """Return an empty suffix, i.e. copy the stable file byte-for-byte."""
    del path
    return b""


def _comment_only_suffix(path: Path) -> bytes:
    """Return a comment-only marker that must not count as a semantic change."""
    if path.suffix in {".py", ""} or path.name == "CMakeLists.txt":
        return b"\n# candidate metadata\n"
    return b"\n// candidate metadata\n"


def _populate_candidate(
    source: Path,
    workspace: Path,
    stable_files: set[str],
    stable_suffix: Callable[[Path], bytes] = _no_suffix,
    renames: Mapping[str, str] | None = None,
) -> None:
    """Copy ``source`` into ``workspace`` as an authored candidate tree."""
    renames = renames or {}
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        if relative in renames:
            destination, suffix = workspace / renames[relative], b""
        elif relative in stable_files:
            destination, suffix = workspace / relative, stable_suffix(path)
        else:
            destination, suffix = workspace / relative, _authored_suffix(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes() + suffix)


def test_detects_full_project_and_kernel_alias_without_pybind(tmp_path: Path) -> None:
    source = _project(tmp_path)

    detection = tile_source.detect_tilelang2ascendc_source(source)
    kernel_detection = tile_source.detect_tilelang2ascendc_source(source / "kernel")

    assert detection.supported is True
    assert detection.arch == "arch35"
    assert detection.method == "tilelang2ascendc_project_layout"
    assert "layout:model_new_ascendc+kernel" in detection.evidence
    assert "kernel/op_host/add.cpp" in detection.evidence
    assert kernel_detection == detection


def test_stages_and_verifies_immutable_tilelang_project(tmp_path: Path) -> None:
    source = _project(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    stage = tile_source.stage_tilelang2ascendc_source_tree(source, workspace)
    state = _state(workspace, stage)
    valid, reason, manifest = tile_source.verify_tilelang2ascendc_source_stage(workspace, state)

    assert valid is True, reason
    assert stage.root == workspace / ".tilelang2ascendc_source"
    assert manifest["schema"] == tile_source.TILELANG2ASCENDC_SOURCE_SCHEMA
    assert manifest["kind"] == tile_source.TILELANG2ASCENDC_SOURCE_KIND
    assert manifest["tree_sha256"] == stage.digest
    assert not (stage.root / "kernel" / "pybind11.cpp").exists()

    with pytest.raises(tile_source.Tilelang2AscendcSourceError, match="SOURCE_STAGE_EXISTS"):
        tile_source.stage_tilelang2ascendc_source_tree(source, workspace)
    assert not list(workspace.glob("...tilelang2ascendc_source.incoming-*"))


def test_tile_candidate_uses_independent_custom_op_contract(tmp_path: Path) -> None:
    source = _project(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    stage = tile_source.stage_tilelang2ascendc_source_tree(source, workspace)
    state = _state(workspace, stage)

    stable_files = {
        "model_new_ascendc.py",
        "kernel/register.cpp",
    }
    _populate_candidate(source, workspace, stable_files)

    proof = _validate_candidate_for_controlled_build(
        workspace,
        tile_source.TILELANG2ASCENDC_SOURCE_KIND,
        stage_manifest := json.loads(stage.manifest.read_text(encoding="utf-8")),
    )

    assert proof["schema"] == npubench_target.TILELANG2ASCENDC_CANDIDATE_INDEPENDENCE_SCHEMA
    assert proof["format"] == "tilelang2ascendc"
    assert proof["pybind_required"] is False
    assert proof["custom_op_calls"] == ["add"]
    assert proof["device_entries"] == ["add_kernel"]
    assert "EXEC_KERNEL_CMD" in proof["host_launch_evidence"]
    assert proof["unchanged_stable_files"] == sorted(stable_files)
    assert proof["changed_kernel_files"]
    assert stage_manifest["tree_sha256"] == stage.digest


def test_tile_candidate_allows_comment_only_changes_in_stable_project_files(tmp_path: Path) -> None:
    source = _project(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    stage = tile_source.stage_tilelang2ascendc_source_tree(source, workspace)
    stable_files = {
        "model_new_ascendc.py",
        "kernel/CMakeLists.txt",
        "kernel/register.cpp",
    }
    _populate_candidate(source, workspace, stable_files, stable_suffix=_comment_only_suffix)

    proof = _validate_candidate_for_controlled_build(
        workspace,
        tile_source.TILELANG2ASCENDC_SOURCE_KIND,
        json.loads(stage.manifest.read_text(encoding="utf-8")),
    )

    assert proof["unchanged_stable_files"] == sorted(stable_files)


def test_tile_candidate_rejects_renamed_copy_of_staged_kernel_source(tmp_path: Path) -> None:
    source = _project(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    stage = tile_source.stage_tilelang2ascendc_source_tree(source, workspace)
    stable_files = {
        "model_new_ascendc.py",
        "kernel/CMakeLists.txt",
        "kernel/register.cpp",
    }
    _populate_candidate(
        source,
        workspace,
        stable_files,
        renames={"kernel/op_host/add.cpp": "kernel/op_host/renamed.cpp"},
    )

    with pytest.raises(npubench_target.TargetTransportError, match="only changes comments/formatting"):
        _validate_candidate_for_controlled_build(
            workspace,
            tile_source.TILELANG2ASCENDC_SOURCE_KIND,
            json.loads(stage.manifest.read_text(encoding="utf-8")),
        )


def test_tile_candidate_rejects_host_stl_in_device_code(tmp_path: Path) -> None:
    source = _project(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    stage = tile_source.stage_tilelang2ascendc_source_tree(source, workspace)
    state = _state(workspace, stage)

    stable_files = {"model_new_ascendc.py", "kernel/register.cpp"}
    _populate_candidate(source, workspace, stable_files)
    device = workspace / "kernel" / "op_kernel" / "add.cpp"
    device.write_text(device.read_text(encoding="utf-8") + "\nstd::min<int>(1, 2);\n", encoding="utf-8")

    with pytest.raises(npubench_target.TargetTransportError, match="host STL"):
        _validate_candidate_for_controlled_build(
            workspace,
            tile_source.TILELANG2ASCENDC_SOURCE_KIND,
            json.loads(stage.manifest.read_text(encoding="utf-8")),
        )


def test_tile_candidate_rejects_register_without_pybind11_module(tmp_path: Path) -> None:
    source = _project(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    stage = tile_source.stage_tilelang2ascendc_source_tree(source, workspace)
    state = _state(workspace, stage)

    stable_files = {"model_new_ascendc.py", "kernel/register.cpp"}
    _populate_candidate(source, workspace, stable_files)
    register = workspace / "kernel" / "register.cpp"
    register.write_text(
        register.read_text(encoding="utf-8").replace(
            'PYBIND11_MODULE(_add_ext, m) { m.doc() = "add extension"; }\n', ""
        ),
        encoding="utf-8",
    )

    with pytest.raises(npubench_target.TargetTransportError, match="PYBIND11_MODULE"):
        _validate_candidate_for_controlled_build(
            workspace,
            tile_source.TILELANG2ASCENDC_SOURCE_KIND,
            json.loads(stage.manifest.read_text(encoding="utf-8")),
        )


def test_stage_rejects_symlink_in_tile_project(tmp_path: Path) -> None:
    source = _project(tmp_path)
    link = source / "kernel" / "op_host" / "linked.cpp"
    link.symlink_to(source / "kernel" / "op_host" / "add.cpp")

    detection = tile_source.detect_tilelang2ascendc_source(source)

    assert detection.supported is False
    assert "symlink" in detection.evidence[0].lower()
