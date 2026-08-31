# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Unit tests for byte-preserving old-format NPUKernelBench staging."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from npubench import npubench_inputs as inputs


def _task_tree(tmp_path: Path) -> tuple[Path, Path, bytes, bytes]:
    root = tmp_path / "npu_benchmark"
    level = root / "level1"
    level.mkdir(parents=True)
    task = level / "3_Add.py"
    task_bytes = (
        b"from pathlib import Path\n"
        b"def get_input_groups():\n"
        b"    return Path(__file__).with_suffix('.json').read_bytes()\n"
    )
    # Old NPUKernelBench commonly stores JSONL in a .json sidecar.
    sidecar_bytes = b'{"case": 0, "shape": [2, 3]}\n{"case": 1, "shape": [4]}\n'
    task.write_bytes(task_bytes)
    task.with_suffix(".json").write_bytes(sidecar_bytes)
    (level / "helper.py").write_bytes(b"VALUE = 7\n")
    return root, task, task_bytes, sidecar_bytes


def test_stage_preserves_old_npubench_bytes_paths_and_jsonl_in_json(tmp_path):
    root, task, task_bytes, sidecar_bytes = _task_tree(tmp_path)
    workspace = tmp_path / "workspace"

    stage = inputs.stage_npubench_inputs(
        workspace, npubench_task=task, npubench_root=root
    )

    assert stage.task_relative_path == "level1/3_Add.py"
    assert stage.sidecar_relative_path == "level1/3_Add.json"
    assert stage.sidecar_encoding == "jsonl"
    assert stage.task_path.read_bytes() == task_bytes
    assert stage.sidecar_path.read_bytes() == sidecar_bytes
    assert (stage.root / "level1" / "helper.py").read_bytes() == b"VALUE = 7\n"
    assert stage.root == workspace / "reference_inputs" / "npubench" / stage.bundle_sha256
    assert stage.state_block()["bundle_manifest_path"] == (
        f"reference_inputs/npubench/{stage.bundle_sha256}/bundle_manifest.json"
    )

    ok, reason, manifest = inputs.verify_npubench_stage(workspace, stage.state_block())
    assert ok, reason
    assert manifest["task_relative_path"] == "level1/3_Add.py"
    assert manifest["sidecar_encoding"] == "jsonl"


def test_bundled_native_example_is_a_stageable_old_format_pair(tmp_path):
    """Keep the documented native tutorial fixture structurally valid without Torch."""
    plugin_root = Path(__file__).resolve().parents[6]
    task = plugin_root / "examples" / "npukernelbench-native" / "level1" / "example_add.py"
    root = task.parent.parent

    args = inputs.validate_cli_npubench_args(task, root)
    assert args is not None
    assert args.task_relative_path.as_posix() == "level1/example_add.py"
    assert args.sidecar_relative_path.as_posix() == "level1/example_add.json"
    assert args.sidecar_encoding == "jsonl"

    stage = inputs.stage_npubench_inputs(
        tmp_path / "workspace", npubench_task=task, npubench_root=root
    )
    assert stage.task_path.read_bytes() == task.read_bytes()
    assert stage.sidecar_path.read_bytes() == task.with_suffix(".json").read_bytes()


def test_stage_rejects_both_same_stem_sidecars(tmp_path):
    root, task, _task_bytes, _sidecar_bytes = _task_tree(tmp_path)
    task.with_suffix(".jsonl").write_text('{"case": 0}\n')

    with pytest.raises(inputs.NpubenchInputError, match="both .json and .jsonl"):
        inputs.validate_cli_npubench_args(task, root)


def test_stage_rejects_task_outside_explicit_root(tmp_path):
    root, task, _task_bytes, _sidecar_bytes = _task_tree(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(inputs.NpubenchInputError, match="contained"):
        inputs.validate_cli_npubench_args(task, outside)


def test_stage_rejects_symlink_in_closure(tmp_path):
    root, task, _task_bytes, _sidecar_bytes = _task_tree(tmp_path)
    target = tmp_path / "outside_helper.py"
    target.write_text("VALUE = 9\n")
    try:
        (root / "level1" / "linked.py").symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable on this filesystem: {exc}")

    with pytest.raises(inputs.NpubenchInputError, match="cannot contain symlink"):
        inputs.stage_npubench_inputs(
            tmp_path / "workspace", npubench_task=task, npubench_root=root
        )


def test_stage_does_not_traverse_reference_inputs_symlink(tmp_path):
    root, task, _task_bytes, _sidecar_bytes = _task_tree(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (workspace / "reference_inputs").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable on this filesystem: {exc}")

    with pytest.raises(inputs.NpubenchInputError, match="non-symlink directory"):
        inputs.stage_npubench_inputs(
            workspace, npubench_task=task, npubench_root=root
        )

    assert not (outside / "npubench").exists()


def test_stage_verification_rejects_mutated_published_input(tmp_path):
    root, task, _task_bytes, _sidecar_bytes = _task_tree(tmp_path)
    workspace = tmp_path / "workspace"
    stage = inputs.stage_npubench_inputs(
        workspace, npubench_task=task, npubench_root=root
    )

    os.chmod(stage.task_path, 0o600)
    stage.task_path.write_bytes(b"tampered\n")
    ok, reason, _manifest = inputs.verify_npubench_stage(workspace, stage.state_block())
    assert not ok
    assert "read-only" in reason or "digest" in reason or "inventory" in reason


def test_bind_preserves_only_the_same_immutable_npubench_bundle(tmp_path):
    root, task, _task_bytes, _sidecar_bytes = _task_tree(tmp_path)
    stage = inputs.stage_npubench_inputs(
        tmp_path / "workspace", npubench_task=task, npubench_root=root
    )
    state: dict = {"opgen_mode": "port_a3_to_a5"}
    inputs.bind_npubench_state(state, stage)
    assert state["reference"] == stage.state_block()
    inputs.bind_npubench_state(state, stage)

    state["reference"]["task_sha256"] = "0" * 64
    with pytest.raises(inputs.NpubenchInputError, match="different NPUKernelBench bundle"):
        inputs.bind_npubench_state(state, stage)


def test_json_sidecar_may_be_one_json_value_but_not_invalid_content(tmp_path):
    root, task, _task_bytes, _sidecar_bytes = _task_tree(tmp_path)
    task.with_suffix(".json").write_text('[{"case": 0}]\n')
    args = inputs.validate_cli_npubench_args(task, root)
    assert args is not None
    assert args.sidecar_encoding == "json"

    task.with_suffix(".json").write_text("not-json\n")
    with pytest.raises(inputs.NpubenchInputError, match="neither strict JSON nor JSONL"):
        inputs.validate_cli_npubench_args(task, root)
