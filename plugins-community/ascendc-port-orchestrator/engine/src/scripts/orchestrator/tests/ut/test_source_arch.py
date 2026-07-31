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

import sys
import json
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[2]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from source_arch import (  # noqa: E402
    detect_source_arch,
    load_port_a3_build_source,
    record_port_a3_build_source,
    stage_source_tree,
    verify_source_stage,
)
from run_kw_graybox import _copy_arch22_tree  # noqa: E402


def _source(
    root: Path,
    relative: str,
    content: str = "class Source { void Process() {} };\n",
) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_explicit_arch22_is_supported(tmp_path):
    _source(tmp_path, "op_kernel/arch22/op.h", "class Op { void Process() {} };\n")
    result = detect_source_arch(tmp_path)
    assert result.arch == "arch22"
    assert result.supported is True
    assert result.method == "explicit_source_dir"


def test_arch22_wins_when_target_directory_is_also_present(tmp_path):
    _source(tmp_path, "op_kernel/arch35/op.h")
    _source(tmp_path, "op_kernel/arch22/op.h")
    result = detect_source_arch(tmp_path)
    assert result.arch == "arch22"
    assert result.supported is True


def test_real_top_level_algorithm_is_supported(tmp_path):
    _source(
        tmp_path,
        "op_kernel/op.h",
        "class Op { __aicore__ void Process() { DataCopy(dst, src, 1); } };\n",
    )
    _source(tmp_path, "op_kernel/arch35/op.h")
    result = detect_source_arch(tmp_path)
    assert result.arch == "arch22"
    assert result.method == "ops_nn_top_level_algorithm"


def test_official_target_shell_shape_is_rejected(tmp_path):
    _source(
        tmp_path,
        "op_kernel/celu.cpp",
        '#include "arch35/celu.h"\nextern "C" void celu() { Celu().Process(); }\n',
    )
    _source(tmp_path, "op_kernel/arch35/celu.h", "class Celu { void Process(); };\n")
    result = detect_source_arch(tmp_path)
    assert result.arch is None
    assert result.supported is False
    assert result.method == "target_dispatch_only"


def test_arch35_only_is_rejected(tmp_path):
    _source(tmp_path, "op_kernel/arch35/op.h")
    result = detect_source_arch(tmp_path)
    assert result.arch is None
    assert result.supported is False
    assert result.method == "unknown"
    assert not result.analyzed_paths


def test_empty_kernel_is_unknown(tmp_path):
    (tmp_path / "op_kernel").mkdir()
    result = detect_source_arch(tmp_path)
    assert result.arch is None
    assert result.supported is False
    assert result.method == "unknown"


def test_whitespace_only_arch22_files_are_not_source_evidence(tmp_path):
    _source(tmp_path, "op_kernel/arch22/empty.h", " \t\n")
    _source(tmp_path, "op_kernel/arch22/CMakeLists.txt", "\n\n")
    result = detect_source_arch(tmp_path)
    assert result.arch is None
    assert result.supported is False
    assert result.method == "unknown"


def test_empty_arch22_does_not_override_nonempty_arch35_target(tmp_path):
    _source(tmp_path, "op_kernel/arch22/empty.h", "\n")
    _source(tmp_path, "op_kernel/arch35/op.h", "class Target {};\n")
    result = detect_source_arch(tmp_path)
    assert result.arch is None
    assert result.supported is False
    assert result.method == "unknown"


def test_target_tree_is_never_opened_or_reported(tmp_path, monkeypatch):
    source = tmp_path / "op_kernel" / "arch22" / "op.h"
    target = tmp_path / "op_kernel" / "arch35" / "op.h"
    _source(tmp_path, "op_kernel/arch22/op.h", "class Source { void Process() {} };\n")
    _source(tmp_path, "op_kernel/arch35/op.h", "class Target {};\n")
    original = Path.read_text

    def _read_text(path: Path, *args, **kwargs):
        if path == target:
            raise AssertionError("target implementation must never be opened")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _read_text)
    result = detect_source_arch(tmp_path)
    assert result.supported is True
    assert result.evidence == (str(source.relative_to(tmp_path)),)
    assert all("arch35" not in path for path in result.analyzed_paths)


def test_graybox_copy_prunes_target_before_traversal(tmp_path, monkeypatch):
    source_root = tmp_path / "input" / "op_kernel"
    target_dir = source_root / "arch35"
    _source(tmp_path / "input", "op_kernel/arch22/op.h", "// arch22\n")
    _source(tmp_path / "input", "op_kernel/arch35/op.h", "// target\n")
    destination = tmp_path / "copied"
    original = Path.iterdir

    def _iterdir(path: Path):
        if path == target_dir:
            raise AssertionError("target implementation directory was traversed")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", _iterdir)
    copied = _copy_arch22_tree(source_root, destination)
    assert copied == ["arch22/op.h"]
    assert (destination / "arch22" / "op.h").read_text() == "// arch22\n"
    assert not (destination / "arch35").exists()


def test_unknown_nested_layout_is_rejected(tmp_path):
    _source(tmp_path, "op_kernel/impl/op.txt", "unknown")
    result = detect_source_arch(tmp_path)
    assert result.arch is None
    assert result.supported is False


def test_conflicting_explicit_source_directories_fail_closed(tmp_path):
    _source(tmp_path, "op_kernel/arch22/op.h")
    _source(tmp_path, "op_kernel/arch21/op.h")
    result = detect_source_arch(tmp_path)
    assert result.arch is None
    assert result.method == "conflicting_source_dirs"
    assert result.supported is False


def test_marker_layout_and_evidence_are_deterministic(tmp_path):
    _source(tmp_path, "op_kernel/impl/dav_c220/z.h")
    _source(tmp_path, "op_kernel/impl/dav_c220/a.cpp")
    result = detect_source_arch(tmp_path)
    assert result.arch == "arch22"
    assert result.method == "source_marker"
    assert list(result.evidence) == sorted(result.evidence)
    assert list(result.analyzed_paths) == sorted(result.analyzed_paths)


def test_whitespace_only_marker_path_is_not_source_evidence(tmp_path):
    _source(tmp_path, "op_kernel/impl/dav_c220/empty.h", " \n\t")
    result = detect_source_arch(tmp_path)
    assert result.arch is None
    assert result.supported is False
    assert result.method == "unknown"


def test_marker_with_nonempty_sibling_arch_fails_closed(tmp_path):
    _source(tmp_path, "op_kernel/impl/dav_c220/op.h", "class Source {};\n")
    _source(tmp_path, "op_kernel/arch23/op.h", "class Other {};\n")
    result = detect_source_arch(tmp_path)
    assert result.arch is None
    assert result.supported is False
    assert result.method == "conflicting_source_dirs"


def test_top_level_algorithm_with_nonempty_sibling_arch_fails_closed(tmp_path):
    _source(
        tmp_path,
        "op_kernel/op.h",
        "class Op { __aicore__ void Process() { DataCopy(dst, src, 1); } };\n",
    )
    _source(tmp_path, "op_kernel/arch21/op.h", "class Other {};\n")
    result = detect_source_arch(tmp_path)
    assert result.arch is None
    assert result.supported is False
    assert result.method == "conflicting_source_dirs"


def test_source_symlink_escape_fails_closed(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.h"
    outside.write_text("class Source { void Process() {} };\n")
    kernel = tmp_path / "op_kernel" / "arch22"
    kernel.mkdir(parents=True)
    (kernel / "op.h").symlink_to(outside)
    result = detect_source_arch(tmp_path)
    assert result.arch is None
    assert result.supported is False
    assert result.method == "source_path_escape"


def test_unreadable_candidate_fails_closed(tmp_path, monkeypatch):
    candidate = tmp_path / "op_kernel" / "arch22" / "op.h"
    _source(tmp_path, "op_kernel/arch22/op.h", "class Source {};\n")
    original = Path.read_text

    def _read_text(path: Path, *args, **kwargs):
        if path == candidate:
            raise OSError("simulated read failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _read_text)
    result = detect_source_arch(tmp_path)
    assert result.arch is None
    assert result.supported is False
    assert result.method == "unreadable_source"


def test_broken_source_symlink_fails_closed(tmp_path):
    kernel = tmp_path / "op_kernel" / "arch22"
    kernel.mkdir(parents=True)
    (kernel / "op.h").symlink_to(kernel / "missing.h")
    result = detect_source_arch(tmp_path)
    assert result.arch is None
    assert result.supported is False
    assert result.method == "unreadable_source"


def test_source_symlink_loop_fails_closed(tmp_path):
    kernel = tmp_path / "op_kernel" / "arch22"
    kernel.mkdir(parents=True)
    loop = kernel / "op.h"
    loop.symlink_to(loop)
    result = detect_source_arch(tmp_path)
    assert result.arch is None
    assert result.supported is False
    assert result.method == "unreadable_source"


def test_source_root_symlink_loop_fails_closed(tmp_path):
    loop = tmp_path / "source"
    loop.symlink_to(loop)
    result = detect_source_arch(loop)
    assert result.arch is None
    assert result.supported is False
    assert result.method == "invalid_source_root"


def test_comment_and_literal_markers_are_not_architecture_evidence(tmp_path):
    _source(
        tmp_path,
        "op_kernel/op.cpp",
        '// arch22 DataCopy(dst, src, 1)\nconst char *note = "dav_c220";\n',
    )
    result = detect_source_arch(tmp_path)
    assert result.supported is False
    assert result.method == "unknown"


def test_comment_only_explicit_arch22_is_rejected(tmp_path):
    _source(tmp_path, "op_kernel/arch22/op.h", "// class Fake { void Process(); };\n")
    result = detect_source_arch(tmp_path)
    assert result.supported is False
    assert result.method == "unknown"


def test_target_sibling_is_pruned_before_filesystem_predicate(tmp_path, monkeypatch):
    target = tmp_path / "op_kernel" / "arch35"
    _source(tmp_path, "op_kernel/arch22/op.h")
    _source(tmp_path, "op_kernel/arch35/op.h")
    original_is_dir = Path.is_dir

    def guarded_is_dir(path):
        if path == target:
            raise AssertionError("target directory was stat'ed")
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", guarded_is_dir)
    assert detect_source_arch(tmp_path).supported is True


def _stage_state(stage):
    manifest = json.loads(stage.manifest.read_text())
    return {
        "schema_version": 2,
        "op": manifest["op"],
        "opgen_mode": "port_a3_to_a5",
        "port_a3_source": str(stage.root),
        "source_stage_manifest": str(stage.manifest),
        "source_stage_digest": stage.digest,
        "graybox_sandbox": True,
        "graybox_arch22_dir": str(stage.root),
    }


def test_source_stage_excludes_target_and_binds_hashes(tmp_path, monkeypatch):
    source = tmp_path / "customer-op"
    workspace = tmp_path / "workspace" / "op"
    target = source / "op_kernel" / "arch35"
    _source(source, "op_kernel/arch22/op.h")
    _source(source, "op_kernel/arch35/answer.h")
    original_is_symlink = Path.is_symlink

    def guarded_is_symlink(path):
        if path == target:
            raise AssertionError("target directory predicate was evaluated")
        return original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", guarded_is_symlink)
    stage = stage_source_tree(source, workspace)
    state = _stage_state(stage)
    (workspace / ".opgen_state.json").write_text(json.dumps(state))

    valid, reason, manifest = verify_source_stage(workspace)
    assert valid is True, reason
    assert not (stage.root / "op_kernel" / "arch35").exists()
    assert str(source) not in json.dumps(manifest)
    assert manifest["op"] == "customer-op"
    assert manifest["tree_sha256"] == stage.digest

    (stage.root / "op_kernel" / "arch22" / "op.h").write_text("tampered\n")
    valid, reason, _manifest = verify_source_stage(workspace)
    assert valid is False
    assert "mismatch" in reason


def test_source_stage_rejects_manifest_op_state_mismatch(tmp_path):
    source = tmp_path / "heaviside"
    workspace = tmp_path / "workspace" / "heaviside"
    _source(source, "op_kernel/arch22/heaviside.h")
    stage = stage_source_tree(source, workspace)
    state = _stage_state(stage)
    state["op"] = "different_op"

    valid, reason, _manifest = verify_source_stage(workspace, state)

    assert valid is False
    assert reason == "source-stage op binding mismatch"


def test_source_stage_rejects_unsafe_original_op_name(tmp_path):
    source = tmp_path / "unsafe op"
    _source(source, "op_kernel/arch22/op.h")

    with pytest.raises(ValueError, match="invalid op name"):
        stage_source_tree(source, tmp_path / "workspace" / "op")


def test_source_stage_rejects_manifest_symlink(tmp_path):
    source = tmp_path / "heaviside"
    workspace = tmp_path / "workspace" / "heaviside"
    _source(source, "op_kernel/arch22/heaviside.h")
    stage = stage_source_tree(source, workspace)
    state = _stage_state(stage)
    saved_manifest = workspace / "manifest.saved.json"
    stage.manifest.rename(saved_manifest)
    stage.manifest.symlink_to(saved_manifest)

    valid, reason, _manifest = verify_source_stage(workspace, state)

    assert valid is False
    assert reason == "source-stage manifest is missing, non-regular, or a symlink"


def test_source_stage_rejects_symlink_input(tmp_path):
    source = tmp_path / "customer-op"
    _source(source, "op_kernel/arch22/op.h")
    link = source / "docs" / "linked.md"
    link.parent.mkdir()
    link.symlink_to(source / "op_kernel" / "arch22" / "op.h")
    with pytest.raises(ValueError, match="symlink"):
        stage_source_tree(source, tmp_path / "workspace" / "op")


def test_private_build_source_binding_survives_outside_worker_workspace(tmp_path):
    source = tmp_path / "ops-nn" / "heaviside"
    _source(source, "op_kernel/arch22/heaviside.h")
    (source / "op_host").mkdir()
    (source / ".source_stage_manifest.json").write_text("stale input manifest\n")
    workspace = tmp_path / "engine" / "workspace" / "heaviside"
    stage = stage_source_tree(source, workspace)

    record = record_port_a3_build_source(
        workspace,
        source,
        source_stage_digest=stage.digest,
    )

    assert record.parent == workspace.parent / ".port_a3_build_sources"
    assert workspace not in record.parents
    assert load_port_a3_build_source(
        workspace,
        source_stage_digest=stage.digest,
    ) == source.resolve()


def test_private_build_source_binding_rejects_another_stage_digest(tmp_path):
    source = tmp_path / "ops-nn" / "heaviside"
    _source(source, "op_kernel/arch22/heaviside.h")
    (source / "op_host").mkdir()
    workspace = tmp_path / "engine" / "workspace" / "heaviside"
    stage = stage_source_tree(source, workspace)
    record_port_a3_build_source(
        workspace,
        source,
        source_stage_digest=stage.digest,
    )

    with pytest.raises(ValueError, match="stage digest mismatch"):
        load_port_a3_build_source(
            workspace,
            source_stage_digest="0" * 64,
        )


def test_private_build_source_binding_rejects_insecure_existing_registry(tmp_path):
    source = tmp_path / "ops-nn" / "heaviside"
    _source(source, "op_kernel/arch22/heaviside.h")
    (source / "op_host").mkdir()
    workspace = tmp_path / "engine" / "workspace" / "heaviside"
    stage = stage_source_tree(source, workspace)
    registry = workspace.parent / ".port_a3_build_sources"
    registry.mkdir(mode=0o755)
    registry.chmod(0o755)

    with pytest.raises(ValueError, match="permissions must be 0700"):
        record_port_a3_build_source(
            workspace,
            source,
            source_stage_digest=stage.digest,
        )


def test_private_build_source_binding_rejects_insecure_record_mode(tmp_path):
    source = tmp_path / "ops-nn" / "heaviside"
    _source(source, "op_kernel/arch22/heaviside.h")
    (source / "op_host").mkdir()
    workspace = tmp_path / "engine" / "workspace" / "heaviside"
    stage = stage_source_tree(source, workspace)
    record = record_port_a3_build_source(
        workspace,
        source,
        source_stage_digest=stage.digest,
    )
    record.chmod(0o644)

    with pytest.raises(ValueError, match="permissions must be 0600"):
        load_port_a3_build_source(
            workspace,
            source_stage_digest=stage.digest,
        )
    with pytest.raises(ValueError, match="permissions must be 0600"):
        record_port_a3_build_source(
            workspace,
            source,
            source_stage_digest=stage.digest,
        )


def test_private_build_source_binding_rejects_source_mutation_before_resume(tmp_path):
    source = tmp_path / "ops-nn" / "heaviside"
    kernel = source / "op_kernel" / "arch22" / "heaviside.h"
    _source(source, "op_kernel/arch22/heaviside.h")
    (source / "op_host").mkdir()
    workspace = tmp_path / "engine" / "workspace" / "heaviside"
    stage = stage_source_tree(source, workspace)
    record_port_a3_build_source(
        workspace,
        source,
        source_stage_digest=stage.digest,
    )
    kernel.write_text("class Changed { void Process() {} };\n")

    with pytest.raises(ValueError, match="no longer matches"):
        load_port_a3_build_source(
            workspace,
            source_stage_digest=stage.digest,
        )
