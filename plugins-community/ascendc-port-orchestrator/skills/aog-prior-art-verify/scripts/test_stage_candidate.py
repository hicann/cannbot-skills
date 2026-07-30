# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for stage_candidate.stage() — Phase 2 of aog-prior-art-verify."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from scan_prior_art import scan, write_scan_result  # noqa: E402
from stage_candidate import stage, write_manifest  # noqa: E402


def _touch(p: Path, content: str = "// stub") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def test_stage_mode_a_arch35(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPGEN_PRESTAGE_ARCH35", "1")
    """Mode A: arch35/ in op's own op_kernel/ → staged to candidate."""
    port_source = tmp_path / "cann" / "ops-nn" / "norm" / "ada_layer_norm"
    workspace = tmp_path / "ws" / "ada_layer_norm"
    workspace.mkdir(parents=True)
    _touch(port_source / "op_kernel" / "arch35" / "ada_layer_norm.h", "// header")
    _touch(port_source / "op_kernel" / "arch35" / "ada_layer_norm.cpp", "// impl")
    # Pre-run scanner so the .prior_art_scan.json exists
    scan_result = scan("ada_layer_norm", port_source, workspace)
    write_scan_result(workspace, scan_result)
    rep = stage("ada_layer_norm", port_source, workspace)
    assert rep.errors == [], rep.errors
    assert "upstream_arch35" in rep.sources_staged
    assert len(rep.files_staged) == 2
    staged_root = rep.candidate_dir
    assert (staged_root / "op_kernel" / "arch35" / "ada_layer_norm.h").is_file()


def test_stage_mode_b_shared_common(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPGEN_PRESTAGE_ARCH35", "1")
    """Mode B: sibling _common dir staged under candidate with its own
    relative path preserved (so #include "../family_common/..." resolves)."""
    base = tmp_path / "cann" / "ops-nn" / "pooling"
    port_source = base / "adaptive_avg_pool3d"
    port_source.mkdir(parents=True)
    workspace = tmp_path / "ws" / "adaptive_avg_pool3d"
    workspace.mkdir(parents=True)
    common = base / "adaptive_pool3d_common" / "op_kernel" / "arch35"
    _touch(common / "adaptive_avg_pool3d_big_kernel.h")
    _touch(common / "adaptive_avg_pool3d_parall_pool.h")
    _touch(common / "adaptive_max_pool3d_helper.h")  # foreign — should not stage
    scan_result = scan("adaptive_avg_pool3d", port_source, workspace)
    write_scan_result(workspace, scan_result)
    rep = stage("adaptive_avg_pool3d", port_source, workspace)
    assert rep.errors == [], rep.errors
    assert "upstream_shared_common" in rep.sources_staged
    assert len(rep.files_staged) == 2  # foreign-op file filtered out
    staged = rep.candidate_dir / "adaptive_pool3d_common" / "op_kernel" / "arch35"
    assert (staged / "adaptive_avg_pool3d_big_kernel.h").is_file()
    assert not (staged / "adaptive_max_pool3d_helper.h").is_file()


def test_stage_apt_and_a5_config_coexist(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Apt.cpp + A5 binary.json config staged alongside arch35."""
    monkeypatch.setenv("OPGEN_PRESTAGE_ARCH35", "1")
    port_source = tmp_path / "cann" / "ops-nn" / "pooling" / "adaptive_avg_pool3d"
    workspace = tmp_path / "ws" / "adaptive_avg_pool3d"
    workspace.mkdir(parents=True)
    _touch(port_source / "op_kernel" / "arch35" / "adaptive_avg_pool3d.h")
    _touch(port_source / "op_kernel" / "adaptive_avg_pool3d_apt.cpp")
    _touch(port_source / "op_host" / "config" / "ascend950" / "adaptive_avg_pool3d_binary.json",
           '{"bin_filename": "test"}')
    _touch(port_source / "op_host" / "config" / "ascend950" / "adaptive_avg_pool3d_simplified_key.ini")
    scan_result = scan("adaptive_avg_pool3d", port_source, workspace)
    write_scan_result(workspace, scan_result)
    rep = stage("adaptive_avg_pool3d", port_source, workspace)
    assert rep.errors == [], rep.errors
    assert set(rep.sources_staged) >= {"upstream_arch35", "upstream_apt",
                                        "upstream_ascend950_config"}
    cd = rep.candidate_dir
    assert (cd / "op_kernel" / "adaptive_avg_pool3d_apt.cpp").is_file()
    assert (cd / "op_host" / "config" / "ascend950" / "adaptive_avg_pool3d_binary.json").is_file()


def test_stage_op_def_with_ascend950(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPGEN_PRESTAGE_ARCH35", "1")
    """_def.cpp with AddConfig('ascend950') gets staged; without it, warning."""
    port_source = tmp_path / "cann" / "ops-nn" / "norm" / "op_a"
    workspace = tmp_path / "ws" / "op_a"
    workspace.mkdir(parents=True)
    _touch(port_source / "op_kernel" / "arch35" / "op_a.h")
    _touch(port_source / "op_host" / "op_a_def.cpp",
           '// def\n.AddConfig("ascend910b", cfg)\n.AddConfig("ascend950", cfg2)\n')
    scan_result = scan("op_a", port_source, workspace)
    write_scan_result(workspace, scan_result)
    rep = stage("op_a", port_source, workspace)
    op_def_staged = [f for f in rep.files_staged
                      if f["source_type"] == "upstream_op_def"]
    assert len(op_def_staged) == 1
    assert op_def_staged[0]["build_rel_path"].endswith("op_host/op_a_def.cpp")


def test_stage_op_def_without_ascend950_warns(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPGEN_PRESTAGE_ARCH35", "1")
    """_def.cpp without ascend950 emits warning, NOT staged."""
    port_source = tmp_path / "cann" / "ops-nn" / "norm" / "op_b"
    workspace = tmp_path / "ws" / "op_b"
    workspace.mkdir(parents=True)
    _touch(port_source / "op_kernel" / "arch35" / "op_b.h")
    _touch(port_source / "op_host" / "op_b_def.cpp",
           '// def\n.AddConfig("ascend910b", cfg)\n')  # no ascend950 reference
    scan_result = scan("op_b", port_source, workspace)
    write_scan_result(workspace, scan_result)
    rep = stage("op_b", port_source, workspace)
    assert not any(s["type"] == "upstream_op_def" for s in scan_result["sources"])
    op_def_staged = [f for f in rep.files_staged
                      if f["source_type"] == "upstream_op_def"]
    assert op_def_staged == []


def test_stage_errors_when_scan_missing(tmp_path: Path) -> None:
    """No .prior_art_scan.json → error, no crash."""
    port_source = tmp_path / "src"
    port_source.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    rep = stage("op", port_source, workspace)
    assert rep.errors
    assert any("prior_art_scan.json" in e for e in rep.errors)


def test_stage_errors_when_has_prior_art_false(tmp_path: Path) -> None:
    """scan has has_prior_art=False → stage records error."""
    port_source = tmp_path / "src"
    port_source.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / ".prior_art_scan.json").write_text(json.dumps({
        "has_prior_art": False, "sources": []
    }))
    rep = stage("op", port_source, workspace)
    assert any("has_prior_art=False" in e for e in rep.errors)


def test_stage_idempotent_overwrites_candidate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPGEN_PRESTAGE_ARCH35", "1")
    """Re-running stage clears the candidate dir before re-populating."""
    port_source = tmp_path / "src" / "op_x"
    workspace = tmp_path / "ws" / "op_x"
    workspace.mkdir(parents=True)
    _touch(port_source / "op_kernel" / "arch35" / "op_x.h", "v1")
    scan_result = scan("op_x", port_source, workspace)
    write_scan_result(workspace, scan_result)
    stage("op_x", port_source, workspace)
    # Drop a stray file in the candidate dir to confirm cleanup
    candidate_dir = workspace / ".prior_art_candidate"
    stray = candidate_dir / "stray.txt"
    stray.write_text("zombie")
    # Run again
    rep2 = stage("op_x", port_source, workspace)
    assert rep2.errors == []
    assert not stray.exists(), "candidate dir should be cleared on re-stage"


def test_write_manifest_records_provenance(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPGEN_PRESTAGE_ARCH35", "1")
    port_source = tmp_path / "src" / "op_y"
    workspace = tmp_path / "ws" / "op_y"
    workspace.mkdir(parents=True)
    _touch(port_source / "op_kernel" / "arch35" / "op_y.h", "stub content")
    scan_result = scan("op_y", port_source, workspace)
    write_scan_result(workspace, scan_result)
    rep = stage("op_y", port_source, workspace)
    manifest_path = write_manifest(rep)
    assert manifest_path.is_file()
    data = json.loads(manifest_path.read_text())
    assert data["op"] == "op_y"
    assert data["file_count"] == 1
    assert "upstream_arch35" in data["sources_staged"]
    assert all("sha" in f and len(f["sha"]) == 64 for f in data["files"])
    assert all("build_rel_path" in f for f in data["files"])
    assert len(data["candidate_digest"]) == 64


def test_default_off_does_not_stage_target_tree(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real scan→stage run cannot rediscover an unconsulted target tree."""
    monkeypatch.delenv("OPGEN_PRESTAGE_ARCH35", raising=False)
    port_source = tmp_path / "cann" / "ops-nn" / "norm" / "op_z"
    workspace = tmp_path / "ws" / "op_z"
    workspace.mkdir(parents=True)
    _touch(port_source / "op_kernel" / "op_z.cpp", "// source entry")
    _touch(port_source / "op_kernel" / "arch35" / "op_z.h", "// target candidate")

    scan_result = scan("op_z", port_source, workspace)
    assert scan_result["consulted_a5_sources"] is False
    rep = stage("op_z", port_source, workspace, scan_result=scan_result)

    assert rep.files_staged == []
    assert any("no digest-bound candidate source" in error for error in rep.errors)
    assert not (rep.candidate_dir / "op_kernel" / "arch35" / "op_z.h").exists()


def test_stage_rejects_scan_to_copy_digest_change(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPGEN_PRESTAGE_ARCH35", "1")
    port_source = tmp_path / "cann" / "ops-nn" / "norm" / "op_t"
    workspace = tmp_path / "ws" / "op_t"
    workspace.mkdir(parents=True)
    target = port_source / "op_kernel" / "arch35" / "op_t.h"
    _touch(target, "v1")
    scan_result = scan("op_t", port_source, workspace)

    target.write_text("v2")
    rep = stage("op_t", port_source, workspace, scan_result=scan_result)

    assert rep.files_staged == []
    assert any("changed after scan" in error for error in rep.errors)


def test_target_registration_files_are_not_a_candidate_by_themselves(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPGEN_PRESTAGE_ARCH35", "1")
    port_source = tmp_path / "cann" / "ops-nn" / "norm" / "op_config_only"
    workspace = tmp_path / "ws" / "op_config_only"
    workspace.mkdir(parents=True)
    _touch(port_source / "op_kernel" / "op_config_only.cpp", "// source only")
    _touch(
        port_source / "op_host" / "config" / "ascend950" / "binary.json",
        "{}",
    )
    scan_result = scan("op_config_only", port_source, workspace)

    assert not any(source["type"] == "upstream_ascend950_config"
                   for source in scan_result["sources"])
    rep = stage(
        "op_config_only", port_source, workspace, scan_result=scan_result
    )
    assert rep.files_staged == []
    assert rep.errors


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
