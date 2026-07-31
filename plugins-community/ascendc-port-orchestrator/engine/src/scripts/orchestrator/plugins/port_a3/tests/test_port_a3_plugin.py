# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Unit tests for the arch22-to-arch35 migration plugin."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent.parent.parent
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

from plugins.port_a3 import PortA3Plugin  # noqa: E402


@pytest.fixture
def plugin() -> PortA3Plugin:
    return PortA3Plugin()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compiled_provenance(workspace: Path) -> dict:
    source = workspace / "kernel" / "foo.cpp"
    deployed = workspace / "build" / "deploy" / "foo.cpp"
    object_file = workspace / "build" / "foo.cpp.o"
    shared_lib = workspace / "build" / "libfoo.so"
    for path in (source, deployed, object_file, shared_lib):
        path.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("void foo() {}\n")
    deployed.write_bytes(source.read_bytes())
    object_file.write_bytes(b"own-build-object")
    shared_lib.write_bytes(b"own-build-shared-library")
    return {
        "source": str(source.relative_to(workspace)),
        "deployed_source": str(deployed.relative_to(workspace)),
        "object": str(object_file.relative_to(workspace)),
        "shared_lib": str(shared_lib.relative_to(workspace)),
        "workspace_source_sha256": _sha256(source),
        "deploy_source_sha256": _sha256(deployed),
        "built_from_source_sha256": _sha256(source),
        "object_sha256": _sha256(object_file),
        "shared_lib_sha256": _sha256(shared_lib),
    }


def _passing_verification(workspace: Path) -> dict:
    return {
        "precision": {"status": "PASS"},
        "build_evidence": {
            "compiled_provenance": _compiled_provenance(workspace),
        },
    }


def test_identity(plugin):
    assert plugin.name == "port_a3_to_a5"
    assert plugin.cli_flag == "--port-a3"


def test_detect_requires_explicit_migration_state(tmp_path, plugin):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "run_a5_verify.py").write_text("# not authoritative\n")
    (workspace / "op_kernel" / "arch35").mkdir(parents=True)
    assert plugin.detect(workspace) is False

    (workspace / ".opgen_state.json").write_text(
        json.dumps({"opgen_mode": "port_a3_to_a5"})
    )
    assert plugin.detect(workspace) is True


def test_detect_other_or_unreadable_state_fails_closed(tmp_path, plugin):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = workspace / ".opgen_state.json"
    state.write_text(json.dumps({"opgen_mode": "backward"}))
    assert plugin.detect(workspace) is False
    state.write_text("not-json")
    assert plugin.detect(workspace) is False


def test_verify_files_include_canonical_entries(plugin):
    assert set(plugin.verify_files()) >= {
        "run_a5_verify.py",
        "pass_a_runner.py",
        "run_pass_b.py",
    }


def test_forbidden_patterns_cover_dispatch_and_cpu_reference(plugin):
    descriptions = [description for _, description in plugin.forbidden_patterns()]
    assert any("F.<op>" in item for item in descriptions)
    assert any("foreach" in item.lower() for item in descriptions)
    assert any("CPU tensor.<op>()" in item for item in descriptions)


def test_binary_provenance_non_pass_is_not_applicable(tmp_path, plugin):
    assert plugin.check_binary_provenance(
        tmp_path,
        {"precision": {"status": "PARTIAL"}},
    ) is None


def test_binary_provenance_requires_own_build_lineage(tmp_path, plugin):
    result = plugin.check_binary_provenance(
        tmp_path,
        {"precision": {"status": "PASS"}, "build_evidence": {}},
    )
    assert result is not None
    assert "compiled_provenance" in result
    assert "installed CANN artifacts are not admissible" in result


def test_binary_provenance_accepts_current_workspace_sha256_chain(tmp_path, plugin):
    assert plugin.check_binary_provenance(
        tmp_path,
        _passing_verification(tmp_path),
    ) is None


def test_binary_provenance_rejects_tampered_source(tmp_path, plugin):
    verification = _passing_verification(tmp_path)
    (tmp_path / "kernel" / "foo.cpp").write_text("void tampered() {}\n")
    result = plugin.check_binary_provenance(tmp_path, verification)
    assert result is not None
    assert "workspace_source_sha256" in result


def test_binary_provenance_rejects_source_deploy_lineage_mismatch(tmp_path, plugin):
    verification = _passing_verification(tmp_path)
    compiled = verification["build_evidence"]["compiled_provenance"]
    compiled["built_from_source_sha256"] = "0" * 64
    result = plugin.check_binary_provenance(tmp_path, verification)
    assert result is not None
    assert "source/deploy/built-from" in result


def test_binary_provenance_rejects_non_string_digest_without_crashing(
    tmp_path, plugin
):
    verification = _passing_verification(tmp_path)
    compiled = verification["build_evidence"]["compiled_provenance"]
    compiled["workspace_source_sha256"] = 0
    result = plugin.check_binary_provenance(tmp_path, verification)
    assert result is not None
    assert "workspace_source_sha256 is not a 64-hex SHA256" in result


def test_binary_provenance_rejects_path_escape(tmp_path, plugin):
    verification = _passing_verification(tmp_path)
    outside = tmp_path.parent / "outside.cpp"
    outside.write_text("void outside() {}\n")
    compiled = verification["build_evidence"]["compiled_provenance"]
    compiled["source"] = str(outside)
    compiled["workspace_source_sha256"] = _sha256(outside)
    compiled["built_from_source_sha256"] = _sha256(outside)
    compiled["deploy_source_sha256"] = _sha256(outside)
    result = plugin.check_binary_provenance(tmp_path, verification)
    assert result is not None
    assert "workspace file" in result


def test_binary_provenance_rejects_installed_tree_substitution(tmp_path, plugin):
    verification = {
        "precision": {"status": "PASS"},
        "build_evidence": {
            "legacy_workspace_hashes": ["foo.cpp: deadbeef"],
            "installed_target_hashes": ["/target/tree/foo.cpp: deadbeef"],
        },
    }
    result = plugin.check_binary_provenance(tmp_path, verification)
    assert result is not None
    assert "compiled_provenance" in result


def test_snake_to_pascal_conversion(plugin):
    assert getattr(plugin, '_snake_to_pascal')("top_k_top_p_sample") == "TopKTopPSample"


def test_archive_layout_mapping(plugin, tmp_path):
    result = plugin.archive_layout_mapping(tmp_path)
    assert result["kernel/arch35/"] == "op_kernel/arch35/"
    assert result["kernel/"] == "op_kernel/"
    assert result["op_host/"] == "op_host/"


def test_archive_target_resolution(plugin):
    assert plugin.archive_project_subdir() == "a3_to_a5_port"
    assert plugin.resolve_archive_target(
        "kernel/arch35/foo.h", "foo"
    ) == "op_kernel/arch35/foo.h"
    assert plugin.resolve_archive_target(
        "kernel/foo_apt.cpp", "foo"
    ) == "op_kernel/foo_apt.cpp"
    assert plugin.resolve_archive_target(
        "foo_a5_migration_plan.md", "foo"
    ) == "docs/foo_a5_migration_plan.md"
    assert plugin.resolve_archive_target(
        "verification.json", "foo"
    ) == "verification.json"


def test_neutral_phase_hooks(plugin):
    assert plugin.kw_brief_phase_a() is None
    assert plugin.kw_brief_phase_d() is None
    assert plugin.kb_subdirs() == ["."]
