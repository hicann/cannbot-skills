# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Finalize-dispatch tests for current-build SHA256 provenance."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import finalize_pipeline as fp  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workspace(tmp_path: Path, *, mode: str = "port_a3_to_a5") -> Path:
    workspace = tmp_path / f"workspace-{mode}"
    workspace.mkdir()
    (workspace / ".opgen_state.json").write_text(
        json.dumps({"op": "test_op", "opgen_mode": mode})
    )
    return workspace


def _passing_verification(workspace: Path) -> dict:
    source = workspace / "kernel" / "test_op.cpp"
    deployed = workspace / "build" / "deploy" / "test_op.cpp"
    object_file = workspace / "build" / "test_op.cpp.o"
    shared_lib = workspace / "build" / "libtest_op.so"
    for path in (source, deployed, object_file, shared_lib):
        path.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("void test_op() {}\n")
    deployed.write_bytes(source.read_bytes())
    object_file.write_bytes(b"object")
    shared_lib.write_bytes(b"shared")
    source_digest = _sha256(source)
    return {
        "precision": {"status": "PASS"},
        "build_evidence": {
            "compiled_provenance": {
                "source": str(source.relative_to(workspace)),
                "deployed_source": str(deployed.relative_to(workspace)),
                "object": str(object_file.relative_to(workspace)),
                "shared_lib": str(shared_lib.relative_to(workspace)),
                "workspace_source_sha256": source_digest,
                "deploy_source_sha256": source_digest,
                "built_from_source_sha256": source_digest,
                "object_sha256": _sha256(object_file),
                "shared_lib_sha256": _sha256(shared_lib),
            }
        },
    }


def test_non_pass_status_skips_gate(tmp_path):
    workspace = _workspace(tmp_path)
    assert getattr(fp, '_check_binary_provenance')(
        workspace,
        {"precision": {"status": "PARTIAL"}},
    ) is None


def test_unsupported_mode_skips_gate(tmp_path):
    workspace = _workspace(tmp_path, mode="unsupported")
    assert getattr(fp, '_check_binary_provenance')(
        workspace,
        {"precision": {"status": "PASS"}, "build_evidence": {}},
    ) is None


def test_backward_mode_requires_local_binary(tmp_path):
    workspace = _workspace(tmp_path, mode="backward")
    result = getattr(fp, '_check_binary_provenance')(
        workspace,
        {"precision": {"status": "PASS"}, "build_evidence": {}},
    )
    assert result is not None and ".so" in result


def test_pass_requires_compiled_provenance(tmp_path):
    workspace = _workspace(tmp_path)
    result = getattr(fp, '_check_binary_provenance')(
        workspace,
        {"precision": {"status": "PASS"}, "build_evidence": {}},
    )
    assert result is not None
    assert "compiled_provenance" in result


def test_current_workspace_sha256_lineage_is_accepted(tmp_path):
    workspace = _workspace(tmp_path)
    verification = _passing_verification(workspace)
    assert getattr(fp, '_check_binary_provenance')(workspace, verification) is None


def test_tampered_object_is_rejected(tmp_path):
    workspace = _workspace(tmp_path)
    verification = _passing_verification(workspace)
    (workspace / "build" / "test_op.cpp.o").write_bytes(b"tampered")
    result = getattr(fp, '_check_binary_provenance')(workspace, verification)
    assert result is not None
    assert "object_sha256" in result


def test_installed_cann_hashes_are_not_admissible(tmp_path):
    workspace = _workspace(tmp_path)
    verification = {
        "precision": {"status": "PASS"},
        "build_evidence": {
            "legacy_workspace_hashes": ["test_op.cpp: deadbeef"],
            "installed_target_hashes": ["/target/tree/test_op.cpp: deadbeef"],
        },
    }
    result = getattr(fp, '_check_binary_provenance')(workspace, verification)
    assert result is not None
    assert "installed CANN artifacts are not admissible" in result


def test_gate_id_value_stable():
    assert fp.GateID.BINARY_PROVENANCE.value == "binary_provenance"
