# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""A.8 compile-only contract gate: request → controlled build → filtered
writeback loop, with a mock build (no CANN, no NPU).

Run: cd src/scripts/orchestrator && python3 -m pytest tests/ut/test_npubench_compile_gate.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

from npubench import npubench_compile_gate as gate  # noqa: E402


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "op"
    (ws / "kernel" / "op_kernel").mkdir(parents=True)
    (ws / "model_new_ascendc.py").write_text("# candidate entry\n")
    (ws / "kernel" / "op_kernel" / "ffn.cpp").write_text("// candidate kernel\n")
    (ws / ".opgen_state.json").write_text("{}")
    (ws / "PROGRESS.md").write_text("# progress\n")  # evaluator-owned runtime file
    return ws


def _write_request(ws: Path, request_id="req01", paths=("kernel/op_kernel/ffn.cpp",), **over):
    directory = gate.request_dir(ws)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": gate.COMPILE_REQUEST_SCHEMA,
        "request_id": over.pop("payload_id", request_id),
        "op": "2_FFN_evo",
        "paths": list(paths),
        **over,
    }
    path = directory / f"{request_id}.json"
    path.write_text(json.dumps(payload))
    return path


def _mock_build(ws, lane, *, build_attempt_id=None):
    return {
        "status": "ERROR",
        "returncode": 1,
        "failure_kind": "candidate_compile",
        "reason": "build_ascendc.py exited non-zero",
        "stdout_tail": "gmake: entering kernel/build\n",
        "stderr_tail": (
            "kernel/op_kernel/ffn.cpp:42:5: error: no matching function for call to 'Cast'\n"
            "/usr/local/Ascend/cann-9.2.0/include/experiment/runtime/rt.h:9: note: candidate\n"
            "ffn.cpp:43:9: error: static assertion failed\n"
        ),
    }


# ---------------------------------------------------------------------------
# The one required loop test: request → execute → filtered writeback
# ---------------------------------------------------------------------------
def test_compile_gate_request_execute_writeback_loop(workspace):
    request_path = _write_request(workspace)
    response = gate.execute_compile_request(workspace, 0, request_path, build_fn=_mock_build)

    assert response["schema"] == gate.COMPILE_RESPONSE_SCHEMA
    assert response["request_id"] == "req01"
    assert response["status"] == "ERROR"
    # Candidate-file diagnostic lines survive — by relative path AND basename.
    assert any("kernel/op_kernel/ffn.cpp:42" in line for line in response["diagnostics"])
    assert any("ffn.cpp:43" in line for line in response["diagnostics"])
    # CANN header lines never leak back into the workspace.
    assert not any("/usr/local/Ascend" in line for line in response["diagnostics"])

    response_file = gate.response_dir(workspace) / "req01.json"
    on_disk = json.loads(response_file.read_text())
    assert on_disk == response
    # Once answered, the request is no longer pending.
    assert gate.pending_compile_requests(workspace) == []


# ---------------------------------------------------------------------------
# Validation: fail closed on anything outside the candidate whitelist
# ---------------------------------------------------------------------------
def test_compile_gate_rejects_escape_and_evaluator_owned_paths(workspace):
    with pytest.raises(gate.CompileGateError, match="escapes"):
        gate.validate_compile_request(
            workspace, _write_request(workspace, "bad01", paths=("../outside.cpp",))
        )
    with pytest.raises(gate.CompileGateError, match="evaluator-owned"):
        gate.validate_compile_request(
            workspace, _write_request(workspace, "bad02", paths=("PROGRESS.md",))
        )
    with pytest.raises(gate.CompileGateError, match="not an existing regular file"):
        gate.validate_compile_request(
            workspace, _write_request(workspace, "bad03", paths=("kernel/missing.cpp",))
        )


def test_compile_gate_rejects_schema_and_id_mismatch(workspace):
    with pytest.raises(gate.CompileGateError, match="schema"):
        gate.validate_compile_request(
            workspace, _write_request(workspace, "bad04", schema="aog.compile_request/v0")
        )
    with pytest.raises(gate.CompileGateError, match="does not match filename"):
        gate.validate_compile_request(
            workspace, _write_request(workspace, "bad05", payload_id="different")
        )


def test_compile_gate_rejects_absolute_path(workspace):
    with pytest.raises(gate.CompileGateError, match="escapes"):
        gate.validate_compile_request(
            workspace,
            _write_request(workspace, "bad06", paths=("/usr/local/Ascend/include/acl.h",)),
        )


def test_run_pending_rejects_invalid_without_dropping_valid(workspace):
    _write_request(workspace, "ok01")
    _write_request(workspace, "bad07", paths=("PROGRESS.md",))
    responses = gate.run_pending_compile_requests(workspace, 0, build_fn=_mock_build)
    by_id = {r["request_id"]: r for r in responses}
    assert by_id["ok01"]["status"] == "ERROR"  # mock build ran
    assert by_id["bad07"]["status"] == "REJECTED"
