# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""W5 (2026-05-12, ROADMAP §1.5) — kw_brief port_from_a3_ascendc mode tests.

Validates:
- kw_brief branches to port_a3 prose when env.opgen_mode == "port_a3_to_a5"
- Port brief reads workspace/a3_reference_runnable.json + surfaces:
  * aclnn entry path
  * gen_data source (or absent indicator)
  * peer_op_dependencies (the cross-op-router-patch trigger from gap audit §1)
- Port brief contents include the canonical migration phase structure (A/B/C/D/E)
- Missing a3_reference_runnable.json doesn't crash brief building
- Cross-op peer dep surfaced when present (ctc_loss_v3 → ctc_loss_v2)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from briefs import kw_brief  # noqa: E402
from briefs._common import AscendCEnv  # noqa: E402


def make_env(opgen_mode: str = "port_a3_to_a5", **kwargs) -> AscendCEnv:
    """Minimal AscendCEnv for tests."""
    defaults = dict(
        target="a5",
        host="198.51.100.35",
        user="root",
        password="",
        container="npu_dev3",
        cann_path="/data/cann_b103/cann-9.0.0",
        soc_version="Ascend950PR_9579",
        benchmark_root="/root/AscendOpGenAgent",
        local_benchmark="/tmp/bench",
        local_project="/tmp/proj",
        archive_project="test_archive",
        build_archive_enabled=False,
        opgen_mode=opgen_mode,
        port_a3_source="",
        a3_host="",
        a3_user="root",
        a3_container="",
        a3_cann_path="",
        a3_soc_version="",
        a3_workspace="",
    )
    defaults.update(kwargs)
    return AscendCEnv(**defaults)


def make_workspace_with_a3_runnable(tmp_path: Path, payload: dict) -> Path:
    """Create a workspace dir with the given a3_reference_runnable.json."""
    ws = tmp_path / "workspace" / "ctc_loss_v3"
    ws.mkdir(parents=True)
    (ws / "a3_reference_runnable.json").write_text(json.dumps(payload))
    return ws


# ---------------------------------------------------------------------------
# Port mode brief — happy path with peer dep
# ---------------------------------------------------------------------------
def test_port_a3_brief_routes_when_opgen_mode_set(tmp_path):
    """env.opgen_mode='port_a3_to_a5' triggers the port phase block."""
    ws = make_workspace_with_a3_runnable(tmp_path, {
        "verdict": "READY_PROBE_ONLY",
        "aclnn_entry": "/cann/ops-nn/loss/ctc_loss_v3/examples/test_aclnn_ctc_loss_v3.cpp",
        "gen_data_source": None,
        "peer_op_dependencies": ["ctc_loss_v2"],
    })
    env = make_env(
        opgen_mode="port_a3_to_a5",
        port_a3_source="/cann/ops-nn/loss/ctc_loss_v3",
        a3_host="198.51.100.70",
        a3_container="npu-a3",
    )
    brief = kw_brief.build_worker_brief(
        op="ctc_loss_v3", workspace=ws, lane=0, spawn_index=1,
        iter_cap_remaining=3, env=env,
    )
    # Port-mode markers present
    assert "port_from_a3_ascendc" in brief or "arch22→arch35 port mode" in brief
    assert "ROADMAP §1.5" in brief
    # OPS-NN source path surfaced
    assert "/cann/ops-nn/loss/ctc_loss_v3" in brief
    # Aclnn entry surfaced
    assert "test_aclnn_ctc_loss_v3.cpp" in brief
    # Peer-op dep surfaced (this is the load-bearing W5 feature)
    assert "ctc_loss_v2" in brief
    # Port-specific phase content
    assert "arch35" in brief
    assert "ascend950" in brief
    assert "apt.cpp" in brief or "_apt.cpp" in brief
    assert "Cross-Op" in brief or "cross-op" in brief.lower() or "CROSS-OP" in brief


def test_port_a3_brief_requires_live_capture_list_truth(tmp_path):
    """The brief must teach the single accepted fresh source-NPU capture shape.

    A CPU-canonical dict is diagnostic context, not migration truth, so the
    worker must reject it instead of treating two provenances as interchangeable.
    """
    ws = make_workspace_with_a3_runnable(tmp_path, {
        "verdict": "READY_PROBE_ONLY",
        "aclnn_entry": "/cann/ops-nn/foo/examples/test_aclnn_foo.cpp",
        "gen_data_source": None,
        "peer_op_dependencies": [],
    })
    env = make_env(opgen_mode="port_a3_to_a5", port_a3_source="/cann/ops-nn/foo")
    brief = kw_brief.build_worker_brief(
        op="foo", workspace=ws, lane=0, spawn_index=1,
        iter_cap_remaining=3, env=env,
    )
    assert "DEBT-137" in brief
    assert "top-level **LIST**" in brief
    assert "assert isinstance(data, list)" in brief
    assert "live A3 capture must be a list of case records" in brief
    assert "Reject any" in brief and "other top-level shape" in brief
    assert "Keep the capture-provenance checks active" in brief
    assert "CPU-canonical fallback" not in brief


def test_port_a3_brief_no_peer_deps_surfaces_none_marker(tmp_path):
    """When peer_op_dependencies is empty, brief shows 'single-op port'."""
    ws = make_workspace_with_a3_runnable(tmp_path, {
        "verdict": "READY_PROBE_ONLY",
        "aclnn_entry": "/foo/bar/examples/test_aclnn_op_x.cpp",
        "gen_data_source": "/foo/bar/tests/ut/op_kernel/op_x_data/gen_data.py",
        "peer_op_dependencies": [],
    })
    env = make_env(
        opgen_mode="port_a3_to_a5",
        port_a3_source="/foo/bar",
    )
    brief = kw_brief.build_worker_brief(
        op="op_x", workspace=ws, lane=0, spawn_index=1,
        iter_cap_remaining=3, env=env,
    )
    # Should say "(none — single-op port)" or similar
    assert "single-op port" in brief or "none" in brief.lower()
    # gen_data present (should NOT say absent)
    assert "gen_data.py" in brief
    assert "absent" not in brief.lower() or "must be hand-authored" not in brief.lower()


def test_port_a3_brief_handles_missing_a3_runnable(tmp_path):
    """If a3_reference_runnable.json doesn't exist yet, brief degrades gracefully."""
    ws = tmp_path / "workspace" / "ctc_loss_v3"
    ws.mkdir(parents=True)
    # NOTE: no a3_reference_runnable.json
    env = make_env(
        opgen_mode="port_a3_to_a5",
        port_a3_source="/cann/ops-nn/loss/ctc_loss_v3",
    )
    brief = kw_brief.build_worker_brief(
        op="ctc_loss_v3", workspace=ws, lane=0, spawn_index=1,
        iter_cap_remaining=3, env=env,
    )
    # Should still produce a brief, with markers for missing context
    assert "port_from_a3_ascendc" in brief or "arch22→arch35 port mode" in brief
    # Should NOT crash — and should hint that Phase O2.5 is pending
    assert "Phase O2.5" in brief or "phase_o25" in brief or "W4" in brief


def test_port_a3_brief_handles_corrupt_a3_runnable(tmp_path):
    """If a3_reference_runnable.json is malformed JSON, brief still builds."""
    ws = tmp_path / "workspace" / "op"
    ws.mkdir(parents=True)
    (ws / "a3_reference_runnable.json").write_text("not valid json {")
    env = make_env(
        opgen_mode="port_a3_to_a5",
        port_a3_source="/cann/ops-nn/x/op",
    )
    brief = kw_brief.build_worker_brief(
        op="op", workspace=ws, lane=0, spawn_index=1,
        iter_cap_remaining=3, env=env,
    )
    # Should not crash + indicate read failure
    assert "failed to read" in brief.lower() or "a3_reference_runnable.json" in brief


def test_directive_text_overrides_port_mode(tmp_path):
    """Even in port mode, a directive_text (from probe/optimizer respawn)
    routes through the directive branch, not the port branch.
    """
    ws = make_workspace_with_a3_runnable(tmp_path, {
        "verdict": "READY_PROBE_ONLY",
        "peer_op_dependencies": ["ctc_loss_v2"],
    })
    env = make_env(opgen_mode="port_a3_to_a5",
                   port_a3_source="/cann/ops-nn/loss/ctc_loss_v3")
    brief = kw_brief.build_worker_brief(
        op="ctc_loss_v3", workspace=ws, lane=0, spawn_index=2,
        iter_cap_remaining=2, env=env,
        directive_text="Fix the BF16 cast in line 234 of arch35/ctc_loss_v3.h",
    )
    # Directive text wins
    assert "DIRECTIVE FROM PRIOR AGENT" in brief
    assert "Fix the BF16 cast" in brief
    # Phase A/B/C/D from directive branch (NOT port-mode phases)
    assert "Apply the directive above" in brief
