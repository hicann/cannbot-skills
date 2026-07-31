# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""port_a3 COMPLETE-DELIVERABLE brief mandate tests (fix/port-a3-complete-deliverable).

Background (celu a3_port incident, 2026-06-16): a NON-FA port_a3 op reached
O5-confirmed (precision PASS) but `finalize` could NOT promote the archive
because it hit 3 port_a3 COMPLETENESS gates in finalize_pipeline.py, each
rollback exposing a new missing artifact:
  ① op_host_completeness  — workspace/op_host/ missing the GE op_host trio
                             (<op>_def.cpp + _tiling.cpp + _tiling.h + CMakeLists)
  ② binary_provenance     — own-build SHA256 lineage missing
  ③ KB_WRITEUP / Findings — knowledge_update.md missing `## Findings`

All 3 gates are NOT FA-specific:
  ① is universal across AscendC modes (base.py BasePlugin.check_op_host_completeness)
  ② is universal for port_a3 mode + PASS verdict (port_a3 plugin check_binary_provenance)
  ③ is universal for ALL modes + PASS verdict (finalize_pipeline KB_WRITEUP gate)

But the brief only MANDATED these deliverables for FA (the FA-scoped
`_fa_ge_host_gen_block`). A non-FA port_a3 op (celu/elu) was never told to
emit them up-front, so finalize rejected it one gate at a time.

These tests assert the GENERAL port_a3 brief (the `_port_a3_phase_instructions_block`
that fires for ANY port_a3 op, FA or not) now carries the full deliverable
mandate matching all 3 gate scopes. The fix is on the PRODUCING side (brief),
NOT the GATE side — the gates are unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from briefs import kw_brief, op_taxonomy  # noqa: E402
from briefs._common import AscendCEnv  # noqa: E402


@pytest.fixture(autouse=True)
def _combined_pr_kb_boundary(monkeypatch):
    """PR #609 is tested with the KB shell supplied by its companion PR."""
    monkeypatch.setattr(op_taxonomy, "validate_manifest_paths", lambda sections: None)


def _make_env(**kwargs) -> AscendCEnv:
    defaults = dict(
        target="a5", host="h", user="u", password="", container="c",
        cann_path="/p", soc_version="s", benchmark_root="", local_benchmark="",
        local_project="", archive_project="a3_to_a5_port",
        build_archive_enabled=False, opgen_mode="port_a3_to_a5",
        port_a3_source="", a3_host="", a3_user="root", a3_container="",
        a3_cann_path="", a3_soc_version="", a3_workspace="",
    )
    defaults.update(kwargs)
    return AscendCEnv(**defaults)


def _nonfa_brief(tmp_path: Path, op: str = "celu") -> str:
    """Build the port_a3 brief for a simple NON-FA unary-activation op."""
    ws = tmp_path / "workspace" / op
    ws.mkdir(parents=True)
    (ws / "a3_reference_runnable.json").write_text(json.dumps({
        "verdict": "READY_PROBE_ONLY",
        "aclnn_entry": f"/cann/ops-nn/activation/{op}/examples/test_aclnn_{op}.cpp",
        "gen_data_source": None,
        "peer_op_dependencies": [],
    }))
    env = _make_env(port_a3_source=f"/cann/ops-nn/activation/{op}")
    return kw_brief.build_worker_brief(
        op=op, workspace=ws, lane=0, spawn_index=1,
        iter_cap_remaining=3, env=env,
    )


def _is_fa_named(op: str) -> bool:
    from plugins.base import is_attention_named
    return is_attention_named(op)


# ── Gate ① op_host_completeness — GE op_host derived from A3 source ─────────
def test_nonfa_brief_mandates_op_host_via_recipe(tmp_path):
    """NON-FA port_a3 op must be told to GENERATE the GE op_host by following
    the per-file CARRY / CARRY+PATCH / REPLACE-HOOK recipe (NOT byte-copy),
    deriving from its A3 (arch22) op_host source — satisfying gate ①.
    """
    op = "celu"
    assert not _is_fa_named(op), "test fixture must be a non-FA op"
    brief = _nonfa_brief(tmp_path, op)
    # The general op-class recipe pointer (the recipe is explicitly op-CLASS-general).
    assert "GE_HOST_TRANSFORM_RECIPE" in brief
    # The three transform classes must be named so the worker applies them.
    assert "CARRY" in brief
    assert "REPLACE-HOOK" in brief or "REPLACE_HOOK" in brief
    # The required deliverable trio + the gate it satisfies.
    assert "_def.cpp" in brief and "_tiling.cpp" in brief and "_tiling.h" in brief
    assert "op_host_completeness" in brief or "PB-33" in brief
    # Derive-from-A3, NOT byte-copy CANN.
    assert "byte-copy" in brief.lower() or "byte copy" in brief.lower()


# ── Gate ② binary_provenance — own-build SHA256 lineage ─────────────────
def test_nonfa_brief_mandates_binary_provenance_sha256(tmp_path):
    """The brief must require a workspace-contained source-to-binary chain."""
    brief = _nonfa_brief(tmp_path, "celu")
    assert "build_evidence" in brief
    assert "compiled_provenance" in brief
    assert "workspace_source_sha256" in brief
    assert "deploy_source_sha256" in brief
    assert "built_from_source_sha256" in brief
    assert "object_sha256" in brief and "shared_lib_sha256" in brief
    assert "installed CANN target tree" in brief


# ── Gate ③ KB_WRITEUP — knowledge_update.md ## Findings ─────────────────────
def test_nonfa_brief_mandates_findings_section(tmp_path):
    """NON-FA port_a3 op must be told the knowledge_update.md MUST carry the
    `## Findings` header (Phase E template) — the KB_WRITEUP gate fires on any
    PASS verdict in every mode.
    """
    brief = _nonfa_brief(tmp_path, "celu")
    assert "## Findings" in brief
    assert "knowledge_update.md" in brief
    # The 5-section structure is named.
    assert "Context" in brief and "Findings" in brief


# ── The 3 deliverables are co-located as ONE completeness contract ──────────
def test_nonfa_brief_completeness_contract_single_block(tmp_path):
    """All 3 deliverables must be presented together as the 'complete
    archivable deliverable' so finalize promotes in ONE pass (no one-by-one
    gate churn). Assert a single contract marker ties them together.
    """
    brief = _nonfa_brief(tmp_path, "celu")
    assert "COMPLETE ARCHIVABLE DELIVERABLE" in brief or \
           "complete archivable deliverable" in brief.lower()


# ── FA stays a specialization (regression: FA path still gets its block) ────
def test_fa_brief_still_has_specialized_ge_host_block(tmp_path):
    """FA remains a SPECIALIZATION: the FA template-assembly path still emits
    the FA-specific GE-host block (wp_fa_host_tiling.h / wfh:: shared layer).
    The generalization must NOT regress FA.
    """
    op = "flash_attention_score"
    assert _is_fa_named(op), "test fixture must be an FA op"
    ws = tmp_path / "workspace" / op
    ws.mkdir(parents=True)
    # The FA template-assembly block only fires when op_classification.json
    # is present (kw_brief._fa_class_template_assembly_block gate).
    (ws / "op_classification.json").write_text(json.dumps({
        "op_class_tags": ["FUSED", "SOFTMAX", "ATTENTION", "REDUCTION"],
    }))
    (ws / "a3_reference_runnable.json").write_text(json.dumps({
        "verdict": "READY_PROBE_ONLY",
        "aclnn_entry": f"/cann/ops-transformer/attention/{op}/examples/test.cpp",
        "gen_data_source": None,
        "peer_op_dependencies": [],
    }))
    env = _make_env(port_a3_source=f"/cann/ops-transformer/attention/{op}")
    brief = kw_brief.build_worker_brief(
        op=op, workspace=ws, lane=0, spawn_index=1,
        iter_cap_remaining=3, env=env,
    )
    # FA specialization markers (the shared arch35 tiling layer).
    assert "wp_fa_host_tiling.h" in brief or "wfh::" in brief
    assert "GE_OPHOST_RAW_CANN_COPY" in brief


# ── Regression: supported workflow brief composition ─────────────────────────
