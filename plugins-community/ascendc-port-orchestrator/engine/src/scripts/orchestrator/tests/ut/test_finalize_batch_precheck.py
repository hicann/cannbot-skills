# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Batch precheck for port_a3 finalize completeness/eligibility gates
(fix/port-a3-finalize-batch-precheck, 2026-06-16).

THE PROBLEM this guards: check_finalize_eligibility early-returns on the FIRST
failed PASS-branch gate → the orchestrator rolls back to await_worker for that
one gate → worker fixes it + respawns → finalize catches the NEXT gate → roll
back again. Serial churn (one expensive worker respawn per gate). celu
2026-06-16/17 hit op_host → md5 → Findings → P146 one-by-one.

THE FIX: finalize_pipeline.batch_precheck runs the SAME gate set (single source
of truth: _pass_branch_gate_specs) in COLLECT-ALL mode and reports EVERY failure
at once, so the worker fixes everything in ONE respawn.

This test asserts:
  1. A port_a3 PASS workspace MISSING op_host + md5 + Findings + P146-perf-
     retraction → batch_precheck reports ALL of those at once (not just first).
  2. A COMPLETE port_a3 PASS workspace → batch_precheck.ok is True.
  3. The aggregate is consistent with check_finalize_eligibility's FIRST gate
     (batch precheck does NOT change pass/fail logic — same gates, aggregated).
"""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import finalize_pipeline as fp  # noqa: E402
from finalize_pipeline import GateID  # noqa: E402
from source_arch import stage_source_tree  # noqa: E402


_KB_WITH_FINDINGS = (
    "# knowledge_update.md — celu\n\n"
    "## Context\nport celu arch22->arch35.\n\n"
    "## Findings\n1. CARRY infershape; PATCH def SOC.\n\n"
    "## KB-promotable patterns\nNone.\n\n"
    "## Cited KB items\nW8/W11.\n\n"
    "## Anti-patterns avoided\nno arch35 include.\n"
)

_KB_NO_FINDINGS = (
    "# knowledge_update.md — celu\n\n"
    "Context: ported celu (no Findings header).\n" + ("filler.\n" * 30)
)

_AUDIT_PASS = "## Verdict\n**Verdict: PASS** — fixture audit body, substantive.\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compiled_evidence(ws: Path) -> dict:
    source = ws / "op_host" / "celu_tiling.cpp"
    deployed = ws / "build" / "deploy" / source.name
    object_file = ws / "build" / f"{source.name}.o"
    shared_lib = ws / "build" / "libcelu.so"
    for path in (deployed, object_file, shared_lib):
        path.parent.mkdir(parents=True, exist_ok=True)
    deployed.write_bytes(source.read_bytes())
    object_file.write_bytes(b"own object")
    shared_lib.write_bytes(b"own shared library")
    source_digest = _sha256(source)
    return {
        "compiled_provenance": {
            "source": str(source.relative_to(ws)),
            "deployed_source": str(deployed.relative_to(ws)),
            "object": str(object_file.relative_to(ws)),
            "shared_lib": str(shared_lib.relative_to(ws)),
            "workspace_source_sha256": source_digest,
            "deploy_source_sha256": source_digest,
            "built_from_source_sha256": source_digest,
            "object_sha256": _sha256(object_file),
            "shared_lib_sha256": _sha256(shared_lib),
        }
    }


def _write_op_host(ws: Path) -> None:
    oh = ws / "op_host"
    oh.mkdir(exist_ok=True)
    (oh / "celu_def.cpp").write_text("// generated A5 op_def\n")
    (oh / "celu_tiling.cpp").write_text("// generated tiling impl\n")
    (oh / "celu_tiling.h").write_text("// tiling header\n")


def _write_entrypoints(ws: Path) -> None:
    (ws / "model.py").write_text(
        "import torch\n"
        "def get_input_groups():\n    return [[torch.zeros(2)]]\n"
    )
    (ws / "model_new_ascendc.py").write_text(
        "import torch.nn as nn\n"
        "class ModelNew(nn.Module):\n    def forward(self,x): return x\n"
        "if __name__ == '__main__':\n    pass\n"
    )


# P146-compliant perf retraction (port_a3 + PASS + perf N/A requires per-option
# infeasibility evidence). This is the form a complete deliverable must carry.
_PERF_RETRACTION_OK = {
    "status": "N/A",
    "reason": "port_a3 perf retraction — see retraction.reason (P146 evidence)",
    "retraction": {
        "reason": (
            "option_1_infeasible_because aclrtEvent on A3 + torch.npu.Event "
            "on A5 cannot be applied: no stream handle exposed by the aclnn "
            "runner shim (cite runner.cpp L40). "
            "option_2_infeasible_because perf_counter cannot wrap each side's "
            "natural call shape: no python entry on the A3 side (cite "
            "build_runner.sh)."
        ),
    },
}


def _make_port_a3_pass_ws(tmp_path: Path, *, complete: bool) -> Path:
    """Build a port_a3 PASS workspace. complete=True clears every PASS-branch
    gate; complete=False omits op_host + md5 + Findings + P146-retraction."""
    ws = tmp_path / "celu"
    ws.mkdir()
    source = tmp_path / "source-celu"
    source_kernel = source / "op_kernel" / "arch22"
    source_kernel.mkdir(parents=True)
    (source_kernel / "celu.h").write_text(
        "class Celu { void Process() { DataCopy(dst, src, 1); } };\n"
    )
    stage = stage_source_tree(source, ws)
    (ws / ".opgen_state.json").write_text(json.dumps({
        "op": "celu",
        "opgen_mode": "port_a3_to_a5",
        "source_arch": "arch22",
        "target_arch": "arch35",
        "port_a3_source": str(stage.root),
        "graybox_arch22_dir": str(stage.root),
        "graybox_sandbox": True,
        "source_stage_manifest": str(stage.manifest),
        "source_stage_digest": stage.digest,
    }))

    vj: dict = {
        "op": "celu",
        "mode": "port_a3_to_a5",
        "precision": {
            "status": "PASS",
            "pass_b": {"status": "N/A", "reason": "OL-68 Case A — torch_npu "
                       "reference unrunnable on A5"},
        },
    }
    if complete:
        _write_op_host(ws)
        vj["build_evidence"] = _compiled_evidence(ws)
        vj["performance"] = _PERF_RETRACTION_OK
    else:
        # MISSING md5 (empty build_evidence) + MISSING P146 retraction (bare
        # N/A perf with no per-option infeasibility evidence).
        vj["build_evidence"] = {}
        vj["performance"] = {"status": "N/A", "reason": "not measured"}
    (ws / "verification.json").write_text(json.dumps(vj))

    _write_entrypoints(ws)
    _write_audit_and_marker(ws)

    if complete:
        (ws / "knowledge_update.md").write_text(_KB_WITH_FINDINGS)  # ③ Findings
    else:
        # ① op_host MISSING (no dir)
        # ③ knowledge_update.md present but NO `## Findings` header
        (ws / "knowledge_update.md").write_text(_KB_NO_FINDINGS)
    return ws


def _write_audit_and_marker(ws: Path) -> None:
    (ws / "audit_self_critic_post_worker.md").write_text(_AUDIT_PASS)
    (ws / ".delegation_scan_passed").write_text("ok")


# ── (2) COMPLETE workspace → precheck passes ────────────────────────────────
def test_complete_port_a3_ws_precheck_passes(tmp_path):
    ws = _make_port_a3_pass_ws(tmp_path, complete=True)
    result = fp.batch_precheck(ws)
    assert result["applicable"] is True
    assert result["ok"] is True, (
        f"complete workspace should clear all gates, got failures: "
        f"{result['failures']}"
    )
    assert result["failures"] == []
    # And the authoritative backstop agrees it's eligible.
    elig = fp.check_finalize_eligibility(ws)
    assert elig["eligible"] is True, elig


# ── (1) INCOMPLETE workspace → ALL 4 failures reported AT ONCE ──────────────
def test_incomplete_ws_reports_all_four_at_once(tmp_path):
    ws = _make_port_a3_pass_ws(tmp_path, complete=False)
    result = fp.batch_precheck(ws)
    assert result["applicable"] is True
    assert result["ok"] is False
    assert result["precondition_block"] is None

    fired = {f["gate"] for f in result["failures"]}
    # The 4 (or all-N) missing deliverables MUST all appear AT ONCE — not just
    # the first one check_finalize_eligibility would early-return on.
    assert GateID.KB_WRITEUP.value in fired, fired           # ③ Findings
    assert GateID.OP_HOST_COMPLETENESS.value in fired, fired  # ① op_host
    assert GateID.BINARY_PROVENANCE.value in fired, fired     # ② md5
    assert GateID.PERF_METHODOLOGY_ASYMMETRY.value in fired, fired  # ④ P146
    assert len(result["failures"]) >= 4, result["failures"]


def test_aggregate_reports_more_than_first_gate(tmp_path):
    """The whole point: batch precheck surfaces MORE than the single gate
    check_finalize_eligibility early-returns on (proves no serial churn).
    """
    ws = _make_port_a3_pass_ws(tmp_path, complete=False)
    elig = fp.check_finalize_eligibility(ws)        # early-return: FIRST gate
    result = fp.batch_precheck(ws)                  # aggregate: ALL gates
    assert elig["eligible"] is False
    first_gate = elig["gate"]
    fired = {f["gate"] for f in result["failures"]}
    # The first gate the backstop fires MUST be among the aggregated set ...
    assert first_gate in fired, (first_gate, fired)
    # ... and the aggregate has strictly MORE (the serial-churn we eliminate).
    assert len(fired) > 1, fired


def test_format_report_lists_every_failure(tmp_path):
    ws = _make_port_a3_pass_ws(tmp_path, complete=False)
    result = fp.batch_precheck(ws)
    report = fp.format_batch_precheck_report(result)
    # Every fired gate id appears in the worker-facing report.
    for f in result["failures"]:
        assert f["gate"] in report, f"{f['gate']} missing from report"
    assert "one respawn" in report.lower()


# ── (3) precheck is byte-equivalent to the per-gate logic ───────────────────
def test_each_aggregated_gate_matches_individual_check(tmp_path):
    """Every reason the aggregate collects equals exactly what the underlying
    _check_* function returns — batch precheck did NOT alter gate logic, it
    only reorders WHEN they run (all at once vs one-by-one).
    """
    ws = _make_port_a3_pass_ws(tmp_path, complete=False)
    v = json.loads((ws / "verification.json").read_text())
    result = fp.batch_precheck(ws)
    reasons = {f["gate"]: f["reason"] for f in result["failures"]}

    # Spot-check the 4 target gates against their source-of-truth predicates.
    assert reasons[GateID.OP_HOST_COMPLETENESS.value] == \
        getattr(fp, '_check_op_host_completeness')(ws)
    assert reasons[GateID.BINARY_PROVENANCE.value] == \
        getattr(fp, '_check_binary_provenance')(ws, v)
    assert reasons[GateID.KB_WRITEUP.value] == getattr(fp, '_check_kb_writeup')(ws, v)
    assert reasons[GateID.PERF_METHODOLOGY_ASYMMETRY.value] == \
        getattr(fp, '_check_perf_methodology')(ws, v)


# ── precondition-block path (non-PASS / missing verification.json) ──────────
def test_missing_verification_is_precondition_block(tmp_path):
    ws = tmp_path / "celu"
    ws.mkdir()
    result = fp.batch_precheck(ws)
    assert result["ok"] is False
    assert result["applicable"] is True
    assert result["precondition_block"]["gate"] == \
        GateID.VERIFICATION_FILE_MISSING.value
    assert result["failures"] == []


def test_non_pass_status_not_applicable(tmp_path):
    ws = tmp_path / "celu"
    ws.mkdir()
    (ws / "verification.json").write_text(json.dumps({
        "op": "celu", "mode": "port_a3_to_a5",
        "precision": {"status": "PARTIAL", "persist_verdict": "PARTIAL_PERSIST"},
    }))
    result = fp.batch_precheck(ws)
    # PARTIAL is handled wholly by check_finalize_eligibility's own branch;
    # the precheck has no PASS-branch gates to aggregate.
    assert result["applicable"] is False
    assert result["ok"] is True
    assert result["failures"] == []


# ── single source of truth: precheck + finalize iterate the SAME specs ──────
def test_precheck_and_finalize_share_gate_specs():
    """DRY invariant: there is exactly ONE gate list. If a new PASS-branch gate
    is added to _pass_branch_gate_specs, BOTH the precheck and the early-return
    finalize path pick it up — no hardcoded duplicate to drift (the failure
    mode that let P146 slip past the brief-contract approach).
    """
    specs = getattr(fp, '_pass_branch_gate_specs')()
    assert len(specs) >= 20, len(specs)
    # All gate ids are GateID members except the plugin-extras sentinel.
    known = {g.value for g in GateID}
    for gate_id, predicate in specs:
        assert callable(predicate)
        if gate_id == getattr(fp, '_PLUGIN_EXTRAS_SENTINEL'):
            continue
        assert gate_id.value in known, gate_id
