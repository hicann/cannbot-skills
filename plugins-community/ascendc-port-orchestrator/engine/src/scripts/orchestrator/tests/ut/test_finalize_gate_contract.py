# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Contract tests for `check_finalize_eligibility` rollback returns.

Sanity invariant: every rollback (`eligible: False`) MUST set a `gate`
field equal to a `GateID` enum value. Without this, the orchestrator's
loop-break detector + rollback-history would have no stable handle for
the rejection class — the only alternatives are reason-text matching
(magic strings) or treating all rollbacks as one signature (over-broad
loop detection).

These tests build minimal workspace fixtures that trigger each gate, then
assert (a) the `gate` field is present, (b) it's a known `GateID` value,
(c) the orchestrator-side default never has to fire.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors
import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import finalize_pipeline as fp  # noqa: E402
from finalize_pipeline import GateID  # noqa: E402


_KB_OK = """# knowledge_update.md — TestOp

## Context
Test op for gate fixture.

## Findings
1. **Test finding**: fixture content for the contract test.

## KB-promotable patterns (proposed)
None.

## Cited KB items
None.

## Anti-patterns avoided
None.
"""


def _write_verification(ws: Path, payload: dict) -> None:
    """Write verification.json. P88c (2026-05-15T08:18Z): auto-add
    perf.status if precision=PASS so universal_entrypoints gate doesn't
    fire before reaching the gate under test."""
    payload = dict(payload)
    prec = payload.get("precision", {})
    if isinstance(prec, dict) and prec.get("status") in ("PASS", "PASS_WITHIN_TOLERANCE"):
        perf = payload.setdefault("performance", {})
        if "status" not in perf and "ratio" not in perf:
            perf["status"] = "N/A"
            perf["reason"] = "fixture default — not the gate under test"
    (ws / "verification.json").write_text(json.dumps(payload))


def _write_kb(ws: Path) -> None:
    (ws / "knowledge_update.md").write_text(_KB_OK)


def _write_audit_pass(ws: Path) -> None:
    (ws / "audit_self_critic_post_worker.md").write_text(
        "## Verdict\n**Verdict: PASS** — fixture audit body."
    )


def _write_delegation_marker(ws: Path) -> None:
    (ws / ".delegation_scan_passed").write_text("ok")


def _write_model_groups(ws: Path) -> None:
    (ws / "model.py").write_text(
        "import torch\n"
        "def get_input_groups():\n"
        "    return [[torch.zeros(2)]]\n"
    )
    # DEBT-NEW (2026-05-14, OL-160): _check_universal_entrypoints gate
    # requires both model.py + model_new_ascendc.py for any PASS verdict.
    # Tests not targeting that gate need both seeded.
    (ws / "model_new_ascendc.py").write_text(
        "import torch.nn as nn\n"
        "class ModelNew(nn.Module):\n    def forward(self,x): return x\n"
        "if __name__ == '__main__':\n    pass\n"
    )


def _write_op_host_complete(ws: Path) -> None:
    """PB-33 (2026-05-14): op_host/ completeness gate satisfaction.

    Seeds the minimum 3 non-config / non-patch files. Tests reaching gates
    AFTER OP_HOST_COMPLETENESS need this helper. Earlier gates fire first
    (cheapest-checks-first principle), so tests targeting POST_WORKER_AUDIT,
    PERSIST_EVIDENCE, etc. need a complete op_host/ fixture.
    """
    op_host = ws / "op_host"
    op_host.mkdir(exist_ok=True)
    (op_host / "stub_def.cpp").write_text("// stub op_def\n")
    (op_host / "stub_tiling.cpp").write_text("// stub tiling impl\n")
    (op_host / "stub_tiling.h").write_text("// stub tiling header\n")

    # Tests that advance beyond op_host completeness also need live arch22
    # evidence now that the architecture gate is deliberately fail-closed.
    source = ws / "fixture_source"
    source_kernel = source / "op_kernel" / "arch22"
    source_kernel.mkdir(parents=True, exist_ok=True)
    (source_kernel / "fixture.cpp").write_text("// detector-approved arch22 source\n")
    (ws / ".opgen_state.json").write_text(
        json.dumps({"port_a3_source": str(source)})
    )


def _write_supported_backward_state(ws: Path) -> None:
    """Declare a supported route for tests that must pass the arch gate."""
    (ws / ".opgen_state.json").write_text(json.dumps({
        "op": "test_op",
        "opgen_mode": "backward",
    }))
    kernel = ws / "kernel"
    kernel.mkdir(exist_ok=True)
    (kernel / "libtest_op.so").write_bytes(b"compiled-extension")
    (ws / "verify_test_op.py").write_text(
        "from model_new_ascendc import ModelNew\n"
        "candidate = ModelNew()\n"
        "output = candidate(inputs)\n"
    )
    verification_path = ws / "verification.json"
    verification = json.loads(verification_path.read_text())
    precision = verification.setdefault("precision", {})
    precision.setdefault("pass_a", {
        "status": "PASS", "tier1_pass": 1, "total": 1,
    })
    verification["harness_pristine"] = {
        "state": "CLEAN", "o5_verdict": "VERIFIED",
        "sampled_at": "o5_post_verify",
    }
    verification_path.write_text(json.dumps(verification))


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    workspace = tmp_path / "test_op"
    workspace.mkdir()
    return workspace


def _assert_gate_known(elig: dict) -> str:
    assert elig["eligible"] is False, f"expected rollback, got {elig!r}"
    assert "gate" in elig, f"rollback missing `gate` field: {elig!r}"
    known = {g.value for g in GateID}
    assert elig["gate"] in known, (
        f"unknown gate {elig['gate']!r}; allowed = {sorted(known)}"
    )
    return elig["gate"]


def test_verification_file_missing(ws: Path) -> None:
    elig = fp.check_finalize_eligibility(ws)
    assert _assert_gate_known(elig) == GateID.VERIFICATION_FILE_MISSING.value


def test_verification_malformed(ws: Path) -> None:
    (ws / "verification.json").write_text("not valid json {{{")
    elig = fp.check_finalize_eligibility(ws)
    assert _assert_gate_known(elig) == GateID.VERIFICATION_MALFORMED.value


def test_model_py_shape_violation(ws: Path) -> None:
    _write_verification(ws, {"precision": {"status": "PASS"}})
    (ws / "model.py").write_text(
        "import torch\n"
        "def get_inputs():\n"
        "    return [torch.zeros(2)]\n"
    )
    elig = fp.check_finalize_eligibility(ws)
    assert _assert_gate_known(elig) == GateID.MODEL_PY_SHAPE.value


def test_pass_a_coverage_shortfall(ws: Path) -> None:
    _write_verification(ws, {
        "precision": {
            "status": "PARTIAL",
            "pass_a": {"status": "PARTIAL", "tier1_pass": 0, "total": 1},
        }
    })
    # 5 JSONL cases declared
    (ws / "test_op.json").write_text("\n".join(
        '{"inputs": []}' for _ in range(5)
    ))
    _write_model_groups(ws)
    elig = fp.check_finalize_eligibility(ws)
    assert _assert_gate_known(elig) == GateID.PASS_A_COVERAGE.value


def _setup_for_kb_gate(ws: Path) -> None:
    """Common scaffolding for KB-writeup tests — verification PASS, model.py
    OK, op_host/ complete — so KB_WRITEUP is the next gate to fire."""
    _write_verification(ws, {"precision": {"status": "PASS"}})
    _write_model_groups(ws)
    _write_op_host_complete(ws)


def test_kb_writeup_missing(ws: Path) -> None:
    _setup_for_kb_gate(ws)
    elig = fp.check_finalize_eligibility(ws)
    assert _assert_gate_known(elig) == GateID.KB_WRITEUP.value


def test_kb_writeup_too_short(ws: Path) -> None:
    _setup_for_kb_gate(ws)
    (ws / "knowledge_update.md").write_text("trivial")
    elig = fp.check_finalize_eligibility(ws)
    assert _assert_gate_known(elig) == GateID.KB_WRITEUP.value


def test_kb_writeup_no_findings_section(ws: Path) -> None:
    _setup_for_kb_gate(ws)
    (ws / "knowledge_update.md").write_text(
        "# random body\n" + "x" * 200
    )
    elig = fp.check_finalize_eligibility(ws)
    assert _assert_gate_known(elig) == GateID.KB_WRITEUP.value


def test_op_host_missing_dir(ws: Path) -> None:
    """PB-33: op_host/ directory absent → OP_HOST_COMPLETENESS gate."""
    _write_verification(ws, {"precision": {"status": "PASS"}})
    _write_model_groups(ws)
    _write_kb(ws)
    _write_audit_pass(ws)
    _write_delegation_marker(ws)
    elig = fp.check_finalize_eligibility(ws)
    assert _assert_gate_known(elig) == GateID.OP_HOST_COMPLETENESS.value
    assert "missing entirely" in elig["reason"]


def test_op_host_only_patch_no_complete_files(ws: Path) -> None:
    """PB-33: op_host/ has only `*_def.cpp.patch` (the canonical bug from
    5 archived ops). Patches are review-aid; gate requires complete files.
    """
    _write_verification(ws, {"precision": {"status": "PASS"}})
    _write_model_groups(ws)
    _write_kb(ws)
    _write_audit_pass(ws)
    _write_delegation_marker(ws)
    op_host = ws / "op_host"
    op_host.mkdir()
    (op_host / "stub_def.cpp.patch").write_text("@@ -1 +1,2 @@\n+regbaseCfg block\n")
    (op_host / "config").mkdir()
    (op_host / "config" / "ascend950").mkdir()
    (op_host / "config" / "ascend950" / "stub_binary.json").write_text("{}")
    elig = fp.check_finalize_eligibility(ws)
    assert _assert_gate_known(elig) == GateID.OP_HOST_COMPLETENESS.value
    assert "minimum 3 required" in elig["reason"]


def test_unknown_precision_status_no_skip_marker(ws: Path) -> None:
    """precision.status='N/A' never bypasses measured verification."""
    _write_verification(ws, {"precision": {"status": "N/A"}})
    _write_model_groups(ws)
    _write_op_host_complete(ws)
    elig = fp.check_finalize_eligibility(ws)
    assert elig.get("eligible") is False
    assert _assert_gate_known(elig) == GateID.UNKNOWN_PRECISION_STATUS.value


def test_check_op_host_completeness_delegated_to_baseplugin() -> None:
    """DEBT-094 phase 3 (2026-05-18): PB-33 gate body lives in BasePlugin,
    not in finalize_pipeline. AscendC paradigm modes inherit; the pipeline
    function _check_op_host_completeness is a thin dispatcher only.

    Regression: if BasePlugin.check_op_host_completeness becomes a stub
    or returns None unconditionally, every AscendC archive ships without
    PB-33 enforcement (5-op recurrence risk from 2026-05-14 incident).
    """
    from plugins.base import BasePlugin  # noqa: WPS433 — direct contract pin
    plugin = BasePlugin()
    assert callable(plugin.check_op_host_completeness), (
        "BasePlugin.check_op_host_completeness must be a callable gate body, "
        "not a stub. DEBT-094 phase 3 moved the PB-33 logic from "
        "finalize_pipeline into BasePlugin; if this method is stubbed, "
        "AscendC modes silently lose PB-33 enforcement."
    )


def test_op_host_completeness_still_fires_for_ascendc(tmp_path: Path) -> None:
    """AscendC workspaces must trip PB-33 when op_host/ is missing."""
    ws_ascendc = tmp_path / "13_Cat"
    ws_ascendc.mkdir()
    result = getattr(fp, '_check_op_host_completeness')(ws_ascendc)
    assert result is not None, (
        "PB-33 gate did NOT fire for AscendC workspace lacking op_host/. "
        "AscendC paradigm behavior must be preserved exactly after the "
        "DEBT-094 phase 3 refactor."
    )
    assert "PB-33" in result or "op_host" in result, (
        f"PB-33 error string unexpected: {result!r}"
    )


def test_finalize_pipeline_has_no_op_host_if_else() -> None:
    """DEBT-094 phase 3 invariant: _check_op_host_completeness must NOT contain
    plugin-aware if/else branches for op_host_required. User direction 2026-05-18 22:08Z:
    "防止当前的这个脚本变成一堆 if else 的这种缝合怪式的组合".

    Pin: the dispatcher is a thin shim; the gate body lives in plugins.

    DEBT-201 (2026-07-06): the function moved to finalize_dispatch.py with the
    rest of the plugin-dispatch cluster (its bare-name _get_active_plugin call
    must stay in the same module as the patched plugin). The invariant is about
    the function BODY content, not which file houses it — so the source-grep
    now reads finalize_dispatch.py.
    """
    src = (_reorg_paths.ORCH_DIR / "finalize_dispatch.py").read_text()
    # Locate the function body
    fn_marker = "def _check_op_host_completeness("
    start = src.index(fn_marker)
    next_def = src.index("\ndef ", start + 1)
    body = src[start:next_def]
    # The pre-refactor body had two plugin-aware branches:
    #   if _plug is not None and not _plug.op_host_required():
    # It must remain absent; completeness belongs inside plugins.base.BasePlugin.
    assert "op_host_required" not in body, (
        "_check_op_host_completeness still references "
        "op_host_required — the boolean-flag plugin probe was supposed to "
        "be deleted in DEBT-094 phase 3. Pipeline must not know about "
        "plugin-specific flags."
    )


def test_audit_doc_missing(ws: Path) -> None:
    _write_verification(ws, {"precision": {"status": "PASS"}})
    _write_model_groups(ws)
    _write_kb(ws)
    _write_op_host_complete(ws)
    _write_supported_backward_state(ws)
    elig = fp.check_finalize_eligibility(ws)
    assert _assert_gate_known(elig) == GateID.POST_WORKER_AUDIT.value


def test_p0aff_audit_verdict_heading_then_value_partial_waiver(ws: Path) -> None:
    """P0aff (2026-05-20, 20_Gather): parser must accept multi-line
    `## Verdict\\n\\n**PARTIAL+waiver**` markdown form (heading + value on
    separate lines), not just single-line `## Verdict: PARTIAL+waiver`.

    20_Gather kw-6 emitted heading-then-value form; parser missed it,
    iter_cap exhausted at 6/6 despite kernel being precision-clean
    (47/47 PA, 132/132 PB, det 47/47). Fix: bounded one-line lookahead
    when verdict line is heading-only.
    """
    _write_verification(ws, {"precision": {"status": "PASS"}})
    _write_model_groups(ws)
    _write_kb(ws)
    _write_op_host_complete(ws)
    _write_supported_backward_state(ws)
    _write_delegation_marker(ws)
    (ws / "audit_self_critic_post_worker.md").write_text(
        "# audit\n"
        "## C18 — delegation\nfalse-positive smoke-test oracle.\n"
        "Waiver: smoke-test oracle in __main__ block, not on kernel surface.\n"
        "## Verdict\n\n**PARTIAL+waiver**\n"
        "Detailed rationale below.\n"
    )
    elig = fp.check_finalize_eligibility(ws)
    # Parser-level assertion: even if other gates fire downstream, the
    # reason must not be "verdict is not PASS (and not PARTIAL+waiver)".
    reason = elig.get("reason", "") or ""
    assert "verdict is not PASS" not in reason, (
        f"P0aff regression: multi-line `## Verdict\\n\\n**PARTIAL+waiver**` "
        f"should parse as PARTIAL+waiver: {elig!r}"
    )


def test_p0aff_audit_verdict_heading_then_value_pass(ws: Path) -> None:
    """P0aff (2026-05-20): same multi-line form for PASS verdict."""
    _write_verification(ws, {"precision": {"status": "PASS"}})
    _write_model_groups(ws)
    _write_kb(ws)
    _write_op_host_complete(ws)
    _write_supported_backward_state(ws)
    _write_delegation_marker(ws)
    (ws / "audit_self_critic_post_worker.md").write_text(
        "# audit\n"
        "## Verdict\n\n**PASS**\n"
        "Detailed rationale below.\n"
    )
    elig = fp.check_finalize_eligibility(ws)
    reason = elig.get("reason", "") or ""
    assert "verdict is not PASS" not in reason, (
        f"P0aff regression: multi-line `## Verdict\\n\\n**PASS**` "
        f"should parse as PASS: {elig!r}"
    )


def test_p0aff_audit_verdict_heading_then_value_fail_blocks(ws: Path) -> None:
    """P0aff (2026-05-20): multi-line FAIL/BLOCK must still block."""
    _write_verification(ws, {"precision": {"status": "PASS"}})
    _write_model_groups(ws)
    _write_kb(ws)
    _write_op_host_complete(ws)
    _write_supported_backward_state(ws)
    _write_delegation_marker(ws)
    (ws / "audit_self_critic_post_worker.md").write_text(
        "# audit\n"
        "## C18 — delegation\n"
        "Cheating detected: kernel surface delegates to torch ops.\n"
        "## C26 — Verdict naming\n"
        "Stable denominator across artifacts.\n"
        "## Verdict\n\n**FAIL** — cheating detected.\n"
        "Detailed rationale and evidence below ensures non-trivial body.\n"
    )
    elig = fp.check_finalize_eligibility(ws)
    # FAIL/BLOCK form MUST still rollback at POST_WORKER_AUDIT with the
    # verdict-parsing reason (not some downstream gate).
    assert _assert_gate_known(elig) == GateID.POST_WORKER_AUDIT.value
    assert "verdict is not PASS" in elig.get("reason", ""), (
        f"P0aff: multi-line FAIL must fire verdict-parse rejection: {elig!r}"
    )


def test_p0aff_lookahead_bounded_no_drift_to_prose(ws: Path) -> None:
    """P0aff (2026-05-20): lookahead must not drift past 3 lines or past
    the first non-empty non-matching line — protects against prose like
    `## Verdict\\n\\nThe agent will issue verdict later.\\n\\n**PASS**`
    silently flipping to PASS from drift.
    """
    _write_verification(ws, {"precision": {"status": "PASS"}})
    _write_model_groups(ws)
    _write_kb(ws)
    _write_op_host_complete(ws)
    _write_supported_backward_state(ws)
    _write_delegation_marker(ws)
    (ws / "audit_self_critic_post_worker.md").write_text(
        "# audit\n"
        "## Verdict\n\nThe agent will declare verdict in a moment.\n\n**PASS**\n"
    )
    elig = fp.check_finalize_eligibility(ws)
    # First non-empty next-line ("The agent will declare ...") has no
    # pass/partial/fail token → lookahead terminates without verdict →
    # gate blocks (no verdict-line-status set).
    assert _assert_gate_known(elig) == GateID.POST_WORKER_AUDIT.value


def test_persist_evidence_missing_probe_report(ws: Path) -> None:
    _write_verification(ws, {
        "precision": {"status": "PARTIAL", "persist_verdict": "PARTIAL_PERSIST"}
    })
    _write_model_groups(ws)
    _write_op_host_complete(ws)
    elig = fp.check_finalize_eligibility(ws)
    assert _assert_gate_known(elig) == GateID.PERSIST_EVIDENCE.value
    assert elig["rollback_state"] == "await_probe"


def test_persist_evidence_partial_no_probe_yet(ws: Path) -> None:
    _write_verification(ws, {"precision": {"status": "PARTIAL"}})
    _write_model_groups(ws)
    _write_op_host_complete(ws)
    elig = fp.check_finalize_eligibility(ws)
    assert _assert_gate_known(elig) == GateID.PERSIST_EVIDENCE.value
    assert elig["rollback_state"] == "await_probe"


def test_unknown_precision_status(ws: Path) -> None:
    _write_verification(ws, {"precision": {"status": "WAT"}})
    _write_model_groups(ws)
    _write_op_host_complete(ws)
    elig = fp.check_finalize_eligibility(ws)
    assert _assert_gate_known(elig) == GateID.UNKNOWN_PRECISION_STATUS.value


def test_eligible_path_omits_gate(ws: Path) -> None:
    """Sanity: the eligible (PASS) path returns no `gate` field — `gate` is
    only meaningful when rollback fires.
    """
    _write_verification(ws, {
        "precision": {
            "status": "PARTIAL",
            "persist_verdict": "PARTIAL_PERSIST",
        }
    })
    _write_model_groups(ws)
    _write_op_host_complete(ws)
    (ws / "probe_report.md").write_text("probe evidence")
    # DEBT-211: PARTIAL_PERSIST now also gates on the delegation-scan marker,
    # so a legitimately-eligible PARTIAL_PERSIST archive must have it (as it
    # does in production, written by _ensure_audit_artifacts pre-finalize).
    _write_delegation_marker(ws)
    elig = fp.check_finalize_eligibility(ws)
    assert elig["eligible"] is True
    assert elig.get("gate") is None or "gate" not in elig


def test_every_gate_id_has_distinct_value() -> None:
    """No two GateID entries share a value — collision would silently
    merge two distinct gate categories into one rollback signature.
    """
    values = [g.value for g in GateID]
    assert len(values) == len(set(values)), (
        f"duplicate GateID values: {values}"
    )


def test_gate_id_values_are_lowercase_snake_case() -> None:
    """Stylistic invariant: GateID values follow lowercase_snake_case (with
    digits allowed for version-like tokens — e.g. `port_a3_pass_b_schema`)
    so they're stable across log/grep/serialization.
    """
    import re
    pat = re.compile(r"^[a-z][a-z0-9_]*$")
    for g in GateID:
        assert pat.match(g.value), (
            f"GateID.{g.name} value {g.value!r} is not lowercase_snake_case"
        )
