# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""port_a3 COMPLETE-DELIVERABLE finalize-gate integration test
(fix/port-a3-complete-deliverable, 2026-06-16).

Companion to test_kw_brief_port_a3_complete_deliverable.py (which asserts the
PRODUCING side — the brief mandate). This asserts the GATE side is intact and
behaves as the brief contract promises:

  WITH all 3 deliverables present → the 3 completeness gates PASS (None).
  WITHOUT each deliverable        → that gate still FAILS-LOUD (non-None).

The gates are deliberately UNCHANGED by this fix (they correctly reject an
incomplete port — anti-reward-hacking). This test locks in that the brief's
"emit all 3 up-front → finalize promotes in one pass" claim is true: a
workspace shaped like the brief mandates clears all 3 gates.

Gates exercised (all NON-FA-specific):
  ① _check_op_host_completeness  — universal AscendC (base.py)
  ② _check_binary_provenance     — universal port_a3 + PASS
  ③ KB_WRITEUP `## Findings`     — universal all modes + PASS (inline structural check)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import finalize_pipeline as fp  # noqa: E402


def _make_port_a3_ws(tmp_path: Path) -> Path:
    ws = tmp_path / "celu"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text(json.dumps({
        "op": "celu",
        "opgen_mode": "port_a3_to_a5",
        "port_a3_source": "/cann/ops-nn/activation/celu",
    }))
    return ws


# ── deliverable ① — GE op_host completeness ─────────────────────────────────
def _write_op_host(ws: Path, n_files: int = 3) -> None:
    oh = ws / "op_host"
    oh.mkdir(exist_ok=True)
    names = ["celu_def.cpp", "celu_tiling.cpp", "celu_tiling.h", "CMakeLists.txt"]
    for name in names[:n_files]:
        (oh / name).write_text("// generated A5 op_host (CARRY+PATCH from A3)\n")


def test_op_host_present_passes_gate(tmp_path):
    ws = _make_port_a3_ws(tmp_path)
    _write_op_host(ws, n_files=4)
    assert getattr(fp, '_check_op_host_completeness')(ws) is None


def test_op_host_missing_dir_fails_loud(tmp_path):
    ws = _make_port_a3_ws(tmp_path)
    # no op_host/ at all
    err = getattr(fp, '_check_op_host_completeness')(ws)
    assert err is not None
    assert "op_host" in err and "PB-33" in err


def test_op_host_too_few_files_fails_loud(tmp_path):
    ws = _make_port_a3_ws(tmp_path)
    _write_op_host(ws, n_files=2)  # < 3 required
    err = getattr(fp, '_check_op_host_completeness')(ws)
    assert err is not None
    assert "minimum 3" in err or "non-config" in err


# ── deliverable ② — own-build SHA256 provenance ────────────────────────────
def _vj_pass(build_evidence: dict | None = None) -> dict:
    vj = {"precision": {"status": "PASS"}, "op": "celu"}
    if build_evidence is not None:
        vj["build_evidence"] = build_evidence
    return vj


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
    object_file.write_bytes(b"object")
    shared_lib.write_bytes(b"shared")
    digest = _sha256(source)
    return {
        "compiled_provenance": {
            "source": str(source.relative_to(ws)),
            "deployed_source": str(deployed.relative_to(ws)),
            "object": str(object_file.relative_to(ws)),
            "shared_lib": str(shared_lib.relative_to(ws)),
            "workspace_source_sha256": digest,
            "deploy_source_sha256": digest,
            "built_from_source_sha256": digest,
            "object_sha256": _sha256(object_file),
            "shared_lib_sha256": _sha256(shared_lib),
        }
    }


def test_binary_provenance_present_passes_gate(tmp_path):
    ws = _make_port_a3_ws(tmp_path)
    _write_op_host(ws)
    vj = _vj_pass(_compiled_evidence(ws))
    (ws / "verification.json").write_text(json.dumps(vj))
    assert getattr(fp, '_check_binary_provenance')(ws, vj) is None


def test_binary_provenance_missing_lineage_fails_loud(tmp_path):
    ws = _make_port_a3_ws(tmp_path)
    vj = _vj_pass({})  # empty build_evidence
    (ws / "verification.json").write_text(json.dumps(vj))
    err = getattr(fp, '_check_binary_provenance')(ws, vj)
    assert err is not None
    assert "compiled_provenance" in err and "SHA256" in err


# ── deliverable ③ — knowledge_update.md ## Findings ─────────────────────────
_FINDINGS_DOC = (
    "## Context\nport celu arch22->arch35.\n\n"
    "## Findings\nCARRY infershape; PATCH def SOC; CARRY tiling.\n\n"
    "## KB-promotable patterns\nnone new.\n\n"
    "## Cited KB items\nW8/W11.\n\n"
    "## Anti-patterns avoided\nno arch35 include.\n"
)


def _findings_gate_check(ws: Path) -> str | None:
    """Mirror the inline KB_WRITEUP structural check in
    finalize_pipeline.check_finalize_eligibility (the `## Findings` branch),
    which fires on PASS verdicts across all modes."""
    ku = ws / "knowledge_update.md"
    if not ku.exists():
        ku = ws / ".harness" / "knowledge_update.md"
    if not ku.exists():
        return "knowledge_update.md missing"
    body = ku.read_text(encoding="utf-8", errors="replace")
    if len(body) < 100:
        return f"only {len(body)} bytes"
    if "## Findings" not in body and "## findings" not in body.lower():
        return "lacks `## Findings` section"
    return None


def test_findings_present_passes_gate(tmp_path):
    ws = _make_port_a3_ws(tmp_path)
    (ws / "knowledge_update.md").write_text(_FINDINGS_DOC)
    assert _findings_gate_check(ws) is None


def test_findings_missing_fails_loud(tmp_path):
    ws = _make_port_a3_ws(tmp_path)
    # writeup present, non-trivial, but NO `## Findings` header
    (ws / "knowledge_update.md").write_text(
        "Context: ported celu.\n" + ("filler line.\n" * 20)
    )
    err = _findings_gate_check(ws)
    assert err is not None
    assert "Findings" in err


# ── all 3 deliverables together → one-pass promotion-eligible ───────────────
def test_all_three_deliverables_present_clears_all_gates(tmp_path):
    """A workspace shaped exactly as the brief mandates clears all 3 gates —
    i.e. finalize promotes in ONE pass, no one-by-one gate churn.
    """
    ws = _make_port_a3_ws(tmp_path)
    _write_op_host(ws, n_files=4)
    vj = _vj_pass(_compiled_evidence(ws))
    (ws / "verification.json").write_text(json.dumps(vj))
    (ws / "knowledge_update.md").write_text(_FINDINGS_DOC)

    assert getattr(fp, '_check_op_host_completeness')(ws) is None         # ①
    assert getattr(fp, '_check_binary_provenance')(ws, vj) is None        # ②
    assert _findings_gate_check(ws) is None                   # ③
