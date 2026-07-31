# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Per-plugin unit and mutual-exclusion coverage for backward generation.

Proves the plugin is scoped and composable:
- backward is discovered + registered with the right identity
- detect() fires ONLY on opgen_mode == "backward" (state-driven)
- detect() does NOT claim unrelated or empty workspaces
- adding backward does NOT change other plugins' detect() verdicts
- detect_plugin() resolves a backward workspace uniquely (no mutex violation)
- every hook still inherits BasePlugin neutral defaults (no premature behavior)

Layer 1 (protocol conformance) + Layer 7 (core mode-name-free) are covered
automatically by plugins/tests/test_protocol_conformance.py (parametrized over
all_plugins(), so backward is included once registered) and
plugins/tests/test_anti_regression.py (scans only core files, which this PR
does not touch).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent.parent.parent  # src/scripts/orchestrator/
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

from plugins import all_plugins, get_plugin, detect_plugin  # noqa: E402


def _write_state(ws: Path, **fields) -> Path:
    ws.mkdir(parents=True, exist_ok=True)
    (ws / ".opgen_state.json").write_text(json.dumps({"schema_version": 1, **fields}))
    return ws


# ── Identity / registration ─────────────────────────────────────────────

def test_backward_registered_with_identity():
    plug = get_plugin("backward")
    assert plug is not None, f"backward not registered: {[p.name for p in all_plugins()]}"
    assert plug.name == "backward"
    assert plug.cli_flag == "--backward"


def test_backward_cli_flag_unique():
    """--backward must not collide with any existing plugin's cli_flag."""
    flags = [p.cli_flag for p in all_plugins() if p.cli_flag is not None]
    assert flags.count("--backward") == 1


# ── detect() positive ────────────────────────────────────────────────────

def test_detect_true_on_backward_state(tmp_path):
    ws = _write_state(tmp_path / "mul_grad", op="mul_grad", opgen_mode="backward")
    assert get_plugin("backward").detect(ws) is True


def test_detect_plugin_resolves_backward_uniquely(tmp_path):
    """Mutual-exclusion (Layer 3): a backward workspace resolves to exactly
    the backward plugin — no RuntimeError, no other plugin co-claims.
    """
    ws = _write_state(tmp_path / "add_grad", op="add_grad", opgen_mode="backward")
    found = detect_plugin(ws)
    assert found is not None and found.name == "backward"


# ── detect() negative ────────────────────────────────────────────────────

def test_detect_false_on_unsupported_mode(tmp_path):
    ws = _write_state(tmp_path / "13_Cat", op="13_Cat", opgen_mode="unsupported")
    assert get_plugin("backward").detect(ws) is False


def test_detect_false_on_port_a3(tmp_path):
    ws = _write_state(tmp_path / "elu", op="elu", opgen_mode="port_a3_to_a5")
    assert get_plugin("backward").detect(ws) is False


def test_detect_false_on_missing_state(tmp_path):
    ws = tmp_path / "no_state"
    ws.mkdir()
    assert get_plugin("backward").detect(ws) is False


def test_detect_false_on_malformed_state(tmp_path):
    ws = tmp_path / "bad_state"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text("{ not valid json")
    assert get_plugin("backward").detect(ws) is False


# ── Drop-in: adding backward doesn't change other plugins ────────────────

def test_other_plugins_unaffected_by_backward_workspace(tmp_path):
    """The migration plugin must not claim a backward workspace."""
    ws = _write_state(tmp_path / "mul_grad", op="mul_grad", opgen_mode="backward")
    assert get_plugin("port_a3_to_a5").detect(ws) is False


def test_backward_does_not_claim_other_modes(tmp_path):
    """backward must not claim any unrelated persisted mode."""
    bw = get_plugin("backward")
    for mode in ("unscoped", "port_a3_to_a5", "unsupported"):
        ws = _write_state(tmp_path / f"ws_{mode}", op="x", opgen_mode=mode)
        assert bw.detect(ws) is False, f"backward wrongly claimed {mode} workspace"


# ── Scoped finalize hooks ────────────────────────────────────────────────

def test_backward_finalize_hooks_fail_closed(tmp_path):
    bw = get_plugin("backward")
    ws = _write_state(tmp_path / "mul_grad", op="mul_grad", opgen_mode="backward")
    vj = {"precision": {"status": "PASS"}}
    assert bw.verify_files() == ()
    assert bw.forbidden_patterns() == []
    assert ".so" in bw.check_binary_provenance(ws, vj)
    assert "verifier missing" in bw.check_verify_path_provenance(ws, vj)
    assert bw.archive_layout_mapping(ws) == {}
    assert bw.archive_project_subdir() == "backward_ops"
    assert bw.kw_brief_phase_a() is None
    assert bw.kw_brief_phase_d() is None


# ── FA-class BACKWARD stitch routing (2026-06-20) ────────────────────────

def _write_classification(ws: Path, tags: list) -> None:
    (ws / "op_classification.json").write_text(json.dumps({"op_class_tags": tags}))


def test_fa_grad_kw_brief_defaults_to_multilaunch(tmp_path):
    """Architecture default (C19, 2026-06-20): an attention-family backward op
    DEFAULTS to the proven MULTI-LAUNCH brief (CAND-FA-GQA-BWD-1,
    precision-core-complete), NOT the fused single-launch stitch — while keeping
    the self-contained autograd verify + the B3.3b finalize SCHEMA CONTRACT.
    """
    bw = get_plugin("backward")
    ws = _write_state(tmp_path / "fag", op="flash_attention_score_grad",
                      opgen_mode="backward")
    _write_classification(ws, ["ATTENTION", "FUSED", "SOFTMAX", "GRADIENT"])
    brief = bw.kw_brief_phase_block(op="flash_attention_score_grad",
                                    workspace=ws, iter_cap_remaining=10)
    # DEFAULT = the multi-launch brief, NOT the fused stitch.
    assert "MULTI-LAUNCH default" in brief
    assert "CAND-FA-GQA-BWD-1" in brief
    assert "FA-class BACKWARD template-stitch" not in brief  # NOT the fused stitch
    assert "IterateMmDyV" not in brief  # the 5-GEMM fused-cube block is fused-only
    # The multi-launch brief comes BEFORE the cold-start verify/finalize PHASES.
    assert (brief.index("MULTI-LAUNCH default")
            < brief.index("# PHASES (cold-start aog-kernel-worker — BACKWARD generation"))
    # The B3.3b finalize schema contract is STILL present (not dropped).
    assert "B3.3b FINALIZE SCHEMA CONTRACT" in brief
    assert "tier1_pass" in brief
    assert "grade_cases" in brief  # cannbot single judge


def test_fa_grad_kw_brief_fused_opt_in(tmp_path):
    """The fused single-launch stitch is gated behind an EXPLICIT opt-in
    (`fa_backward_arch=="fused"`) — only then does the kw get the P-P103 stitch
    recipe instead of the multi-launch default.
    """
    bw = get_plugin("backward")
    ws = _write_state(tmp_path / "fag_fused", op="flash_attention_score_grad",
                      opgen_mode="backward", fa_backward_arch="fused")
    _write_classification(ws, ["ATTENTION", "FUSED", "SOFTMAX", "GRADIENT"])
    brief = bw.kw_brief_phase_block(op="flash_attention_score_grad",
                                    workspace=ws, iter_cap_remaining=10)
    # OPT-IN = the fused stitch recipe, NOT the multi-launch default.
    assert "FA-class BACKWARD template-stitch" in brief
    assert "IterateMmDyV" in brief  # 5-GEMM cube block (stitch recipe)
    assert "NO CORE-FILL" in brief  # the graybox asset-gap clarification
    assert "MULTI-LAUNCH default, the precision" not in brief
    assert "B3.3b FINALIZE SCHEMA CONTRACT" in brief
    assert "copying source blocks or lines" in brief
    assert "allowed-input provenance review" in brief
    assert "logs every target/prior advisory read" in brief
    assert "use of such context as truth" in brief
    assert "COPY the CANN" not in brief


def test_fa_grad_kw_brief_fused_opt_in_via_large_s(tmp_path):
    """`fa_backward_large_s` also opts into the fused stitch (the regime where the
    fused MIX *may* amortize its overhead — hypothesis-flagged in KB).
    """
    bw = get_plugin("backward")
    ws = _write_state(tmp_path / "fag_ls", op="flash_attention_score_grad",
                      opgen_mode="backward", fa_backward_large_s=True)
    _write_classification(ws, ["ATTENTION", "FUSED", "SOFTMAX", "GRADIENT"])
    brief = bw.kw_brief_phase_block(op="flash_attention_score_grad",
                                    workspace=ws, iter_cap_remaining=10)
    assert "FA-class BACKWARD template-stitch" in brief
    assert "MULTI-LAUNCH default, the precision" not in brief


def test_non_fa_backward_kw_brief_unchanged(tmp_path):
    """A non-attention backward op (rms_norm_grad) gets NO stitch prefix — the
    analytic-derive brief is unchanged (no mis-route, no regression).
    """
    bw = get_plugin("backward")
    ws = _write_state(tmp_path / "rng", op="rms_norm_grad", opgen_mode="backward")
    _write_classification(ws, ["ELEMENTWISE", "GRADIENT"])
    brief = bw.kw_brief_phase_block(op="rms_norm_grad", workspace=ws,
                                    iter_cap_remaining=10)
    assert "FA-class BACKWARD template-stitch" not in brief
    assert brief.startswith("# PHASES (cold-start aog-kernel-worker — BACKWARD generation")
    assert "B3.3b FINALIZE SCHEMA CONTRACT" in brief
