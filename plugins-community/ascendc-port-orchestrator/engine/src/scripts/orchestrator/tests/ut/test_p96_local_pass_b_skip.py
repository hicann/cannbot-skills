# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-172 (2026-06-24) regression — _run_verifier_local P96 plugin-skip.

Regression for the harness gap surfaced by top_k_top_p_sample kw-2:
_run_verifier_local (CONTAINER=local path) was missing the P96 plugin-skip
logic that ssh_runner has. A port_a3 op on a local container would fail with
"no Pass B verifier found locally" even when verification.json correctly
declares pass_b.status=N/A per the port_a3 D.6 brief contract.

This test verifies:
1. _run_verifier_local SKIPS pass_b (returns SKIPPED dict) when:
   - plugin is port_a3 (pass_b_required()=False), AND
   - verification.json has precision.pass_b.status=N/A
2. _run_verifier_local STILL fails when pass_b is required (non-port_a3)
   and no verifier file exists (guardrail: doesn't silently skip real ops)
3. _run_canonical_pass_a_local SKIPS for port_a3 (returns skip-reason string)

See:
- phase_o5_runner.py::_run_verifier_local (P96 plugin-skip, DEBT-172 2026-06-24)
- phase_o5_runner.py::_run_canonical_pass_a_local (plugin canonical-skip)
- CLAUDE.md "Fix Harness for Next Customer" rule
- top_k_top_p_sample kw-2 post-mortem
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))


def _make_workspace(tmp_path: Path, pass_b_status: str = "N/A") -> Path:
    """Create a minimal workspace with verification.json."""
    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "verification.json").write_text(json.dumps({
        "precision": {
            "pass_a": {"status": "PASS", "tier1_pass": 10, "total": 10},
            "pass_b": {"status": pass_b_status, "reason": "test"},
        }
    }))
    (ws / "model.py").write_text("# stub")
    (ws / "model_new_ascendc.py").write_text("# stub")
    return ws


def _mock_port_a3_plugin():
    """Mock plugin that mimics PortA3Plugin's pass_b_required()=False."""
    p = MagicMock()
    p.name = "port_a3_to_a5"
    p.pass_b_required.return_value = False
    p.canonical_pass_a_skip_reason.return_value = "port_a3 native two-tier skip"
    return p


def test_local_verifier_skips_pass_b_for_port_a3(tmp_path):
    """port_a3 op on local container: pass_b SKIPPED, no file search."""
    import phase_o5_runner

    ws = _make_workspace(tmp_path, pass_b_status="N/A")
    mock_plugin = _mock_port_a3_plugin()

    with patch("plugins.detect_plugin", return_value=mock_plugin), \
         patch.object(phase_o5_runner, "_run_canonical_pass_a_local",
                      return_value=None):
        result = getattr(phase_o5_runner, '_run_verifier_local')(
            ws, "test_op", {"A5_CONTAINER": "local"}, lane=0
        )
    assert result.runner_error is None, \
        f"Expected pass_b SKIPPED, got runner_error={result.runner_error!r}"
    assert result.pass_b is not None
    assert result.pass_b.get("status") == "SKIPPED"


def test_local_verifier_fails_pass_b_when_required(tmp_path):
    """Non-port_a3 op (pass_b required): still fails without verifier file."""
    import phase_o5_runner

    ws = _make_workspace(tmp_path, pass_b_status="PASS")

    # Mock supported plugin that requires pass_b.
    mock_plugin = MagicMock()
    mock_plugin.name = "backward"
    mock_plugin.pass_b_required.return_value = True

    with patch("plugins.detect_plugin", return_value=mock_plugin):
        result = getattr(phase_o5_runner, '_run_verifier_local')(
            ws, "test_op", {"A5_CONTAINER": "local"}, lane=0
        )
    assert result.runner_error is not None
    assert "no Pass B verifier found locally" in result.runner_error


def test_local_verifier_fails_when_pass_b_not_na(tmp_path):
    """port_a3 op but pass_b.status != N/A: should still look for file."""
    import phase_o5_runner

    ws = _make_workspace(tmp_path, pass_b_status="PASS")
    mock_plugin = _mock_port_a3_plugin()

    with patch("plugins.detect_plugin", return_value=mock_plugin):
        result = getattr(phase_o5_runner, '_run_verifier_local')(
            ws, "test_op", {"A5_CONTAINER": "local"}, lane=0
        )
    # pass_b.status=PASS (not N/A) → plugin-skip does NOT fire → no file → error
    assert result.runner_error is not None
    assert "no Pass B verifier found locally" in result.runner_error


def test_canonical_pass_a_local_is_harness_authoritative_for_port_a3(tmp_path):
    """① (owner 2026-06-30, supersedes the old DEBT-172 skip): _run_canonical_pass_a_local no longer
    returns a SKIP string for port_a3 — it runs the HARNESS grader precision_eval_port_a3_two_tier
    (compare.py 生态) as AUTHORITATIVE and returns a DICT verdict. It does NOT run the generic
    benchmark evaluator (precision_eval_two_tier). With no capture tensors → an honest FAIL dict
    (no fake pass), proving the harness — not the worker — produces the verdict.
    """
    import phase_o5_runner

    ws = _make_workspace(tmp_path, pass_b_status="N/A")
    mock_plugin = _mock_port_a3_plugin()

    with patch("plugins.detect_plugin", return_value=mock_plugin):
        result = getattr(phase_o5_runner, '_run_canonical_pass_a_local')(
            ws, "test_op", {"A5_CONTAINER": "local"}, lane=0
        )
    # ①: a DICT verdict from the port_a3 harness grader (NOT a skip string, NOT the generic normalizer)
    assert isinstance(result, dict), f"expected harness verdict dict, got {result!r}"
    assert result.get("method") == "canonical_precision_eval_port_a3_two_tier"
    # no capture tensors in this minimal workspace → honest FAIL (never a fabricated pass)
    assert result.get("status") == "FAIL"


def test_local_pass_a_normalization_carries_tier2_status():
    """Gap 3 regression: _normalize_verifier_output with label "pass_a"
    (NOT "pass_a_local") triggers port_a3 two-tier normalization so
    tier2_status is preserved in the output.
    """
    import phase_o5_runner

    # Simulate worker runner output with tier2_status
    parsed = {
        "tier1_pass": 10,
        "tier2_pass": 0,
        "tier1_pass_inclusive": 10,
        "total": 10,
        "tier2_status": "A3_UNAVAILABLE",
        "status": "PASS",
    }

    # With label "pass_a" (the fixed value) — should trigger port_a3 normalization
    result = getattr(phase_o5_runner, '_normalize_verifier_output')(parsed, "pass_a")
    assert result.get("tier2_status") == "A3_UNAVAILABLE", \
        f"tier2_status missing from normalized output: {result}"
    assert result.get("tier1_pass_inclusive") == 10

    # Verify the OLD broken label would have dropped it (guardrail)
    result_broken = getattr(phase_o5_runner, '_normalize_verifier_output')(parsed, "pass_a_local")
    assert result_broken.get("tier2_status") is None, \
        "Expected tier2_status to be dropped with 'pass_a_local' label (the bug)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
