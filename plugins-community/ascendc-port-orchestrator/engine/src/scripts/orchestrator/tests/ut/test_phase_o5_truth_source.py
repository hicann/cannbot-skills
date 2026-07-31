# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""W6 (2026-05-12, ROADMAP §1.5) — phase_o5 truth-source observability tests.

Validates that:
- phase_o05.init_durable_state writes opgen_mode to .opgen_state.json
- phase_o05.init_durable_state requires a supported explicit mode
- phase_o05.init_durable_state rejects mode changes on refresh
- phase_o5.expected_truth_source returns "a3_cann" for port mode
- phase_o5.expected_truth_source returns "backward_autograd" for backward mode
- missing, malformed, and unsupported state fails closed
- O5Report.truth_source field populated in all branches (SKIPPED, RUNNER_FAILED,
  VERIFIED, MISMATCH)
- post_verify cannot mint a verdict without supported workflow state
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import phase_o05  # noqa: E402
import phase_o5  # noqa: E402


# ---------------------------------------------------------------------------
# phase_o05: opgen_mode persistence
# ---------------------------------------------------------------------------
def test_init_durable_state_requires_explicit_mode(tmp_path):
    with pytest.raises(ValueError, match="must be explicit"):
        phase_o05.init_durable_state(tmp_path, "13_Cat", lane=0, target="a5")


def test_init_durable_state_writes_opgen_mode_port_a3(tmp_path):
    """opgen_mode='port_a3_to_a5' persisted."""
    rep = phase_o05.init_durable_state(
        tmp_path, "ctc_loss_v3", lane=0, target="a5",
        opgen_mode="port_a3_to_a5",
    )
    state = json.loads((tmp_path / ".opgen_state.json").read_text())
    assert state["opgen_mode"] == "port_a3_to_a5"
    assert rep.verdict == "INITIALIZED"


def test_init_durable_state_rejects_mode_change(tmp_path):
    phase_o05.init_durable_state(tmp_path, "op", opgen_mode="backward")
    with pytest.raises(ValueError, match="workspace mode conflict"):
        phase_o05.init_durable_state(
            tmp_path, "op", opgen_mode="port_a3_to_a5"
        )


def test_init_durable_state_rejects_unsupported_mode(tmp_path):
    with pytest.raises(ValueError, match="must be explicit"):
        phase_o05.init_durable_state(tmp_path, "13_Cat", opgen_mode="unsupported")


# ---------------------------------------------------------------------------
# expected_truth_source helper
# ---------------------------------------------------------------------------
def test_expected_truth_source_port_a3(tmp_path):
    """opgen_mode=port_a3_to_a5 → a3_cann."""
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({"op": "ctc_loss_v3", "opgen_mode": "port_a3_to_a5"})
    )
    assert phase_o5.expected_truth_source(tmp_path) == "a3_cann"


def test_expected_truth_source_backward(tmp_path):
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({"op": "mul_grad", "opgen_mode": "backward"})
    )
    assert phase_o5.expected_truth_source(tmp_path) == "backward_autograd"


def test_expected_truth_source_no_state_file(tmp_path):
    with pytest.raises(RuntimeError, match="cannot resolve supported truth source"):
        phase_o5.expected_truth_source(tmp_path)


def test_expected_truth_source_malformed_state(tmp_path):
    (tmp_path / ".opgen_state.json").write_text("not valid json {")
    with pytest.raises(RuntimeError, match="cannot resolve supported truth source"):
        phase_o5.expected_truth_source(tmp_path)


def test_expected_truth_source_missing_mode_field(tmp_path):
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({"op": "13_Cat", "schema_version": 1})
    )
    with pytest.raises(RuntimeError, match="cannot resolve supported truth source"):
        phase_o5.expected_truth_source(tmp_path)


# ---------------------------------------------------------------------------
# O5Report.truth_source propagation in post_verify_for_finalize
# ---------------------------------------------------------------------------
def test_post_verify_skip_carries_truth_source(tmp_path):
    """SKIPPED path still annotates truth_source."""
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({"op": "op", "opgen_mode": "port_a3_to_a5"})
    )
    rep = phase_o5.post_verify_for_finalize(tmp_path, "op", skip=True)
    assert rep.verdict == "SKIPPED"
    assert rep.truth_source == "a3_cann"


def test_post_verify_missing_verification_json_carries_truth_source(tmp_path):
    """RUNNER_FAILED (missing verification.json) still annotates truth_source."""
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({"op": "op", "opgen_mode": "port_a3_to_a5"})
    )
    rep = phase_o5.post_verify_for_finalize(tmp_path, "op")
    assert rep.verdict == "RUNNER_FAILED"
    assert rep.truth_source == "a3_cann"


def test_post_verify_malformed_verification_json(tmp_path):
    """RUNNER_FAILED (malformed JSON) still annotates truth_source."""
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({"op": "op", "opgen_mode": "port_a3_to_a5"})
    )
    (tmp_path / "verification.json").write_text("not json {")
    rep = phase_o5.post_verify_for_finalize(tmp_path, "op")
    assert rep.verdict == "RUNNER_FAILED"
    assert rep.truth_source == "a3_cann"


def test_post_verify_verified_path_with_a3_cann(tmp_path):
    """Full VERIFIED path: port mode → truth_source=a3_cann in report."""
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({"op": "op", "opgen_mode": "port_a3_to_a5"})
    )
    # Worker claimed N/A passes — early exit "no verifiable passes" path
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {
            "pass_a": {"status": "N/A"},
            "pass_b": {"status": "N/A"},
        }
    }))
    rep = phase_o5.post_verify_for_finalize(tmp_path, "op")
    assert rep.verdict == "VERIFIED"  # nothing to verify, vacuous PASS
    assert rep.truth_source == "a3_cann"


def test_post_verify_backward_all_na_fails_closed(tmp_path):
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({"op": "mul_grad", "opgen_mode": "backward"})
    )
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {
            "pass_a": {"status": "N/A"},
            "pass_b": {"status": "N/A"},
        }
    }))
    rep = phase_o5.post_verify_for_finalize(tmp_path, "mul_grad")
    assert rep.verdict == "RUNNER_FAILED"
    assert rep.truth_source == "backward_autograd"


def test_post_verify_without_scoped_state_fails_closed(tmp_path):
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {"pass_a": {"status": "N/A"}, "pass_b": {"status": "N/A"}}
    }))
    with pytest.raises(RuntimeError, match="cannot resolve supported truth source"):
        phase_o5.post_verify_for_finalize(tmp_path, "op")
