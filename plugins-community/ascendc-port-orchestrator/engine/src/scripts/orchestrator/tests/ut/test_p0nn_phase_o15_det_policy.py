# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P0nn (2026-05-06): Phase O1.5 DET_POLICY classification.

Plan: docs/design/CONTRACT_AND_MATURITY_NOTES.md#fsm-phase-gap-fix-plan Step 5.

Classifies op's determinism policy from analysis.md or op_taxonomy tags.
Result stored in .opgen_state.json so downstream phases act on it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import phase_o05  # noqa: E402
import phase_o15  # noqa: E402


def test_explicit_override_wins(tmp_path):
    rep = phase_o15.classify_det_policy(tmp_path, "x", explicit="required")
    assert rep.policy == "required"
    assert rep.source == "explicit"


def test_invalid_explicit_falls_through(tmp_path):
    rep = phase_o15.classify_det_policy(tmp_path, "x", explicit="bogus")
    assert rep.policy == "best_effort"  # default


def test_analysis_md_required(tmp_path):
    (tmp_path / "analysis.md").write_text(
        "# 5_Cumsum analysis\n\nDET_POLICY: required\n"
    )
    rep = phase_o15.classify_det_policy(tmp_path, "5_Cumsum")
    assert rep.policy == "required"
    assert rep.source == "analysis_md"


def test_analysis_md_best_effort(tmp_path):
    (tmp_path / "analysis.md").write_text(
        "DET_POLICY: best_effort\n"
    )
    rep = phase_o15.classify_det_policy(tmp_path, "x")
    assert rep.policy == "best_effort"


def test_analysis_md_n_a_variations(tmp_path):
    """Tolerant of n/a, n_a, na variants."""
    for variant in ("n/a", "n_a", "na"):
        (tmp_path / "analysis.md").write_text(f"DET_POLICY: {variant}\n")
        rep = phase_o15.classify_det_policy(tmp_path, "x")
        assert rep.policy == "n_a", f"variant {variant!r} → {rep.policy}"


def test_analysis_md_case_insensitive(tmp_path):
    (tmp_path / "analysis.md").write_text("det policy: REQUIRED\n")
    rep = phase_o15.classify_det_policy(tmp_path, "x")
    assert rep.policy == "required"


def test_analysis_md_no_marker_falls_through(tmp_path):
    (tmp_path / "analysis.md").write_text("# random analysis text\n")
    rep = phase_o15.classify_det_policy(tmp_path, "x")
    assert rep.policy == "best_effort"
    assert rep.source == "default"


def test_op_taxonomy_tag_norm_backward_required(tmp_path):
    rep = phase_o15.classify_det_policy(tmp_path, "x",
                                          op_tags=["norm-backward"])
    assert rep.policy == "required"
    assert rep.source == "op_taxonomy"


def test_op_taxonomy_tag_path_a_n_a(tmp_path):
    rep = phase_o15.classify_det_policy(tmp_path, "x",
                                          op_tags=["path-a-cpu-truth"])
    assert rep.policy == "n_a"


def test_op_taxonomy_tag_unknown_defaults(tmp_path):
    rep = phase_o15.classify_det_policy(tmp_path, "x",
                                          op_tags=["unknown-tag"])
    assert rep.policy == "best_effort"
    assert rep.source == "default"


def test_priority_explicit_beats_analysis_md(tmp_path):
    (tmp_path / "analysis.md").write_text("DET_POLICY: required\n")
    rep = phase_o15.classify_det_policy(tmp_path, "x", explicit="best_effort")
    assert rep.policy == "best_effort"
    assert rep.source == "explicit"


def test_priority_analysis_md_beats_op_taxonomy(tmp_path):
    (tmp_path / "analysis.md").write_text("DET_POLICY: best_effort\n")
    rep = phase_o15.classify_det_policy(tmp_path, "x",
                                          op_tags=["norm-backward"])
    assert rep.policy == "best_effort"  # analysis.md wins


def test_store_in_durable_state(tmp_path):
    """classify → store → verify in .opgen_state.json"""
    phase_o05.init_durable_state(
        tmp_path, "test_op", opgen_mode="port_a3_to_a5"
    )
    rep = phase_o15.classify_det_policy(tmp_path, "test_op", explicit="required")
    phase_o15.store_in_durable_state(tmp_path, rep.policy)
    state = json.loads((tmp_path / phase_o05.STATE_FILE).read_text())
    assert state["det_policy"] == "required"


def test_store_without_phase_o05_noop(tmp_path):
    """store_in_durable_state shouldn't crash if .opgen_state.json missing."""
    phase_o15.store_in_durable_state(tmp_path, "required")  # no-op, no crash


def test_store_handles_malformed_state_gracefully(tmp_path):
    (tmp_path / phase_o05.STATE_FILE).write_text("{ malformed")
    phase_o15.store_in_durable_state(tmp_path, "required")  # no crash
