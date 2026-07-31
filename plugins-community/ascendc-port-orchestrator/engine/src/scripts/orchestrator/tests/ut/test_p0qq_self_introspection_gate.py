# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P0qq (2026-05-06): self-introspection in-context gate.

Plan: docs/design/CONTRACT_AND_MATURITY_NOTES.md#fsm-phase-gap-fix-plan Step 8.

User correction (2026-05-06): the previous aog-self-critic skill spawned
in an isolated context — fresh model, no access to the agent's reasoning
trace — which is post-hoc audit, not introspection. Real self-critic =
same model in same context reflecting on its own work BEFORE emitting a
terminal verdict.

This test suite verifies:
  1. Each of the 6 brief builders embeds the self_introspection_block().
  2. schema_norm rejects `done` / `partial_persist` handoffs whose
     workspace PROGRESS.md lacks `## Self-introspection`.
  3. schema_norm rejects partial Self-introspection (heading present but
     subsections missing).
  4. schema_norm passes when all 4 required subsections are present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from briefs._common import self_introspection_block  # noqa: E402
from briefs.kw_brief import build_worker_brief  # noqa: E402
from briefs.pp_brief import build_probe_brief  # noqa: E402
from briefs.ko_brief import build_optimizer_brief  # noqa: E402
from briefs.fo_brief import build_fused_optimizer_brief  # noqa: E402
from briefs.ar_brief import build_researcher_brief  # noqa: E402
from briefs.da_brief import build_det_analyzer_brief  # noqa: E402
import schema_norm  # noqa: E402


# ---------------------------------------------------------------------------
# Part 1: brief builders embed self_introspection_block()
# ---------------------------------------------------------------------------


def _fake_env():
    from briefs._common import AscendCEnv
    return AscendCEnv(
        target="a5",
        host="x", user="y", password="z", container="c",
        cann_path="/p", soc_version="Ascend950PR",
        benchmark_root="/b", local_benchmark="/lb", local_project="/lp",
        archive_project="a3_to_a5_port", build_archive_enabled=False,
        opgen_mode="port_a3_to_a5",
        port_a3_source="/fixture/arch22/test_op",
    )


@pytest.mark.parametrize("builder_name,builder", [
    ("kw", build_worker_brief),
    ("pp", build_probe_brief),
    ("ko", build_optimizer_brief),
    ("fo", build_fused_optimizer_brief),
    ("ar", build_researcher_brief),
    ("da", build_det_analyzer_brief),
])
def test_each_brief_contains_introspection_block(builder_name, builder, tmp_path):
    """All 6 brief builders must include the self_introspection_block."""
    out = builder(
        op="test_op",
        workspace=tmp_path,
        lane=0,
        spawn_index=1,
        iter_cap_remaining=4,
        env=_fake_env(),
    )
    assert "SELF-INTROSPECTION CHECKPOINT" in out, (
        f"{builder_name}_brief is missing the introspection checkpoint"
    )
    assert "## Self-introspection" in out
    # Must list the 4 required subsection names
    assert "Pressure modes I felt" in out
    assert "Decisions I almost rationalized" in out
    assert "Verifications I might have skipped" in out
    assert "Confidence calibration" in out


def test_introspection_block_has_required_subsections():
    """The block itself defines the canonical 4 subsections."""
    text = self_introspection_block()
    for sub in (
        "Pressure modes I felt",
        "Decisions I almost rationalized",
        "Verifications I might have skipped",
        "Confidence calibration",
    ):
        assert sub in text


def test_introspection_block_states_in_context_constraint():
    """The block must explicitly state it's NOT a separate spawn."""
    text = self_introspection_block()
    assert "NOT a separate spawn" in text
    assert "SAME\ncontext" in text or "SAME context" in text


# ---------------------------------------------------------------------------
# Part 2: schema_norm gate rejects terminal handoffs without introspection
# ---------------------------------------------------------------------------


def _seed_passing_verification(workspace: Path):
    """Helper: write verification.json with PASS so the only gate that can
    fail is the introspection gate."""
    vj = {
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": 31, "total": 31},
            "pass_b": {"status": "PASS", "tier1_pass": 11, "total": 11},
        },
        "performance": {"status": "PASS", "ratio": 1.05,
                        "independent_re_measure": {"ran": True, "ratio": 1.05, "delta_vs_kw_self_report": 0.0}},
    }
    (workspace / "verification.json").write_text(json.dumps(vj))
    # P0aay (2026-05-11): seed knowledge_update.md for pre-handoff gate
    (workspace / "knowledge_update.md").write_text(
        "## Context\nTest stub.\n\n## Findings\n- Stub\n\n"
        "## KB-promotable patterns (proposed)\nNone\n\n"
        "## Cited KB items\nNone\n\n## Anti-patterns avoided\nNone\n"
    )


def test_terminal_done_rejected_when_progress_missing(tmp_path):
    """`done` handoff with no PROGRESS.md → reject."""
    _seed_passing_verification(tmp_path)
    res = getattr(schema_norm, '_check_evidence_for_terminal')(
        tmp_path, "done", "finalize", entry={}
    )
    assert res["passes"] is False
    assert "Self-introspection" in res["reason"] or "PROGRESS.md" in res["reason"]


def test_terminal_done_rejected_when_introspection_heading_missing(tmp_path):
    """PROGRESS.md present but no `## Self-introspection` heading → reject."""
    _seed_passing_verification(tmp_path)
    (tmp_path / "PROGRESS.md").write_text("# op log\n\nsome notes\n")
    res = getattr(schema_norm, '_check_evidence_for_terminal')(
        tmp_path, "done", "finalize", entry={}
    )
    assert res["passes"] is False
    assert "Self-introspection" in res["reason"]
    assert "P0qq" in res["reason"]


def test_terminal_done_rejected_when_subsections_partial(tmp_path):
    """Heading present but only 2 of 4 subsections → reject with named missing."""
    _seed_passing_verification(tmp_path)
    (tmp_path / "PROGRESS.md").write_text(
        "# op\n\n## Self-introspection (op-kw-1)\n\n"
        "### Pressure modes I felt\nP1, P3.\n\n"
        "### Decisions I almost rationalized\nfoo\n\n"
        # Missing: Verifications I might have skipped, Confidence calibration
    )
    res = getattr(schema_norm, '_check_evidence_for_terminal')(
        tmp_path, "done", "finalize", entry={}
    )
    assert res["passes"] is False
    assert "Verifications I might have skipped" in res["reason"]
    assert "Confidence calibration" in res["reason"]


def test_terminal_done_passes_when_introspection_complete(tmp_path):
    """All 4 subsections present + verification PASS → introspection gate
    accepts (so downstream precision/perf gates can run).
    """
    _seed_passing_verification(tmp_path)
    (tmp_path / "PROGRESS.md").write_text(
        "# op\n\n## Self-introspection (op-kw-1)\n\n"
        "### Pressure modes I felt\nP1, P5.\n\n"
        "### Decisions I almost rationalized\nbar\n\n"
        "### Verifications I might have skipped\nnone — ran all\n\n"
        "### Confidence calibration\nprecision: HIGH\nperf: HIGH\narchitectural fit: HIGH\n"
    )
    res = getattr(schema_norm, '_check_evidence_for_terminal')(
        tmp_path, "done", "finalize", entry={}
    )
    # Should pass now (precision PASS + perf 1.05 ≥ parity + introspection complete)
    assert res["passes"] is True


def test_terminal_partial_persist_also_gated_on_introspection(tmp_path):
    """Same gate applies to partial_persist alias."""
    vj = {
        "precision": {"status": "PARTIAL"},
        "performance": {"status": "PASS", "ratio": 0.7},
    }
    (tmp_path / "verification.json").write_text(json.dumps(vj))
    (tmp_path / "probe_report.md").write_text("x" * 200)  # evidence
    # PROGRESS.md WITHOUT introspection
    (tmp_path / "PROGRESS.md").write_text("# op\nno introspection here\n")
    res = getattr(schema_norm, '_check_evidence_for_terminal')(
        tmp_path, "partial_persist", "finalize", entry={}
    )
    assert res["passes"] is False
    assert "Self-introspection" in res["reason"]


def test_introspection_helper_exists_and_returns_dict():
    """Direct call to _check_self_introspection should return passes/reason."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        res = getattr(schema_norm, '_check_self_introspection')(ws)
        assert "passes" in res
        assert "reason" in res
        assert res["passes"] is False  # no PROGRESS.md
