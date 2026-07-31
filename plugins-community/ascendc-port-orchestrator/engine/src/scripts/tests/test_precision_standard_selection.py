# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""controllable-harness #4 — explicit precision-standard selection (2026-07-04).

THE INVARIANT (docs/design/CONTROLLABLE_HARNESS_DESIGN.md): the precision standard
(生态 ecosystem | 商用 commercial) is an EXPLICIT, selectable choice that MUST be
(1) provenance-recorded, (2) fail-loud when ambiguous/invalid, (3) never a silent
coverage-narrowing knob. These tests lock in the resolver precedence, the fail-loud
cases, and that the chosen standard is recorded + threaded to the grader route.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))                 # src/scripts
sys.path.insert(0, str(_HERE.parent.parent / "orchestrator"))  # orchestrator

import precision_eval_port_a3_two_tier as pa3  # noqa: E402
from a3_ref_validate import write_a3_capture_provenance  # noqa: E402
from source_arch import stage_source_tree  # noqa: E402


def _seed_live_capture(workspace: Path, source_output) -> None:
    """Create the minimum harness-owned source-NPU provenance for the grader."""
    source = workspace.parent / "source_op"
    kernel = source / "op_kernel" / "arch22"
    kernel.mkdir(parents=True)
    (kernel / "op.h").write_text("class SourceOp { void Process() {} };\n")
    stage = stage_source_tree(source, workspace)
    (workspace / ".opgen_state.json").write_text(json.dumps({
        "schema_version": 2,
        "opgen_mode": "port_a3_to_a5",
        "source_arch": "arch22",
        "target_arch": "arch35",
        "port_a3_source": str(stage.root),
        "graybox_arch22_dir": str(stage.root),
        "graybox_sandbox": True,
        "source_stage_manifest": str(stage.manifest),
        "source_stage_digest": stage.digest,
    }))
    torch = pytest.importorskip("torch")
    torch.save([None], workspace / "edge_inputs.pt")
    torch.save(
        [{"case_id": 0, "inputs": None, "a3_outputs": source_output}],
        workspace / "edge_dataset.pt",
    )
    (workspace / "a3_baseline_perf.json").write_text(json.dumps({
        "median_ms_per_case": {"case_0": 1.0},
    }))
    (workspace / "run_a3_reference.py").write_text("# harness test runner\n")
    ok, reason, manifest = write_a3_capture_provenance(
        workspace,
        capture_id="capture-test",
        capture_started_ts=datetime.now(timezone.utc).isoformat(),
        npu_id=0,
    )
    assert ok, reason
    assert manifest is not None


# ---------------------------------------------------------------------------
# resolve_precision_standard — precedence
# ---------------------------------------------------------------------------
def test_default_is_ecosystem_recorded_not_silent(tmp_path):
    """Documented default may apply, but its source is RECORDED as 'default'."""
    std, src = pa3.resolve_precision_standard(tmp_path, cli_value=None, env={})
    assert std == "ecosystem"
    assert src == "default"


def test_cli_wins_over_env_and_file(tmp_path):
    (tmp_path / ".ascendc_env").write_text("PRECISION_STANDARD=ecosystem\n")
    std, src = pa3.resolve_precision_standard(
        tmp_path, cli_value="commercial", env={"PRECISION_STANDARD": "ecosystem"})
    assert (std, src) == ("commercial", "cli")


def test_env_used_when_no_file(tmp_path):
    """env is the authority when there is no .ascendc_env file (no conflict)."""
    std, src = pa3.resolve_precision_standard(
        tmp_path, env={"PRECISION_STANDARD": "commercial"})
    assert (std, src) == ("commercial", "env")


def test_ascendc_env_file_used_when_no_cli_or_env(tmp_path):
    # canonical file for a workspace is workspace.parent/.ascendc_env
    op_ws = tmp_path / "some_op"
    op_ws.mkdir()
    (tmp_path / ".ascendc_env").write_text("A3_HOST=x\nPRECISION_STANDARD=commercial\n")
    std, src = pa3.resolve_precision_standard(op_ws, env={})
    assert (std, src) == ("commercial", "ascendc_env")


# ---------------------------------------------------------------------------
# FAIL-LOUD cases (THE INVARIANT #2)
# ---------------------------------------------------------------------------
def test_invalid_cli_value_fails_loud(tmp_path):
    with pytest.raises(ValueError) as ei:
        pa3.resolve_precision_standard(tmp_path, cli_value="bogus", env={})
    assert "bogus" in str(ei.value)
    assert "ecosystem" in str(ei.value) and "commercial" in str(ei.value)


def test_invalid_env_value_fails_loud(tmp_path):
    with pytest.raises(ValueError) as ei:
        pa3.resolve_precision_standard(tmp_path, env={"PRECISION_STANDARD": "SGX"})
    assert "SGX" in str(ei.value)


def test_invalid_file_value_fails_loud(tmp_path):
    (tmp_path / ".ascendc_env").write_text("PRECISION_STANDARD=nonsense\n")
    with pytest.raises(ValueError) as ei:
        pa3.resolve_precision_standard(tmp_path, env={})
    assert "nonsense" in str(ei.value)


def test_conflicting_env_and_file_is_ambiguous_fails_loud(tmp_path):
    """env says commercial, file says ecosystem → ambiguous → refuse (don't guess)."""
    (tmp_path / ".ascendc_env").write_text("PRECISION_STANDARD=ecosystem\n")
    with pytest.raises(ValueError) as ei:
        pa3.resolve_precision_standard(tmp_path, env={"PRECISION_STANDARD": "commercial"})
    msg = str(ei.value)
    assert "AMBIGUOUS" in msg or "ambiguous" in msg.lower()


def test_matching_env_and_file_is_not_ambiguous(tmp_path):
    """Same value in both authorities is fine — env wins, no error."""
    (tmp_path / ".ascendc_env").write_text("PRECISION_STANDARD=commercial\n")
    std, src = pa3.resolve_precision_standard(
        tmp_path, env={"PRECISION_STANDARD": "commercial"})
    assert std == "commercial"


# ---------------------------------------------------------------------------
# Threading to the grader route + recorded in summary
# ---------------------------------------------------------------------------
def test_load_and_classify_records_precision_standard(tmp_path):
    """load_and_classify records precision_standard + source even on the missing-file
    FAIL path is NOT required; here we assert an invalid explicit value fails loud.
    """
    with pytest.raises(ValueError):
        pa3.load_and_classify(tmp_path, verbose=False, precision_standard="oops")


def test_load_and_classify_threads_and_records(tmp_path):
    torch = pytest.importorskip("torch")
    workspace = tmp_path / "op"
    workspace.mkdir()
    # Minimal aligned captures: target == source-NPU truth. The selected
    # compatibility standard is recorded but only controls CPU diagnostics.
    t = torch.ones(8, dtype=torch.float16)
    torch.save([t.clone()], workspace / "a5_capture.pt")
    _seed_live_capture(workspace, t.clone())
    torch.save({"golden_kind": "cpu_fp64", "outputs": [t.to(torch.float64)]},
               workspace / "cpu_truth_outputs.pt")
    s = pa3.load_and_classify(workspace, verbose=False,
                              precision_standard="ecosystem", precision_standard_source="cli")
    assert s["status"] == "PASS"
    assert s["precision_standard"] == "ecosystem"
    assert s["precision_standard_source"] == "cli"
    assert s["precision_standard_role"] == "cpu_diagnostic_only"
    assert s["primary_truth"] == "source_npu_arch22"
    assert "fresh arch22 NPU truth" in s["grader"]


def test_main_ambiguous_standard_returns_2(tmp_path, monkeypatch):
    """main() catches the fail-loud ValueError (ambiguous env vs file) → rc 2 (not a crash)."""
    (tmp_path / ".ascendc_env").write_text("PRECISION_STANDARD=ecosystem\n")
    monkeypatch.setenv("PRECISION_STANDARD", "commercial")
    op_ws = tmp_path / "op"
    op_ws.mkdir()
    rc = pa3.main([str(op_ws), "--quiet"])
    assert rc == 2
