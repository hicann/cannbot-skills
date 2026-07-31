# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Fresh source-NPU truth tests for the arch22-to-arch35 precision gate."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_HERE = Path(__file__).resolve()
_SCRIPTS = _HERE.parent.parent
_ORCHESTRATOR = _SCRIPTS / "orchestrator"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_ORCHESTRATOR))

import precision_eval_port_a3_two_tier as pa3  # noqa: E402
from a3_ref_validate import write_a3_capture_provenance  # noqa: E402
from source_arch import stage_source_tree  # noqa: E402


def _make_live_workspace(
    tmp_path: Path,
    *,
    ours: list,
    source_outputs: list,
    cpu_outputs: list | None = None,
) -> Path:
    """Create a real staged-source state plus a harness-owned capture manifest."""
    source = tmp_path / "source_op"
    kernel = source / "op_kernel" / "arch22"
    kernel.mkdir(parents=True)
    (kernel / "op.h").write_text("class SourceOp { void Process() {} };\n")

    workspace = (tmp_path / "workspace" / "source_op").resolve()
    workspace.mkdir(parents=True)
    stage = stage_source_tree(source, workspace)
    state = {
        "schema_version": 2,
        "opgen_mode": "port_a3_to_a5",
        "source_arch": "arch22",
        "target_arch": "arch35",
        "port_a3_source": str(stage.root),
        "graybox_arch22_dir": str(stage.root),
        "graybox_sandbox": True,
        "source_stage_manifest": str(stage.manifest),
        "source_stage_digest": stage.digest,
    }
    (workspace / ".opgen_state.json").write_text(json.dumps(state) + "\n")

    inputs = [torch.zeros_like(output) for output in source_outputs]
    torch.save({"inputs": inputs}, workspace / "edge_inputs.pt")
    torch.save(
        {"inputs": inputs, "a3_outputs": source_outputs},
        workspace / "edge_dataset.pt",
    )
    (workspace / "a3_baseline_perf.json").write_text(
        json.dumps(
            {
                "median_ms_per_case": {
                    f"case_{index}": 1.0 for index in range(len(source_outputs))
                }
            }
        )
        + "\n"
    )
    (workspace / "run_a3_reference.py").write_text("# harness test runner\n")
    ok, reason, manifest = write_a3_capture_provenance(
        workspace,
        capture_id="capture-test",
        capture_started_ts=datetime.now(timezone.utc).isoformat(),
        npu_id=0,
    )
    assert ok, reason
    assert manifest is not None
    torch.save(ours, workspace / "a5_capture.pt")
    if cpu_outputs is not None:
        torch.save(cpu_outputs, workspace / "cpu_truth_outputs.pt")
    return workspace


def test_to_tensor_list_int_keyed_dict_yields_all_cases():
    dataset = {i: {"a3_outputs": torch.ones(2)} for i in range(43)}
    assert len(getattr(pa3, '_to_tensor_list')(dataset)) == 43


def test_coerce_case_list_preserves_documented_schemas():
    assert getattr(pa3, '_coerce_case_list')({0: "a", 1: "b"}) == ["a", "b"]
    documented = {"inputs": [1], "a3_outputs": [2]}
    assert getattr(pa3, '_coerce_case_list')(documented) is documented
    non_contiguous = {0: "a", 2: "b"}
    assert getattr(pa3, '_coerce_case_list')(non_contiguous) is non_contiguous


def test_source_npu_match_passes_even_when_cpu_disagrees():
    source = torch.ones(64, dtype=torch.float32)
    ours = source.clone()
    cpu = source + 100.0
    result = pa3.classify_port_a3_case(ours, source, cpu)
    assert result["verdict"] == "PASS_T1"
    assert result["tier1_pass"] is True
    assert result["primary_truth"] == "source_npu_arch22"
    assert result["cpu_truth_role"] == "diagnostic_only"


def test_source_npu_mismatch_fails_even_when_cpu_matches_target():
    ours = torch.ones(64, dtype=torch.float32)
    source = ours + 1.0
    cpu = ours.clone()
    result = pa3.classify_port_a3_case(ours, source, cpu)
    assert result["verdict"] == "FAIL"
    assert result["tier1_pass"] is False


def test_legacy_precision_choice_cannot_change_primary_verdict():
    ours = torch.ones(64, dtype=torch.float32)
    source = ours + 1.0
    cpu = ours.clone()
    ecosystem = pa3.classify_port_a3_case(ours, source, cpu, route="ecosystem")
    commercial = pa3.classify_port_a3_case(ours, source, cpu, route="commercial")
    assert ecosystem["verdict"] == commercial["verdict"] == "FAIL"


def test_cpu_truth_is_optional_for_source_npu_grade():
    source = torch.ones(64, dtype=torch.float16)
    result = pa3.classify_port_a3_case(source.clone(), source, None)
    assert result["verdict"] == "PASS_T1"
    assert "cpu_diagnostic" not in result


def test_missing_source_npu_tensor_is_evaluation_error():
    ours = torch.ones(8)
    result = pa3.classify_port_a3_case(ours, None, ours)
    assert result["verdict"] == "EVAL_ERR"
    assert result["tier2_status"] == "A3_UNAVAILABLE"


def test_genuine_source_may_be_bit_equal_to_cpu_diagnostic():
    value = torch.ones(8)
    result = pa3.classify_port_a3_case(value.clone(), value, value.clone())
    assert result["verdict"] == "PASS_T1"


def test_summarize_fails_on_any_source_truth_error():
    summary = pa3.summarize(
        [
            {
                "tier1_pass": True,
                "verdict": "PASS_T1",
                "tier2_status": "N/A_SOURCE_NPU_PRIMARY",
            },
            {
                "tier1_pass": False,
                "verdict": "EVAL_ERR",
                "tier2_status": "A3_UNAVAILABLE",
            },
        ]
    )
    assert summary["status"] == "FAIL"
    assert summary["n_err"] == 1
    assert summary["primary_truth"] == "source_npu_arch22"


def test_load_passes_without_cpu_or_native_capture(tmp_path):
    source = [torch.ones(32, dtype=torch.float16) for _ in range(2)]
    workspace = _make_live_workspace(
        tmp_path,
        ours=[item.clone() for item in source],
        source_outputs=source,
    )
    result = pa3.load_and_classify(workspace, verbose=False)
    assert result["status"] == "PASS"
    assert result["tier1_pass"] == 2
    assert result["truth_source"] == "fresh_live_arch22_npu_capture"
    assert result["capture_id"] == "capture-test"
    assert result["native_provision_ok"] is False


def test_load_cpu_diagnostic_cannot_turn_source_failure_into_pass(tmp_path):
    ours = [torch.ones(32)]
    workspace = _make_live_workspace(
        tmp_path,
        ours=ours,
        source_outputs=[ours[0] + 1.0],
        cpu_outputs=[ours[0].clone()],
    )
    result = pa3.load_and_classify(workspace, verbose=False)
    assert result["status"] == "FAIL"
    assert result["results"][0]["cpu_diagnostic"]["verdict"].startswith("PASS")


def test_load_precision_choice_is_diagnostic_only(tmp_path):
    ours = [torch.ones(32)]
    workspace = _make_live_workspace(
        tmp_path,
        ours=ours,
        source_outputs=[ours[0] + 1.0],
        cpu_outputs=[ours[0].clone()],
    )
    ecosystem = pa3.load_and_classify(
        workspace,
        verbose=False,
        precision_standard="ecosystem",
    )
    commercial = pa3.load_and_classify(
        workspace,
        verbose=False,
        precision_standard="commercial",
    )
    assert ecosystem["status"] == commercial["status"] == "FAIL"
    assert ecosystem["precision_standard_role"] == "cpu_diagnostic_only"


def test_load_rejects_missing_capture_manifest(tmp_path):
    source = [torch.ones(8)]
    workspace = _make_live_workspace(tmp_path, ours=source, source_outputs=source)
    (workspace / "a3_capture_manifest.json").unlink()
    result = pa3.load_and_classify(workspace, verbose=False)
    assert result["status"] == "FAIL"
    assert "provenance" in result["error"]


def test_load_rejects_tampered_source_capture(tmp_path):
    source = [torch.ones(8)]
    workspace = _make_live_workspace(tmp_path, ours=source, source_outputs=source)
    with (workspace / "edge_dataset.pt").open("ab") as stream:
        stream.write(b"tamper")
    result = pa3.load_and_classify(workspace, verbose=False)
    assert result["status"] == "FAIL"
    assert "mismatch" in result["error"]


def test_load_rejects_tampered_source_snapshot(tmp_path):
    source = [torch.ones(8)]
    workspace = _make_live_workspace(tmp_path, ours=source, source_outputs=source)
    staged_file = workspace / ".source_arch22" / "op_kernel" / "arch22" / "op.h"
    staged_file.write_text("class Tampered {};\n")
    result = pa3.load_and_classify(workspace, verbose=False)
    assert result["status"] == "FAIL"
    assert "snapshot" in result["error"]


def test_load_rejects_case_count_mismatch(tmp_path):
    source = [torch.ones(8)]
    workspace = _make_live_workspace(
        tmp_path,
        ours=[source[0].clone(), source[0].clone()],
        source_outputs=source,
    )
    result = pa3.load_and_classify(workspace, verbose=False)
    assert result["status"] == "FAIL"
    assert "case-count mismatch" in result["error"]


def test_load_rejects_missing_source_tensor(tmp_path):
    source = [torch.ones(8)]
    workspace = _make_live_workspace(tmp_path, ours=source, source_outputs=source)
    # Replacing a post-manifest tensor also requires updating its hash; this test
    # deliberately exercises the stronger provenance gate first.
    torch.save({"inputs": [torch.ones(8)], "a3_outputs": [None]}, workspace / "edge_dataset.pt")
    result = pa3.load_and_classify(workspace, verbose=False)
    assert result["status"] == "FAIL"


def test_load_rejects_synthetic_truth_marker(tmp_path):
    source = [torch.ones(8)]
    workspace = _make_live_workspace(tmp_path, ours=source, source_outputs=source)
    (workspace / ".truth_source_override").write_text(
        "truth_source=cpu_canonical_via_synthetic_edge_dataset\n"
    )
    result = pa3.load_and_classify(workspace, verbose=False)
    assert result["status"] == "FAIL"
    assert "cannot substitute" in result["error"]


def test_load_rejects_invalid_capture_timestamp(tmp_path):
    source = [torch.ones(8)]
    workspace = _make_live_workspace(tmp_path, ours=source, source_outputs=source)
    manifest_path = workspace / "a3_capture_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["capture_started_ts"] = "not-a-time"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    result = pa3.load_and_classify(workspace, verbose=False)
    assert result["status"] == "FAIL"
    assert "timestamps" in result["error"]


def test_load_rejects_missing_target_capture(tmp_path):
    source = [torch.ones(8)]
    workspace = _make_live_workspace(tmp_path, ours=source, source_outputs=source)
    (workspace / "a5_capture.pt").unlink()
    result = pa3.load_and_classify(workspace, verbose=False)
    assert result["status"] == "FAIL"
    assert "missing required capture" in result["error"]


def test_schema_validator_accepts_compatibility_shape(tmp_path):
    import check_verification_schema as schema

    verification = {
        "precision": {
            "pass_a": {
                "tier1_pass": 2,
                "tier2_pass": 0,
                "tier1_pass_inclusive": 2,
                "total": 2,
                "tier2_status": "N/A_ALL_T1",
                "status": "PASS",
            },
            "pass_b": {"status": "N/A", "reason": "arch migration"},
        }
    }
    path = tmp_path / "verification.json"
    path.write_text(json.dumps(verification))
    ok, reason = schema.check(path)
    assert ok, reason


def test_phase_o5_normalizer_retains_compatibility_fields():
    import phase_o5_runner as runner

    normalized = getattr(runner, '_normalize_port_a3_two_tier_pass_a')(
        {
            "tier1_pass": 2,
            "tier2_pass": 0,
            "tier1_pass_inclusive": 2,
            "total": 2,
            "tier2_status": "N/A_ALL_T1",
            "status": "PASS",
        }
    )
    assert normalized["tier1_pass"] == 2
    assert normalized["tier2_pass"] == 0
    assert normalized["method"] == "canonical_precision_eval_port_a3_two_tier"
