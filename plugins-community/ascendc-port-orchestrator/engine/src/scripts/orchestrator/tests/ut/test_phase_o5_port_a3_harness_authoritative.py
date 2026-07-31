# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""① HARNESS-AUTHORITATIVE port_a3 pass_a (owner-directed 2026-06-30).

phase_o5's canonical pass_a for port_a3 now runs the HARNESS grader
`precision_eval_port_a3_two_tier.load_and_classify` (compare.py 生态) on the worker-EMITTED
tensors as the AUTHORITATIVE verdict — SUPERSEDING the worker pass_a_runner.py self-verdict
(anti-reward-hack). These tests exercise the LOCAL-container path end-to-end on WSL (it runs a
local subprocess reading .pt files — no NPU, no SSH; the SSH variant is VALIDATED-BY-③). They
also prove that optional native_capture.pt diagnostics cannot change the bound source-NPU verdict.

CRITICAL: NO pass_a_runner.py is created in these fixtures — the verdict is produced entirely by
the harness grader, proving the worker no longer decides pass/fail.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import phase_o5_runner  # noqa: E402
from a3_ref_validate import write_a3_capture_provenance  # noqa: E402
from source_arch import stage_source_tree  # noqa: E402


def _mk_port_a3_workspace(tmp_path, *, with_native: bool):
    """Build a staged-source workspace with harness-owned source-NPU truth."""
    ws = tmp_path / "near_zero_op"
    ws.mkdir()
    source = tmp_path / "source_op"
    kernel = source / "op_kernel" / "arch22"
    kernel.mkdir(parents=True)
    (kernel / "op.h").write_text("class SourceOp { void Process() {} };\n")
    stage = stage_source_tree(source, ws)
    (ws / ".opgen_state.json").write_text(json.dumps({
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
    (ws / "model.py").write_text("import torch.nn as nn\nclass Model(nn.Module):\n    def forward(self, x): return x\n")
    (ws / "model_new_ascendc.py").write_text(
        "import torch.nn as nn\n"
        "class ModelNew(nn.Module):\n"
        "    def forward(self, x): return x\n"
    )
    (ws / "manifest.json").write_text(json.dumps({"inputs": [], "outputs": []}))

    g = torch.cat([torch.abs(torch.randn(2000).double()) + 0.5,
                   torch.full((48,), 1e-8, dtype=torch.float64)])   # fp32 near-zero tail
    ours = g.to(torch.float32).clone()
    ours[2000:] += 1e-6                                             # near-zero error
    native = ours.clone()                                           # CPU-same-precision baseline
    source_npu = ours.clone()

    torch.save([ours], ws / "a5_capture.pt")
    torch.save([None], ws / "edge_inputs.pt")
    torch.save(
        [{"case_id": 0, "inputs": None, "a3_outputs": source_npu}],
        ws / "edge_dataset.pt",
    )
    (ws / "a3_baseline_perf.json").write_text(json.dumps({
        "median_ms_per_case": {"case_0": 1.0},
    }))
    (ws / "run_a3_reference.py").write_text("# harness test runner\n")
    torch.save([g], ws / "cpu_truth_outputs.pt")
    if with_native:
        # whitelisted provider-shape native (dict + native_kind); a bare list is now fail-closed
        torch.save({"native_kind": "cpu_same_precision", "outputs": [native]}, ws / "native_capture.pt")
    ok, reason, manifest = write_a3_capture_provenance(
        ws,
        capture_id="capture-test",
        capture_started_ts=datetime.now(timezone.utc).isoformat(),
        npu_id=0,
    )
    assert ok, reason
    assert manifest is not None
    return ws


def test_local_canonical_is_harness_authoritative_with_native(tmp_path):
    """The harness grades emitted tensors against bound source-NPU truth."""
    ws = _mk_port_a3_workspace(tmp_path, with_native=True)
    assert getattr(phase_o5_runner, '_is_port_a3_mode')(ws) is True
    assert not (ws / "pass_a_runner.py").exists()        # worker verdict producer ABSENT

    res = getattr(phase_o5_runner, '_run_canonical_pass_a_local')(ws, "near_zero_op", env={}, lane=0)
    assert isinstance(res, dict), f"expected harness verdict dict, got {res!r}"
    # Canonical shape comes from the harness grader, not a worker self-verdict.
    assert "tier2_status" in res and "tier1_pass_inclusive" in res
    assert res.get("total") == 1
    assert res.get("status") == "PASS"
    assert res.get("tier1_pass") == 1
    assert "fresh arch22 NPU truth" in res.get("grader", "")
    assert res.get("native_capture_present") is True


def test_local_canonical_without_native_keeps_source_truth_authoritative(tmp_path):
    """Missing optional CPU-native diagnostics cannot change source-NPU status."""
    ws = _mk_port_a3_workspace(tmp_path, with_native=False)
    assert not (ws / "native_capture.pt").exists()
    res = getattr(phase_o5_runner, '_run_canonical_pass_a_local')(ws, "near_zero_op", env={}, lane=0)
    assert isinstance(res, dict)
    assert res.get("status") == "PASS"
    assert "fresh arch22 NPU truth" in res.get("grader", "")
    assert res.get("native_capture_present") is False
    assert res.get("native_provision_ok") is False


def test_native_capture_in_o5_sync_payload():
    """When present, native diagnostics must reach the remote re-grade."""
    src = Path(phase_o5_runner.__file__).read_text()
    # it appears in the push_files tuple (sync payload), not only in a comment
    assert '"native_capture.pt"' in src


def test_native_capture_in_force_update_scripts():
    """Harness-authored native diagnostics must not reuse stale remote bytes."""
    src = Path(phase_o5_runner.__file__).read_text()
    marker = "FORCE_UPDATE_SCRIPTS = {"
    start = src.index(marker)
    block = src[start:src.index("}", start)]
    assert '"native_capture.pt"' in block, "native_capture.pt not in FORCE_UPDATE_SCRIPTS set"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_normalizer_keeps_native_provisioning_fields():
    """The O5 normalizer keeps optional native-provisioning diagnostics."""
    parsed = {
        "total": 2, "tier1_pass": 1, "tier2_pass": 0, "tier1_pass_inclusive": 1,
        "status": "FAIL", "tier2_status": "FAIL",
        "native_capture_present": True, "native_usable": False, "native_kind": "cpu_fp32_fallback",
        "native_provision_ok": False, "native_provision_failed": True,
        "n_native_used": 0, "n_native_missing": 2, "missing_native_case_ids": [0, 1],
        "reason": "native_provision_failed",
    }
    out = getattr(phase_o5_runner, '_normalize_port_a3_two_tier_pass_a')(parsed)
    for k in ("native_capture_present", "native_usable", "native_kind", "native_provision_ok",
              "native_provision_failed", "n_native_used", "n_native_missing",
              "missing_native_case_ids", "reason"):
        assert out.get(k) == parsed[k], f"normalizer DROPPED {k}"
