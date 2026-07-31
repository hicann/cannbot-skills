# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""B3.3b increment-2: backward orchestrator end-to-end finalize/phase_o5 awareness.

Covers the harness touchpoints that let `orch --backward` flow a backward op through
finalize WITHOUT manual spawn/finalize (design §12 TP3/TP4):

  1. BackwardPlugin.check_op_host_completeness → None — backward uses the pybind11 .so
     paradigm (like benchmark), not the port_a3 op_host/ deliverable. PB-33 must not block.
  2. phase_o5_runner.backward_verify_runner — re-measures via verify_<op>.py (self-contained
     autograd verify), maps pass/total → pass_a, pass_b N/A by design. Author≠measurer.
  3. runner selection — orchestrator picks backward_verify_runner iff
     phase_o5.expected_truth_source == "backward_autograd".
  4. finalize eligibility — a backward workspace carrying the B3.3b schema contract
     (pass_a counts + pass_b N/A + canonical perf.status + KB/audit/scan artifacts, NO
     edge_dataset/op_host) reaches eligible=true with ZERO finalize_pipeline gate edits.

Pure CPU + mocks; no hardware.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent  # src/scripts/orchestrator/
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

import phase_o5  # noqa: E402
import phase_o5_runner  # noqa: E402
import finalize_pipeline as fp  # noqa: E402
from plugins.backward import BackwardPlugin  # noqa: E402
from phase_o5 import MeasuredResult  # noqa: E402


# ── 1. PB-33 op_host exemption ────────────────────────────────────────────────

def test_backward_plugin_op_host_completeness_is_none(tmp_path):
    """Backward (pybind .so paradigm) must NOT be blocked by PB-33 op_host gate."""
    assert BackwardPlugin().check_op_host_completeness(tmp_path) is None


# ── 2. backward_verify_runner ─────────────────────────────────────────────────

_ENV = {"TARGET": "a3", "A3_HOST": "h", "A3_CONTAINER": "npu-a3-back"}


def _mk_ws(tmp_path, *, with_verify=True, op="rms_norm_grad"):
    ws = tmp_path / op
    ws.mkdir()
    (ws / ".opgen_state.json").write_text(json.dumps({"op": op, "opgen_mode": "backward"}))
    if with_verify:
        # recomputes autograd truth in-process; does NOT read verification.json
        (ws / f"verify_{op}.py").write_text(
            "import torch\nprint('JSON:'+__import__('json').dumps([{'pass':True}]))\n"
            'print(__import__("json").dumps({"tier1_pass":6,"total":6,"status":"PASS"}))\n'
        )
    return ws


def test_backward_runner_missing_env(tmp_path):
    ws = _mk_ws(tmp_path)
    with patch.object(phase_o5_runner, "_read_ascendc_env", return_value={}):
        r = phase_o5_runner.backward_verify_runner(ws, "rms_norm_grad")
    assert r.runner_error and ".ascendc_env" in r.runner_error


def test_backward_runner_missing_verify_script(tmp_path):
    ws = _mk_ws(tmp_path, with_verify=False)
    with patch.object(phase_o5_runner, "_read_ascendc_env", return_value=_ENV):
        r = phase_o5_runner.backward_verify_runner(ws, "rms_norm_grad")
    assert r.runner_error and "verify_rms_norm_grad.py" in r.runner_error


def test_backward_runner_success_maps_summary_to_pass_a(tmp_path):
    """Remote path: resync + _run_verifier return a normalized summary → pass_a;
    pass_b stays None (N/A by design).
    """
    ws = _mk_ws(tmp_path)
    with patch.object(phase_o5_runner, "_read_ascendc_env", return_value=_ENV), \
         patch.object(phase_o5_runner, "_verify_runner_independence", return_value=None), \
         patch.object(phase_o5_runner, "_resync_workspace_to_container", return_value=None), \
         patch.object(phase_o5_runner, "_run_verifier",
                      return_value={"tier1_pass": 6, "total": 6, "status": "PASS"}) as mrv:
        r = phase_o5_runner.backward_verify_runner(ws, "rms_norm_grad", lane=3)
    assert r.runner_error is None
    assert r.pass_a == {"tier1_pass": 6, "total": 6, "status": "PASS"}
    assert r.pass_b is None
    # ran the self-contained verify script as pass_a, on the requested lane
    args, kwargs = mrv.call_args
    assert args[3] == "verify_rms_norm_grad.py" and args[4] == "pass_a"
    assert kwargs.get("lane") == 3


def test_backward_runner_independence_guard_propagates(tmp_path):
    """A verify script that self-cites verification.json is rejected (anti-cycle)."""
    ws = _mk_ws(tmp_path)
    with patch.object(phase_o5_runner, "_read_ascendc_env", return_value=_ENV), \
         patch.object(phase_o5_runner, "_verify_runner_independence",
                      return_value="WORKER-SELF-CITING-VERIFIER: cites verification.json"):
        r = phase_o5_runner.backward_verify_runner(ws, "rms_norm_grad")
    assert r.runner_error and "SELF-CITING" in r.runner_error


# ── 3. runner selection by truth_source ───────────────────────────────────────

def test_truth_source_selects_backward_runner(tmp_path):
    ws = _mk_ws(tmp_path)
    assert phase_o5.expected_truth_source(ws) == "backward_autograd"
    # Missing/unsupported workflow ownership must fail closed.
    other = tmp_path / "unsupported"
    other.mkdir()
    (other / ".opgen_state.json").write_text(json.dumps({"opgen_mode": "unsupported"}))
    with pytest.raises(RuntimeError, match="cannot resolve supported truth source"):
        phase_o5.expected_truth_source(other)


# ── 4. finalize eligibility on the B3.3b schema ───────────────────────────────

def _backward_archive(ws: Path, op="rms_norm_grad"):
    """A backward workspace carrying the mandated B3.3b finalize schema."""
    ws.mkdir(parents=True, exist_ok=True)
    (ws / ".opgen_state.json").write_text(json.dumps({"op": op, "opgen_mode": "backward"}))
    (ws / "model.py").write_text(
        "import torch, torch.nn as nn\n"
        "class Model(nn.Module):\n"
        "    def forward(self, x, w, gy): return x\n"
        "def get_input_groups(): return [[]]\n"
    )
    (ws / "model_new_ascendc.py").write_text(
        "import torch, torch.nn as nn\n"
        "class ModelNew(nn.Module):\n"
        "    def forward(self, x, w, gy): return x\n"
        "if __name__ == '__main__':\n    pass\n"
    )
    kdir = ws / "kernel"
    kdir.mkdir()
    (kdir / "pybind11.cpp").write_text("// pybind\n")
    (kdir / f"{op}_kernels.cpp").write_text("// kernel\n")
    (kdir / f"_{op}_ext.so").write_bytes(b"compiled-extension")
    (ws / f"verify_{op}.py").write_text(
        "from model_new_ascendc import ModelNew\n"
        "candidate = ModelNew()\n"
        "out = candidate(x, w, gy)\n"
    )
    (ws / "knowledge_update.md").write_text(
        "# Knowledge Update: rms_norm_grad backward\n\n"
        "## Findings\n"
        "- AscendC VEC Rsqrt is a ~1e-3 HW approximation on V220 → 2 Newton-Raphson "
        "refinement steps recover fp32-grade gradient precision (OL-103).\n"
        "- grad_w race fixed via depth-2 VECIN TQue (emblayerbwd K2 pattern).\n\n"
        "## KB-promotable patterns\n- Consumes OL-103, OL-200; no new pattern.\n\n"
        "## KB entries touched\n- OL-103 (transcendental fp16 ceiling carries into backward).\n"
    )
    (ws / "audit_self_critic_post_worker.md").write_text(
        "# Post-worker self-critic audit: rms_norm_grad backward\n\n"
        "## Verdict: PASS\n"
        "No C13/C18/C25/C26 watchpoints triggered. Kernel uses real AscendC (DataCopy/VEC/"
        "TQue), no torch.autograd / CPU fallback / aclnn delegation. Precision 6/6 vs fp64 "
        "autograd truth; perf at honest vendor parity. Self-contained verify recomputes the "
        "autograd oracle in-process (not self-citing verification.json).\n"
    )
    (ws / ".delegation_scan_passed").touch()
    vj = {
        "op": op, "target": "a3", "soc_version": "Ascend910_9382", "arch": "arch22",
        "build": {"status": "SUCCESS"},
        "precision": {
            "pass": 6, "total": 6, "status": "PASS",
            "cases": [{"dtype": "float32", "shape": [4, 1024], "pass": True}],
            "dtype_summary": {"float32": "6/6"},
            "pass_a": {"status": "PASS", "tier1_pass": 6, "total": 6,
                       "method": "self-contained autograd verify (verify_<op>.py vs fp64 torch.autograd.grad)"},
            "pass_b": {"status": "N/A",
                       "reason": (
                           "backward mode: pass_b degenerate by design — autograd grad IS truth; "
                           "verify cases ARE pass_a (like port_a3 edge_dataset)"
                       ),
                       "method": "n/a — backward mode pass_b not applicable"},
        },
        "performance": {"status": "N/A",
                        "reason": "rough vendor parity (~1.0x, launch-overhead-bound); canonical N/A for parity-class"},
        "determinism": {"policy": "best_effort", "observed_deterministic": True},
        "harness_pristine": {
            "state": "CLEAN", "o5_verdict": "VERIFIED",
            "sampled_at": "o5_post_verify",
        },
    }
    (ws / "verification.json").write_text(json.dumps(vj, indent=1))
    return ws


def test_backward_archive_is_finalize_eligible(tmp_path):
    """The full B3.3b schema reaches eligible=true — proves no finalize_pipeline gate
    edits are needed beyond the op_host plugin override (gates self-handle backward).
    """
    ws = _backward_archive(tmp_path / "rms_norm_grad")
    res = fp.check_finalize_eligibility(ws)
    assert res.get("eligible") is True, f"unexpected gate block: {res.get('gate')}: {res.get('reason')}"


def test_backward_archive_op_host_gate_does_not_fire(tmp_path):
    """No workspace/op_host/ present, yet op_host gate must not block (plugin override)."""
    ws = _backward_archive(tmp_path / "rms_norm_grad")
    assert not (ws / "op_host").exists()
    plugin = BackwardPlugin()
    assert plugin.check_op_host_completeness(ws) is None


def test_backward_archive_missing_pass_b_status_blocks(tmp_path):
    """Guard: if the worker omits the canonical pass_b N/A, finalize blocks (the schema
    contract is load-bearing, not cosmetic).
    """
    ws = _backward_archive(tmp_path / "rms_norm_grad")
    vj = json.loads((ws / "verification.json").read_text())
    del vj["precision"]["pass_b"]
    (ws / "verification.json").write_text(json.dumps(vj))
    res = fp.check_finalize_eligibility(ws)
    assert res.get("eligible") is False


def test_backward_archive_missing_binary_blocks(tmp_path):
    ws = _backward_archive(tmp_path / "rms_norm_grad")
    next(ws.rglob("*.so")).unlink()
    res = fp.check_finalize_eligibility(ws)
    assert res.get("eligible") is False
    assert res.get("gate") == fp.GateID.BINARY_PROVENANCE.value


def test_backward_archive_verifier_bypass_blocks(tmp_path):
    ws = _backward_archive(tmp_path / "rms_norm_grad")
    (ws / "verify_rms_norm_grad.py").write_text("print('counts only')\n")
    res = fp.check_finalize_eligibility(ws)
    assert res.get("eligible") is False
    assert res.get("gate") == fp.GateID.VERIFIER_USES_MODELNEW.value


def test_backward_archive_without_o5_stamp_blocks(tmp_path):
    ws = _backward_archive(tmp_path / "rms_norm_grad")
    vj = json.loads((ws / "verification.json").read_text())
    del vj["harness_pristine"]
    (ws / "verification.json").write_text(json.dumps(vj))
    reason = BackwardPlugin().check_verify_path_provenance(ws, vj)
    assert reason is not None and "O5 independent" in reason


@pytest.mark.parametrize("status,passed,total", [("FAIL", 0, 1), ("PASS", 0, 0)])
def test_backward_archive_requires_concrete_pass_a(
    tmp_path, status, passed, total
):
    ws = _backward_archive(tmp_path / "rms_norm_grad")
    vj = json.loads((ws / "verification.json").read_text())
    vj["precision"]["pass_a"].update({
        "status": status, "tier1_pass": passed, "total": total,
    })
    (ws / "verification.json").write_text(json.dumps(vj))
    res = fp.check_finalize_eligibility(ws)
    assert res.get("eligible") is False
    assert res.get("gate") == fp.GateID.PASS_COUNT.value


@pytest.mark.parametrize("status,passed,total", [("FAIL", 0, 1), ("PASS", 0, 0)])
def test_backward_runner_rejects_nonpassing_or_empty_summary(
    tmp_path, status, passed, total
):
    ws = _mk_ws(tmp_path)
    summary = {"tier1_pass": passed, "total": total, "status": status}
    with patch.object(phase_o5_runner, "_read_ascendc_env", return_value=_ENV), \
         patch.object(phase_o5_runner, "_verify_runner_independence", return_value=None), \
         patch.object(phase_o5_runner, "_resync_workspace_to_container", return_value=None), \
         patch.object(phase_o5_runner, "_run_verifier", return_value=summary):
        result = phase_o5_runner.backward_verify_runner(ws, "rms_norm_grad")
    assert result.runner_error

# ── 5. P149 V220-EC-41 output-pad structural carve-out (2026-06-03, main-authorized) ──


def _v220_pybind_ec41(ws: Path, *, pad=True, narrow=True, datacopypad=False, aligned_dc=True):
    """Write a pybind (pad+narrow output pattern) + a kernel .h with the structural signal."""
    ws.mkdir(parents=True, exist_ok=True)
    kdir = ws / "kernel"
    kdir.mkdir(exist_ok=True)
    pb = "#include <torch/extension.h>\n"
    if pad:
        pb += "  constexpr int64_t PAD_ELEMS = 16;\n  auto dx_padded = torch::empty({n + PAD_ELEMS}, opt);\n"
    if narrow:
        pb += "  auto dx = dx_padded.narrow(0, 0, n).view(sizes);\n"
    (kdir / "pybind11.cpp").write_text(pb)
    kh = "// OL-120: 3-arg DataCopy only (no DataCopyPad on V220).\n"
    if aligned_dc:
        kh += "  DataCopy(dx_global_[off], dx_local, AlignUp32(cur));\n"
    if datacopypad:
        kh += "  DataCopyPad(dx_global_[off], dx_local, params);\n"
    (kdir / "op_kernel.h").write_text(kh)
    return ws


_V220_VJ = {
    "arch": "V220 / arch22 / SIMD-only",
    "target": "a3",
    "precision": {
        "status": "PASS",
        "pass_a": {
            "status": "PASS",
            "tier1_pass": 6,
            "total": 6,
            "method": (
                "self-contained autograd verify "
                "(verify_<op>.py vs fp64 torch.autograd.grad)"
            ),
        },
    },
}


def test_p149_selfdeclared_precision_no_autograd_not_exempt(tmp_path):
    """main #2: precision PASS but WITHOUT the independent autograd-oracle method signature
    (self-declared) → no exemption. Worker can't earn the carve-out by self-asserting PASS.
    """
    ws = _v220_pybind_ec41(tmp_path / "op")
    vj = {"arch": "arch22", "target": "a3",
          "precision": {"status": "PASS", "pass_a": {"status": "PASS", "method": "textual identity reasoning"}}}
    assert getattr(fp, '_is_v220_ec41_output_pad_exempt')(ws, vj) is False


def test_p149_v220_ec41_pad_exempt(tmp_path):
    """V220 + aligned 3-arg DataCopy + no DataCopyPad + precision PASS → exempt → P149 None."""
    ws = _v220_pybind_ec41(tmp_path / "gelu_grad")
    assert getattr(fp, '_is_v220_ec41_output_pad_exempt')(ws, _V220_VJ) is True
    assert getattr(fp, '_check_pybind_host_logic')(ws, _V220_VJ) is None


def test_p149_a5_pad_not_exempt(tmp_path):
    """A5 (arch35/V351) — DataCopyPad available → pad pattern is a real cleanup → fires."""
    ws = _v220_pybind_ec41(tmp_path / "op")
    vj = {"arch": "arch35 / V351", "target": "a5", "precision": {"status": "PASS"}}
    assert getattr(fp, '_is_v220_ec41_output_pad_exempt')(ws, vj) is False
    assert getattr(fp, '_check_pybind_host_logic')(ws, vj) is not None  # P149 fires


def test_p149_precision_fail_not_exempt(tmp_path):
    """Precision not verified → valid region not proven bit-correct → no exemption."""
    ws = _v220_pybind_ec41(tmp_path / "op")
    vj = {"arch": "arch22", "target": "a3", "precision": {"status": "FAIL"}}
    assert getattr(fp, '_is_v220_ec41_output_pad_exempt')(ws, vj) is False


def test_p149_datacopypad_call_not_exempt(tmp_path):
    """Kernel has a real DataCopyPad( call → not the V220-forced 3-arg path → no exemption."""
    ws = _v220_pybind_ec41(tmp_path / "op", datacopypad=True)
    assert getattr(fp, '_is_v220_ec41_output_pad_exempt')(ws, _V220_VJ) is False
    assert getattr(fp, '_check_pybind_host_logic')(ws, _V220_VJ) is not None


def test_p149_no_aligned_datacopy_not_exempt(tmp_path):
    """No aligned 3-arg DataCopy signal → can't prove the pad is the forced scratch."""
    ws = _v220_pybind_ec41(tmp_path / "op", aligned_dc=False)
    assert getattr(fp, '_is_v220_ec41_output_pad_exempt')(ws, _V220_VJ) is False


def test_p149_cpu_offload_never_exempt_even_v220(tmp_path):
    """The carve-out is Pattern-2 (output-pad) ONLY — Pattern-1 CPU-offload still fires on V220."""
    ws = _v220_pybind_ec41(tmp_path / "op")
    # inject a CPU-offload into the pybind → Pattern-1 must still fire
    pb = (ws / "kernel" / "pybind11.cpp")
    pb.write_text(pb.read_text() + "  auto g = idx.to(at::kCPU).contiguous();\n")
    r = getattr(fp, '_check_pybind_host_logic')(ws, _V220_VJ)
    assert r is not None and "CPU offload" in r


# ── B3.3c: backward independent perf re-measure (perf-runner gap fix) ──────────
# Regression fixture = the REAL abs_nocase_grad verify-summary perf block that
# blocked finalize 2026-06-17 (perf measured but performance.independent_re_measure
# never populated → "never trust skill-reported performance" rollback).

_PERF_FIXTURE = {
    "status": "PASS",
    "ratio": 1.4,
    "median_ratio": 1.4,
    "method": ("torch_npu.profiler device_self_duration (kernel_details.csv); "
               "schedule warmup=5, active=5; symmetric both sides"),
    "rows": [
        {"dtype": "float32", "shape": [4096], "ours_device_us": 1.864,
         "vendor_device_us": 2.616, "ratio": 1.4},
        {"dtype": "float16", "shape": [4096], "ours_device_us": 2.8,
         "vendor_device_us": 2.864, "ratio": 1.02},
    ],
}


def test_backward_runner_captures_perf_block(tmp_path):
    """backward_verify_runner must lift the verify summary's `performance` block into
    MeasuredResult.perf (orchestrator's INDEPENDENT re-measure, not worker self-report).
    """
    ws = _mk_ws(tmp_path)
    summary = {"tier1_pass": 30, "total": 30, "status": "PASS",
               "performance": _PERF_FIXTURE}
    with patch.object(phase_o5_runner, "_read_ascendc_env", return_value=_ENV), \
         patch.object(phase_o5_runner, "_verify_runner_independence", return_value=None), \
         patch.object(phase_o5_runner, "_resync_workspace_to_container", return_value=None), \
         patch.object(phase_o5_runner, "_run_verifier", return_value=summary):
        r = phase_o5_runner.backward_verify_runner(ws, "rms_norm_grad", lane=1)
    assert r.runner_error is None
    assert r.pass_a and r.pass_a.get("tier1_pass") == 30
    assert r.perf is not None and r.perf.get("median_ratio") == 1.4
    assert r.perf.get("rows") and r.perf["rows"][0]["ours_device_us"] == 1.864


def test_post_verify_writes_independent_re_measure(tmp_path):
    """post_verify_for_finalize maps measured.perf → verification.json
    performance.independent_re_measure {ran:true, ratio, delta_vs_kw_self_report,
    source}. Gate logic untouched; this just supplies the field finalize checks.
    """
    op = "rms_norm_grad"
    ws = tmp_path / op
    ws.mkdir()
    (ws / ".opgen_state.json").write_text(json.dumps({"op": op, "opgen_mode": "backward"}))
    # worker's verification.json: precision.pass_a claimed + perf self-report (ratio 1.38)
    # but NO independent_re_measure yet → this is what blocked finalize.
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"pass_a": {"status": "PASS", "tier1_pass": 30, "total": 30}},
        "performance": {"status": "PASS", "ratio": 1.38},  # worker self-report
    }))

    def _runner(workspace, opn, lane):
        return MeasuredResult(
            pass_a={"tier1_pass": 30, "total": 30, "status": "PASS"},
            pass_b=None,
            perf=_PERF_FIXTURE,  # orchestrator's independent re-measure
        )

    rep = phase_o5.post_verify_for_finalize(ws, op, runner=_runner)
    assert rep.verdict == "VERIFIED", rep.summary
    v = json.loads((ws / "verification.json").read_text())
    irm = v["performance"]["independent_re_measure"]
    assert irm["ran"] is True
    assert irm["ratio"] == 1.4 and irm["median_ratio"] == 1.4
    # delta vs the worker's self-reported 1.38 → independence is observable
    assert irm["delta_vs_kw_self_report"] == round(1.4 - 1.38, 4)
    assert "independent" in irm["source"].lower() and "self-report" in irm["source"].lower()
    # absolute device-time carried (ratio-optional modes rely on this)
    assert irm["rows"][0]["ours_device_us"] == 1.864


def test_post_verify_no_perf_is_noop(tmp_path):
    """Runners that don't set measured.perf (legacy benchmark) → no independent_re_measure
    written (mode-agnostic block is a no-op, doesn't clobber).
    """
    op = "rms_norm_grad"
    ws = tmp_path / op
    ws.mkdir()
    (ws / ".opgen_state.json").write_text(json.dumps({"op": op, "opgen_mode": "backward"}))
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"pass_a": {"status": "PASS", "tier1_pass": 30, "total": 30}},
        "performance": {"status": "PASS", "ratio": 1.38},
    }))

    def _runner(workspace, opn, lane):
        return MeasuredResult(pass_a={"tier1_pass": 30, "total": 30, "status": "PASS"}, pass_b=None)

    rep = phase_o5.post_verify_for_finalize(ws, op, runner=_runner)
    assert rep.verdict == "VERIFIED", rep.summary
    v = json.loads((ws / "verification.json").read_text())
    assert "independent_re_measure" not in v.get("performance", {})
