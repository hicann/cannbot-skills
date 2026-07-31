# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""PB (2026-07-06): `_render_verification_conclusion` — customer-facing archive README
「验证结论」 block. Guards the 4 render bugs independent review found + landed equivalently in a5_ops
source (re-sync converges to 0-diff): empty verdict, precision without pass-count, `None×`
perf, raw internal truth_source token. Also locks provenance-honesty (a `;`-suffixed
`ours=a3_capture` must NOT flip a CPU-canonical reference into an A3-golden label)."""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))  # orchestrator/ on path
import finalize_pipeline as fp  # noqa: E402


def _r(vj):
    return getattr(fp, '_render_verification_conclusion')(vj)


def test_no_none_times_on_na_perf():
    out = _r({"precision": {"status": "PASS", "pass_a": {"tier1_pass": 87, "total": 87}},
              "performance": {"status": "N/A"}, "determinism": {"observed_deterministic": True},
              "truth_source": "cpu_canonical_via_synthetic_edge_dataset; ours=a3_capture_via_pybind"})
    assert "None×" not in out and "None x" not in out
    assert "N/A（移植" in out


def test_precision_shows_tier1_count():
    out = _r({"precision": {"status": "PASS", "pass_a": {"tier1_pass": 87, "total": 87}}})
    assert "87/87" in out, out


def test_no_raw_truth_token_and_provenance_honest():
    # `;`-suffixed ours=a3_capture must NOT flip the CPU reference to an A3-golden label
    out = _r({"truth_source": "cpu_canonical_via_synthetic_edge_dataset; ours=a3_capture_via_pybind"})
    assert "cpu_canonical_via_synthetic" not in out, "raw token leaked"
    assert "A3-CANN" not in out, "provenance bug: CPU reference mislabeled A3-golden"
    assert "CPU 真值" in out


def test_empty_verdict_synthesized_from_precision():
    out = _r({"precision": {"status": "PASS", "pass_a": {"tier1_pass": 9, "total": 9}},
              "performance": {"status": "BELOW_THRESHOLD", "ratio": 0.46}})
    assert "总体**: 通过（精度达标，性能待优化）" in out, out
    assert "`—`" not in out and "Verdict**: `—`" not in out


def test_determinism_line_present():
    assert "确定性**: 确定" in _r({"determinism": {"observed_deterministic": True}})
    assert "确定性**: 非确定" in _r({"determinism": {"observed_deterministic": False}})
    assert "确定性**: 不适用" in _r({"determinism": {}})


def test_numeric_perf_ratio_rendered():
    out = _r({"performance": {"ratio": 0.6312}})
    assert "0.63×" in out


def test_autograd_reference_humanized():
    out = _r({"truth_source": "autograd_grad_cpu_fp64"})
    assert "autograd" in out and "梯度真值" in out


def test_a3_golden_reference_labeled_when_real():
    # when the reference genuinely IS a3-capture/aclnn (not a `;`-suffix), label A3-golden
    out = _r({"truth_source": "a3_capture_via_aclnn"})
    assert "A3-CANN" in out
