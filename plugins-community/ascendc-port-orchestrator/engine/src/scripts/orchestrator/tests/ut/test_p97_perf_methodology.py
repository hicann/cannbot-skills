# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P97 regression — PERF_METHODOLOGY_ASYMMETRY gate.

User catch 2026-05-16 02:30Z: "Worker 没作弊 我认为这就是作弊, 因为可以让
数值高, 生成更快". Reframe: outcome-based reward hacking does NOT require
intent. A3 baseline measured via torch_npu Python dispatch + A5 measured
via C++ aclnn-direct = methodology asymmetric = inflated ratio = cheating.

Per PR #103 `ascendc-operator-performance-eval` skill: both baseline and
candidate MUST use `torch_npu.profiler` with `warmup=5, active=5` on NPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import finalize_pipeline as fp


def _make_vj(precision_status, perf_status, method):
    return {
        "precision": {"status": precision_status, "pass_a": {}, "pass_b": {}},
        "performance": {"status": perf_status, "ratio": 2.5, "method": method},
    }


def test_gate_id_stable():
    assert fp.GateID.PERF_METHODOLOGY_ASYMMETRY.value == "perf_methodology_asymmetry"


def test_skip_status_bypasses_gate(tmp_path):
    """When precision.status is N/A or SKIPPED, no perf claim → gate idle."""
    vj = _make_vj("N/A", "N/A", "")
    assert getattr(fp, '_check_perf_methodology')(tmp_path, vj) is None


def test_partial_status_with_proper_method_passes(tmp_path):
    """PARTIAL status with correct method → gate idle."""
    vj = _make_vj("PARTIAL", "PASS",
                  "5-run via torch_npu.profiler schedule warmup=5, active=5")
    assert getattr(fp, '_check_perf_methodology')(tmp_path, vj) is None


def test_pass_with_torch_npu_profiler_method_passes(tmp_path):
    """PASS + torch_npu.profiler + warmup=5/active=5 → gate idle."""
    vj = _make_vj("PASS", "PASS",
                  "Both sides measured via torch_npu.profiler with "
                  "schedule warmup=5, active=5 on Ascend NPU.")
    assert getattr(fp, '_check_perf_methodology')(tmp_path, vj) is None


def test_pass_with_empty_method_legacy_allowed(tmp_path):
    """PASS with empty method → P97 transition rule: allow (legacy fixture).
    Future strictening: once all archives declare method, tighten to reject.
    """
    vj = _make_vj("PASS", "PASS", "")
    assert getattr(fp, '_check_perf_methodology')(tmp_path, vj) is None


def test_pass_with_chrono_method_rejected(tmp_path):
    """PASS with std::chrono (C++) instead of torch_npu.profiler → reject.
    This is exactly the 2026-05-16 incident.
    """
    vj = _make_vj("PASS", "PASS",
                  "5-run median per case via runner internal timing "
                  "(aclrtSynchronizeStream + std::chrono::high_resolution_clock).")
    result = getattr(fp, '_check_perf_methodology')(tmp_path, vj)
    assert result is not None
    assert "torch_npu.profiler" in result


def test_pass_with_torchnpu_but_wrong_warmup_rejected(tmp_path):
    """torch_npu.profiler used but with non-standard warmup → reject."""
    vj = _make_vj("PASS", "PASS",
                  "torch_npu.profiler schedule warmup=10, active=3 — custom")
    result = getattr(fp, '_check_perf_methodology')(tmp_path, vj)
    assert result is not None
    assert "warmup=5" in result or "active=5" in result


def test_pass_with_torchnpu_no_schedule_rejected(tmp_path):
    """torch_npu.profiler declared but no warmup/active values → reject."""
    vj = _make_vj("PASS", "PASS",
                  "torch_npu.profiler used; schedule not declared")
    result = getattr(fp, '_check_perf_methodology')(tmp_path, vj)
    assert result is not None
    assert "warmup=5" in result or "active=5" in result


def test_pass_within_tolerance_subject_to_gate(tmp_path):
    """PASS_WITHIN_TOLERANCE also enforces methodology check."""
    vj = _make_vj("PASS_WITHIN_TOLERANCE", "PASS",
                  "std::chrono timing (no profiler)")
    result = getattr(fp, '_check_perf_methodology')(tmp_path, vj)
    assert result is not None


def test_method_in_method_note_field_accepted(tmp_path):
    """Schema allows either `method` or `method_note` field."""
    vj = {
        "precision": {"status": "PASS"},
        "performance": {
            "status": "PASS", "ratio": 2.5,
            "method_note": "torch_npu.profiler warmup=5 active=5 both sides",
        },
    }
    assert getattr(fp, '_check_perf_methodology')(tmp_path, vj) is None


# ── P135.PR (2026-05-18): port_a3 mode accepts profiler-CSV as Option 1 ────


def test_p135pr_port_a3_profiler_csv_method_accepted(tmp_path):
    """P135.PR: port_a3 mode + perf method declares torch_npu.profiler +
    operator_details.csv + Device Self Duration → P141 gate accepts
    using the established symmetric profiler primitive.
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "PASS", "ratio": 2.1,
            "method": ("a3=torch_npu.profiler.profile around test_fn, parse "
                       "operator_details.csv groupby Name sum Device Self "
                       "Duration(us); a5=same primitive locally. warmup=5 "
                       "active=5 method=Option1 profiler-CSV symmetric"),
        },
    }
    assert getattr(fp, '_check_perf_methodology')(tmp_path, vj) is None


def test_p135pr_port_a3_perf_counter_with_profiler_label_accepted(tmp_path):
    """P135.PR: even if method mentions perf_counter alongside, the
    profiler-CSV signature wins (worker may declare both for transparency
    when option2_wrapper_inclusive is reported as secondary cross-check).
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "PASS", "ratio": 1.5,
            "method": ("a3=torch_npu.profiler.profile schedule(warmup=5, "
                       "active=5) + operator_details.csv Device Self "
                       "Duration(us) for primary timing on torch_npu.npu_op "
                       "call; a5=same primitive around ACLRT_LAUNCH_KERNEL "
                       "pybind. Secondary perf_counter Option 2 cross-check "
                       "in option2_wrapper_inclusive."),
        },
    }
    assert getattr(fp, '_check_perf_methodology')(tmp_path, vj) is None


def test_p135pr_port_a3_perf_counter_only_still_rejected(tmp_path):
    """P135.PR negative: bare perf_counter wrap WITHOUT profiler-CSV
    signature still triggers P141 (the original anti-pattern this gate
    catches is unchanged).
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "PASS", "ratio": 4.5,
            "method": ("a3=perf_counter around torch_npu.npu_clipped_swiglu "
                       "(aclnn dispatch); a5=perf_counter around "
                       "ACLRT_LAUNCH_KERNEL macro pybind"),
        },
    }
    result = getattr(fp, '_check_perf_methodology')(tmp_path, vj)
    assert result is not None
    assert "P141" in result


def test_p135pr_port_a3_profiler_csv_partial_match_rejected(tmp_path):
    """P135.PR: 'torch_npu.profiler' alone (without operator_details.csv
    + Device Self Duration tokens) is NOT enough — must declare full
    primitive chain so the gate confirms the right CSV column is parsed.
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "PASS", "ratio": 2.0,
            "method": ("a3=perf_counter around torch_npu.npu_op aclnn; "
                       "a5=perf_counter around ACLRT_LAUNCH_KERNEL"),
            # Note: "torch_npu.profiler" appears nowhere; partial-match
            # test: just naming the package without the CSV column
        },
    }
    result = getattr(fp, '_check_perf_methodology')(tmp_path, vj)
    assert result is not None  # falls back to original perf_counter rejection


def test_p135pr_p97_check_unchanged_for_non_port_a3(tmp_path):
    """P135.PR backward-compat: non-port_a3 modes still use P97
    methodology check (warmup=5 + active=5 etc.); P135.PR addition is
    scoped to port_a3 mode only.
    """
    vj = {
        "mode": "backward",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "PASS", "ratio": 1.5,
            # P97 wants torch_npu.profiler + warmup=5/active=5.
            "method": "perf via std::chrono timer",
        },
    }
    # The generic methodology gate rejects non-profiler methods.
    result = getattr(fp, '_check_perf_methodology')(tmp_path, vj)
    assert result is not None


# ── P135.DS (2026-05-18): Event-vs-perf_counter direction sanity check ────


def test_p135ds_event_lower_than_perf_counter_accepted(tmp_path):
    """P135.DS: Option 1 Event ratio LOWER than Option 2 perf_counter ratio
    is the EXPECTED direction (Event excludes host; A3 host overhead >
    A5 host overhead; excluding host lowers A3 number more → ratio
    lowers). foreach_neg single-tensor case: Event ~40µs < perf_counter
    ~60µs → Option 1 ratio < Option 2 ratio.
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "PASS",
            "ratio": 1.16,  # Option 1 device-event
            "method": ("a3=aclrtEventElapsedTime around aclnn execute; "
                       "a5=torch.npu.Event around ACLRT_LAUNCH_KERNEL; "
                       "method=Option1 device-event symmetric=true"),
            "option2_wrapper_inclusive": {
                "a3_us": 60.0, "a5_us": 25.0, "ratio": 2.40,
                "a3_wrapper_composition": "perf_counter + sync around torch_npu.npu_foreach_neg",
                "a5_wrapper_composition": "perf_counter + sync around model_new_ascendc.forward (pybind+ACLRT_LAUNCH)",
            },
        },
    }
    # Option 1 1.16× < Option 2 2.40× — Event correctly excludes host
    assert getattr(fp, '_check_perf_methodology')(tmp_path, vj) is None


def test_p135ds_event_higher_than_perf_counter_rejected(tmp_path):
    """P135.DS: Option 1 Event ratio HIGHER than Option 2 perf_counter
    ratio indicates stream stall leakage (foreach_sqrt 2026-05-18: Event
    2.70× > perf_counter 2.42× — wrong direction). Gate rejects when
    Option 1 / Option 2 > 1.10.
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "PASS",
            "ratio": 2.70,  # Option 1 device-event INFLATED
            "method": ("a3=aclrtEventElapsedTime around foreach_sqrt aclnn; "
                       "a5=torch.npu.Event around ACLRT_LAUNCH_KERNEL macro; "
                       "method=Option1 device-event symmetric=true"),
            "option2_wrapper_inclusive": {
                "a3_us": 58.0, "a5_us": 24.0, "ratio": 2.42,
                "a3_wrapper_composition": "perf_counter + sync around torch._foreach_sqrt",
                "a5_wrapper_composition": "perf_counter + sync around modelnew.forward",
            },
        },
    }
    # 2.70 / 2.42 = 1.116 > 1.10 threshold — reject
    result = getattr(fp, '_check_perf_methodology')(tmp_path, vj)
    assert result is not None
    assert "P135.DS" in result or "EVENT_LEAKAGE" in result


def test_p135ds_event_within_10pct_tolerance_accepted(tmp_path):
    """P135.DS: noise tolerance — if Option 1 is within 10% of Option 2,
    accept (measurement noise, not stream-stall leakage).
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "PASS",
            "ratio": 1.30,
            "method": ("a3=aclrtEventElapsedTime; a5=torch.npu.Event; "
                       "method=Option1 device-event symmetric=true"),
            "option2_wrapper_inclusive": {
                "ratio": 1.22,  # 1.30 / 1.22 = 1.066 < 1.10 tolerance
            },
        },
    }
    assert getattr(fp, '_check_perf_methodology')(tmp_path, vj) is None


def test_p135ds_skipped_when_no_option2_cross_check(tmp_path):
    """P135.DS only fires when BOTH Option 1 (event) AND Option 2
    (wrapper-inclusive) ratios present. Worker who declares Option 1
    only (no cross-check) bypasses this check — but workers should
    declare cross-check per brief D.4 'emit BOTH Option 1 ratio AND
    Option 2 ratio when feasible'.
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "PASS",
            "ratio": 2.70,  # Option 1 only
            "method": ("a3=aclrtEventElapsedTime; a5=torch.npu.Event; "
                       "method=Option1 device-event symmetric=true"),
            # No option2_wrapper_inclusive
        },
    }
    # No cross-check possible — accepted (other gates handle method label only)
    assert getattr(fp, '_check_perf_methodology')(tmp_path, vj) is None


def test_p135ds_skipped_when_profiler_csv_used(tmp_path):
    """P135.DS doesn't fire for profiler-CSV path (Option 1 PRIMARY) since
    profiler-CSV doesn't have the stream-stall leakage problem this gate
    catches. Only torch.npu.Event wrap is at risk.
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "PASS",
            "ratio": 2.70,  # Profiler-CSV primary ratio (no event wrap)
            "method": ("a3=torch_npu.profiler.profile schedule(warmup=5, "
                       "active=5) + operator_details.csv Device Self "
                       "Duration(us); a5=same primitive locally"),
            "option2_wrapper_inclusive": {
                "ratio": 2.42,  # Wall-clock cross-check (might be lower)
            },
        },
    }
    # has_profiler_csv=True → P135.DS skip the event-leakage check
    assert getattr(fp, '_check_perf_methodology')(tmp_path, vj) is None
