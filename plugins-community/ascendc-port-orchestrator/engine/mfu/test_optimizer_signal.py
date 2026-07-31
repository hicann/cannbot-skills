#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""optimizer_signal / mfu_model 的测试。验证 MFU 信号对已知输入给出正确的
current_mfu / achievable / bottleneck / lever / done 判定。

run: python3 mfu/test_optimizer_signal.py   (exit 0 = all pass)
"""
import logging
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mfu_model import HW, DTYPE_BYTES, flash_attention_ops, solve_mfu, matmul_ops
from optimizer_signal import mfu_signal

FAILS = []
LOGGER = logging.getLogger(__name__)


def check(name, cond, detail=""):
    LOGGER.info("  [%s] %s%s", "PASS" if cond else "FAIL", name,
                f" — {detail}" if detail and not cond else "")
    if not cond:
        FAILS.append(name)


def test_mm_grad_single_aic_flags_continue():
    """mm_grad 单AIC shipped op: 必须判 CONTINUE + 低 MFU + multi-AIC 杠杆(owner实证)。"""
    M, K, N = 512, 256, 512
    flops = 2 * M * K * N * 2
    hbm = (M * K + K * N + M * N) * DTYPE_BYTES["fp16"] * 2
    s = mfu_signal(flops, hbm, 78.0, "910C_die", "fp16", "matmul", n_aic_used=1)
    check("mm_grad current_mfu<5%", s["current_mfu"] < 0.05, f"got {s['current_mfu']}")
    check("mm_grad NOT done", s["done"] is False)
    check("mm_grad lever 含 multi-AIC", any("multi-AIC" in l for l in s["levers"]))
    check("mm_grad gap<<1", s["gap_to_achievable"] < 0.1, f"got {s['gap_to_achievable']}")


def test_near_ceiling_matmul_is_done():
    """充分优化的大 matmul (用满24AIC, 实测≈理论×η): 应判 done。"""
    M = K = N = 4096
    flops = 2 * M * K * N
    hbm = (M * K + K * N + M * N) * DTYPE_BYTES["bf16"]
    peak = HW["910C_die"].cube_peak("bf16")
    # 实测时间 = 理论compute/0.88 (即 η=0.88, 接近 achievable) -> 满24AIC
    us = flops / peak / 0.88 * 1e6
    s = mfu_signal(flops, hbm, us, "910C_die", "bf16", "matmul", n_aic_used=24)
    check("near-ceiling done", s["done"] is True, f"mfu={s['current_mfu']} ach={s['achievable_mfu']}")
    check("near-ceiling current≈achievable", s["gap_to_achievable"] >= 0.8, f"got {s['gap_to_achievable']}")


def test_memory_bound_op_flags_AI_lever():
    """memory-bound 算子(瘦GEMV): 必须判 memory + AI 杠杆。"""
    M, K, N = 1, 4096, 4096
    flops = 2 * M * K * N
    hbm = (M * K + K * N + M * N) * DTYPE_BYTES["bf16"]
    us = hbm / HW["910C_die"].hbm_bw * 1e6  # 跑在 HBM 带宽上限
    s = mfu_signal(flops, hbm, us, "910C_die", "bf16", "matmul", n_aic_used=24)
    check("GEMV bottleneck=memory", s["bottleneck"] == "memory", f"got {s['bottleneck']}")
    check("GEMV lever 含 arithmetic intensity", any("arithmetic" in l or "AI" in l for l in s["levers"]))


def test_ceiling_never_exceeded():
    """天花板不被突破: 任何实测的 current_mfu 不应 > 1.0 (除非time太小=峰值口径错)。"""
    # 用实测 A3 matmul 321 TFLOPS (n=4096 bf16) vs 理论 363
    M = K = N = 4096
    flops = 2 * M * K * N
    us = flops / 321e12 * 1e6
    s = mfu_signal(flops, (M * K + K * N + M * N) * 2, us, "910C_die", "bf16", "matmul", n_aic_used=24)
    check("A3实测matmul MFU≈0.88", abs(s["current_mfu"] - 0.88) < 0.05, f"got {s['current_mfu']}")
    check("天花板不破 current_mfu<=1.0", s["current_mfu"] <= 1.0)


def test_model_matches_kb_fa_case():
    """模型对 KB 实测 FA case (B1H8S2048D128 fp16 causal): compute-bound, AI≈512, ceiling≈17us。"""
    fwd, _ = flash_attention_ops(1, 8, 2048, 128, "fp16", causal=True)
    r = solve_mfu(fwd, HW["950PR"])
    check("KB-FA compute-bound", r["bottleneck"] == "compute", f"got {r['bottleneck']}")
    check("KB-FA AI≈512", abs(r["arithmetic_intensity"] - 512) < 5, f"got {r['arithmetic_intensity']}")


def test_overhead_bound_flagged_and_corrected():
    """iter2教训: tiny mm_grad shape overhead-bound — 无overhead时须flag警告;
    传t_overhead后achievable下修, 近vendor水平的实测应判done。
    """
    M, K, N = 512, 256, 512
    flops = 2 * M * K * N * 2
    hbm = (M * K + K * N + M * N) * DTYPE_BYTES["fp16"] * 2
    # 无overhead: 应 flag overhead_bound + 警告lever (compute0.74us«67us)
    s = mfu_signal(flops, hbm, 67.2, "910C_die", "fp16", "matmul", n_aic_used=24)
    check("overhead_bound flagged", s.get("overhead_bound") is True, f"got {s.get('overhead_bound')}")
    check("warns flat-η over-states", any("overhead-bound" in l for l in s["levers"]))
    # 传t_overhead=65us: achievable下修到~vendor水平, v2(1.1%)≈achievable -> done
    s2 = mfu_signal(flops, hbm, 67.2, "910C_die", "fp16", "matmul", n_aic_used=24, t_overhead_us=65.0)
    check("overhead-corrected achievable low", s2["achievable_mfu"] < 0.05, f"got {s2['achievable_mfu']}")
    check("near-vendor tiny op -> done", s2["done"] is True, f"mfu={s2['current_mfu']} ach={s2['achievable_mfu']}")


def test_peak_source_per_op_class():
    """main accuracy-critical gate: peak 必须按算子类取对。matmul->CUBE, elementwise->VEC。
    用错 peak => MFU 错 => headroom 错 (PR#400-类 bug)。且 vec 缺 dtype 须 fail-loud 不静默回落 cube。
    """
    from optimizer_signal import _peak_for_op_class
    hw = HW["950PR"]
    cube = hw.cube_peak("fp16")           # ~378 TF
    vec = hw.vec_peak["fp16"]             # 54 TF
    check("matmul->CUBE peak", _peak_for_op_class(hw, "matmul", "fp16") == cube)
    check("elementwise->VEC peak", _peak_for_op_class(hw, "elementwise", "fp16") == vec)
    check("cube!=vec (否则测试无意义)", cube != vec)
    # fail-loud: vector op 缺 dtype 的 vec_peak 不静默回落 cube
    raised = False
    try:
        _peak_for_op_class(hw, "softmax", "fp8")  # fp8 不在 vec_peak
    except ValueError:
        raised = True
    check("vec缺dtype fail-loud(不回落cube)", raised)
    # unknown op_kind fail-loud
    raised2 = False
    try:
        _peak_for_op_class(hw, "mystery_op", "fp16")
    except ValueError:
        raised2 = True
    check("unknown op_kind fail-loud", raised2)


def test_eta_provenance_present():
    """main纪律: 每个标定的 η 须有 provenance(来源+日期+revalidate触发), 防 stale-η 错估。"""
    from mfu_model import eta_with_provenance

    cases = [
        ("matmul", "910C_die"),
        ("matmul", "950PR"),
        ("flash_attention", "910C_die"),
        ("flash_attention", "950PR"),
    ]
    for op, hw in cases:
        eta, prov = eta_with_provenance(op, hw)
        check(
            f"{op}/{hw} η有provenance",
            prov is not None and "measured" in prov and "revalidate_on" in prov,
            f"prov={prov}",
        )


def test_verification_hook_injects_mfu_ceiling():
    """机械注入: hook 读 verification.json -> 写 mfu_ceiling 块(ko 读的字段)。
    验证低-MFU op 被正确标 CONTINUE + multi-AIC lever, 且 mfu_ceiling 进了 json。
    """
    import json
    import tempfile
    from verification_hook import inject_mfu_ceiling
    # 造一个最小 verification.json(mm_grad 单AIC 实测形态)
    vj = {"op": "mm_grad",
          "precision": {"cases": [{"MKN": [512, 256, 512], "dtype": "float16"}]},
          "performance": {"ours_us_two_runs": {"[512, 256, 512]": [76.6, 76.6]}}}
    fd, p = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    Path(p).write_text(json.dumps(vj), encoding="utf-8")
    c = inject_mfu_ceiling(p, op_kind="mm_grad", hw_name="910C_die", n_aic_used=1)
    reloaded = json.loads(Path(p).read_text(encoding="utf-8"))
    os.unlink(p)
    check("mfu_ceiling written into json", "mfu_ceiling" in reloaded)
    sig = c["per_case"].get("[512,256,512]") or list(c["per_case"].values())[0]
    check("hook flags low-MFU CONTINUE", sig["done"] is False, f"got {sig}")
    check("hook surfaces multi-AIC lever", any("multi-AIC" in l for l in sig["levers"]))
    check("hook current_mfu ~1%", sig["current_mfu"] < 0.05, f"got {sig['current_mfu']}")


def test_pipeline_injection_via_phase_o5():
    """真链入(main gate 要求): phase_o5 perf 写完 verification.json 后, _inject_mfu_ceiling_safe
    机械注入 mfu_ceiling — 非死代码. op_kind 从 op-class 取(非写死). fail-safe.
    """
    import json
    import tempfile
    import shutil
    import sys as _s
    root = Path(__file__).resolve().parents[1]
    _s.path.insert(0, str(root / "src/scripts/orchestrator"))
    try:
        import phase_o5
    except Exception as e:
        check("phase_o5 importable", False, f"import failed: {e!r}")
        return
    ws = Path(tempfile.mkdtemp())
    try:
        vj = {"op": "mm_grad", "target": "a3", "soc_version": "Ascend910_9392",
              "precision": {"cases": [{"MKN": [512, 256, 512], "dtype": "float16"}]},
              "performance": {"ours_us_two_runs": {"[512, 256, 512]": [76.6, 76.6]}}}
        vp = ws / "verification.json"
        vp.write_text(json.dumps(vj))
        (ws / "op_classification.json").write_text(json.dumps({"class": "matmul_backward"}))
        check("op_kind from op-class (not hardcoded)", getattr(phase_o5, "_mfu_op_kind")(ws, "mm_grad") == "mm_grad")
        getattr(phase_o5, "_inject_mfu_ceiling_safe")(vp, ws, "mm_grad", vj)
        out = json.loads(vp.read_text())
        check("pipeline writes mfu_ceiling (not dead code)", "mfu_ceiling" in out)
        pc = out.get("mfu_ceiling", {}).get("per_case", {})
        sig = list(pc.values())[0] if pc else {}
        check("pipeline per_case has done-signal", sig.get("done") is False, f"got {sig}")
        # fail-safe: garbage verification never raises
        bad = ws / "bad.json"
        bad.write_text("{not json")
        getattr(phase_o5, "_inject_mfu_ceiling_safe")(bad, ws, "mm_grad", {})  # must not raise
        check("fail-safe on bad input (no raise)", True)
    finally:
        shutil.rmtree(ws)


def test_applicability_gate_abstains_on_vector():
    """owner-approved 安全 gate: vector/未验证 op-class -> 方向弃权(不发误导 lever), 仍给 ceiling/stop.
    复现 selective_scan 场景: 绝不出 raise-AI/fuse/multi-AIC(误导).
    """
    # selective_scan-like vector op (real numbers from independent review)
    flops = int(1.72e9)
    hbm = 225 * 2**20
    s = mfu_signal(flops, hbm, 7134.0, "950PR", "fp16", "softmax", n_aic_used=1)
    check("vector op -> applicable_direction=False",
          s["applicable_direction"] is False, f"got {s.get('applicable_direction')}")
    check("vector op -> NO misleading lever emitted", not any(("raise arithmetic" in l or "multi-AIC" in l)
          for l in s["levers"]), f"levers={s['levers']}")
    check("vector op -> direction_note explains abstain", s.get("direction_note") is not None)
    check("vector op -> verdict says ABSTAIN", "ABSTAIN" in s["verdict"])
    check("vector op -> ceiling still given (current_mfu present)", s["current_mfu"] > 0)
    # validated op (matmul) still emits levers (gate doesn't over-suppress)
    s2 = mfu_signal(2 * 512 * 256 * 512, (512 * 256 + 256 * 512 + 512 * 512)
                    * 2, 76.6, "910C_die", "fp16", "matmul", n_aic_used=1)
    check("matmul still applicable_direction=True", s2["applicable_direction"] is True)
    check("matmul still emits multi-AIC lever", any("multi-AIC" in l for l in s2["levers"]))


def test_multi_aic_lever_engine_aware():
    """multi-AIC lever 绝不出在 vector(AIV) 算子上(selective_scan bug: cube-28x 错用在 aic=0 的 AIV 算子)."""
    # 即便强行传 vector op + n_aic_used, 也因弃权不发 multi-AIC
    s = mfu_signal(int(1.72e9), 225 * 2**20, 7134.0, "950PR", "fp16", "elementwise", n_aic_used=1)
    check("vector op never gets cube multi-AIC lever",
          not any("multi-AIC" in l for l in s["levers"]), f"levers={s['levers']}")


def test_matmul_flops_accounting():
    """matmul fwd/bwd FLOPs 记账正确: bwd=2×fwd。"""
    fwd, bwd = matmul_ops(512, 512, 512, "bf16")
    check("matmul fwd flops=2MNK", fwd.flops == 2 * 512**3)
    check("matmul bwd flops=4MNK(2x fwd)", bwd.flops == 4 * 512**3)


if __name__ == "__main__":
    for t in [test_mm_grad_single_aic_flags_continue, test_near_ceiling_matmul_is_done,
              test_memory_bound_op_flags_AI_lever, test_ceiling_never_exceeded,
              test_model_matches_kb_fa_case, test_overhead_bound_flagged_and_corrected,
              test_peak_source_per_op_class, test_eta_provenance_present,
              test_verification_hook_injects_mfu_ceiling,
              test_pipeline_injection_via_phase_o5,
              test_applicability_gate_abstains_on_vector,
              test_multi_aic_lever_engine_aware,
              test_matmul_flops_accounting]:
        LOGGER.info("%s:", t.__name__)
        t()
    LOGGER.info("%s\n%s", "=" * 50,
                "ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
