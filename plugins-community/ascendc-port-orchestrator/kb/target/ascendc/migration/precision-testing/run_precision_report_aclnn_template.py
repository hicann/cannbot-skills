#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Run precision evaluation for {{OP_NAME}} on ascend950 and generate report.
Uses aclnn interface for operator invocation.

Template placeholders (replace before use):
  {{OP_NAME}}                  -> operator name, e.g. acosh
  {{NPU_CALL}}                 -> NPU invocation expr using `x`, e.g. torch.ops.npu.acosh(x)
  {{CPU_REF}}                  -> CPU reference expr using `x` and `dtype`, e.g. torch.acosh(x.cpu().float()).to(dtype)
  {{SUPPORTED_DTYPES}}         -> dtype list, e.g. [torch.float16, torch.float32]
  {{INPUT_LOW}}                -> domain lower bound for random input, e.g. 1.0
  {{INPUT_HIGH}}               -> domain upper bound for random input, e.g. 11.0
  {{TEST_SHAPES}}              -> list of (category, description, shape) tuples
  {{BOUNDARY_VALUES}}          -> list of (description, scalar_value) tuples for boundary tests
  {{QUANT_DTYPE}}              -> whether operator involves quantized types, e.g. False
  {{DETERMINISM_SHAPE}}        -> shape for determinism test, e.g. (8, 16)
  {{DETERMINISM_RUNS}}         -> number of repeated runs for determinism test, e.g. 3
  {{TOTAL_CASES}}              -> total number of test cases (pre-computed)
  {{NUM_STAGES}}               -> total number of test stages (including extra)
  {{EXTRA_TEST_CONFIGS}}       -> operator-specific test config variables (can be empty)
  {{EXTRA_TEST_RUN_BLOCKS}}    -> operator-specific test run code blocks in main() (can be empty)
"""

import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch_npu

device = torch.device("npu:0")
LOGGER = logging.getLogger(__name__)

SUPPORTED_DTYPES = {{SUPPORTED_DTYPES}}

THRESHOLD = {
    torch.float32: 2**-13,
    torch.float16: 2**-10,
    torch.bfloat16: 2**-7,
    torch.float8_e4m3fn: 2**-3,
    torch.float8_e5m2: 2**-2,
}

QUANT_DTYPE = {{QUANT_DTYPE}}

DTYPE_NAMES = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float8_e4m3fn: "float8_e4m3fn",
    torch.float8_e5m2: "float8_e5m2",
    torch.int8: "int8",
    torch.int32: "int32",
}

TEST_SHAPES = {{TEST_SHAPES}}
BOUNDARY_VALUES = {{BOUNDARY_VALUES}}
BOUNDARY_SHAPE = (1024,)

DETERMINISM_SHAPE = {{DETERMINISM_SHAPE}}
DETERMINISM_RUNS = {{DETERMINISM_RUNS}}

TOTAL_CASES = {{TOTAL_CASES}}
NUM_STAGES = {{NUM_STAGES}}

{{EXTRA_TEST_CONFIGS}}


def log(msg):
    LOGGER.info("%s", msg)


def make_random(shape, dtype):
    x = torch.rand(shape, dtype=torch.float32) * ({{INPUT_HIGH}} - {{INPUT_LOW}}) + {{INPUT_LOW}}
    return x.to(dtype=dtype, device=device)


def make_constant(shape, value, dtype):
    x = torch.full(shape, float(value), dtype=torch.float32)
    return x.to(dtype=dtype, device=device)


def compute_metrics(npu_out, cpu_ref):
    npu_f = npu_out.cpu().float()
    ref_f = cpu_ref.float()
    abs_err = (npu_f - ref_f).abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    rel_err = abs_err / (ref_f.abs() + 1e-7)
    mare = rel_err.max().item()
    mere = rel_err.mean().item()
    cos = torch.nn.functional.cosine_similarity(
        npu_f.flatten().unsqueeze(0), ref_f.flatten().unsqueeze(0)
    ).item()
    return max_abs, mean_abs, mare, mere, cos


def run_one(case_id, cat, desc, x, dtype):
    thresh = THRESHOLD.get(dtype, 2**-10)
    dtype_name = DTYPE_NAMES.get(dtype, str(dtype))

    try:
        npu_result = {{NPU_CALL}}
        cpu_ref = {{CPU_REF}}
        max_abs, mean_abs, mare, mere, cos = compute_metrics(npu_result, cpu_ref)
        if QUANT_DTYPE and dtype in (torch.int8, torch.int32):
            passed = max_abs == 0
        else:
            passed = (mere < thresh) and (mare < 10 * thresh)
    except Exception as e:
        max_abs = mean_abs = mare = mere = -1.0
        cos = -1.0
        passed = False
        LOGGER.error("  [CASE %02d] ERROR: %s", case_id, e)

    status = "PASS" if passed else "FAIL"
    shape_list = list(x.shape)
    numel = x.numel()
    log(f"  [{status}] Case {case_id:02d}/{TOTAL_CASES} {cat}/{desc} | shape={shape_list} "
        f"dtype={dtype_name} "
        f"| MERE={mere:.2e} MARE={mare:.2e} MaxAbsErr={max_abs:.2e} CosSim={cos:.10f}")

    return {
        "case_id": case_id,
        "category": cat,
        "description": desc,
        "shape": str(shape_list),
        "dtype": dtype_name,
        "numel": numel,
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
        "MARE": mare,
        "MERE": mere,
        "cosine_sim": cos,
        "threshold": thresh,
        "passed": passed,
    }, passed


CATEGORY_DISPLAY_ORDER = [
    "1D", "2D", "3D", "4D", "5D",
    "Boundary", "Determinism",
]


def append_report_overview(lines, op_name, pass_count, fail_count):
    total = pass_count + fail_count
    pass_rate = f"{pass_count/total*100:.1f}" if total > 0 else "0.0"
    lines.extend([
        f"# {op_name} 算子精度验证报告（ascend950）",
        "",
        "**测试平台**: Ascend 950",
        "**调用接口**: aclnn",
        "**精度标准**: 生态算子开源精度标准（MERE/MARE）",
        "",
        "## 总览",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| 总用例数 | {total} |",
        f"| 通过数 | {pass_count} |",
        f"| 失败数 | {fail_count} |",
        f"| 通过率 | {pass_rate}% |",
        "",
    ])


def append_threshold_section(lines):
    lines.extend([
        "## 精度阈值标准",
        "",
        "通过条件：MERE < Threshold **且** MARE < 10 × Threshold",
        "",
        "| dtype | Threshold | MERE 上限 | MARE 上限 (10×) |",
        "|-------|-----------|----------|----------------|",
    ])

    for dtype in SUPPORTED_DTYPES:
        dn = DTYPE_NAMES.get(dtype, str(dtype))
        thr = THRESHOLD.get(dtype, 2**-10)
        if QUANT_DTYPE and dtype in (torch.int8, torch.int32):
            lines.append(f"| {dn} | 0 (精确匹配) | 0 | 0 |")
        else:
            lines.append(f"| {dn} | {thr:.2e} | {thr:.2e} | {10*thr:.2e} |")
    lines.append("")


def append_determinism_rows(lines, category_results):
    lines.append("| # | 描述 | Shape | dtype | MERE | MARE | CosSim | 一致性 | 结果 |")
    lines.append("|---|------|-------|-------|------|------|--------|--------|------|")
    for result in category_results:
        status = "✅" if result["passed"] else "❌"
        lines.append(f"| {result['case_id']:02d} | {result['description']} | {result['shape']} | {result['dtype']} | "
                     f"{result['MERE']:.2e} | {result['MARE']:.2e} | {result['cosine_sim']:.10f} | "
                     f"{'是' if result['passed'] else '否'} | {status} |")


def append_boundary_rows(lines, category_results):
    lines.append("| # | 描述 | dtype | MERE | MARE | MaxAbsErr | CosSim | 结果 |")
    lines.append("|---|------|-------|------|------|-----------|--------|------|")
    for result in category_results:
        status = "✅" if result["passed"] else "❌"
        lines.append(f"| {result['case_id']:02d} | {result['description']} | {result['dtype']} | "
                     f"{result['MERE']:.2e} | {result['MARE']:.2e} | {result['max_abs_err']:.2e} | "
                     f"{result['cosine_sim']:.10f} | {status} |")


def append_regular_rows(lines, category_results):
    lines.append("| # | 描述 | Shape | dtype | 元素数 | MERE | MARE | MaxAbsErr | CosSim | 结果 |")
    lines.append("|---|------|-------|-------|--------|------|------|-----------|--------|------|")
    for result in category_results:
        status = "✅" if result["passed"] else "❌"
        lines.append(f"| {result['case_id']:02d} | {result['description']} | {result['shape']} | {result['dtype']} | "
                     f"{result['numel']} | {result['MERE']:.2e} | {result['MARE']:.2e} | {result['max_abs_err']:.2e} | "
                     f"{result['cosine_sim']:.10f} | {status} |")


def append_category_sections(lines, results):
    by_cat = defaultdict(list)
    for result in results:
        by_cat[result["category"]].append(result)

    ordered_cats = [c for c in CATEGORY_DISPLAY_ORDER if c in by_cat]
    remaining_cats = [c for c in by_cat if c not in CATEGORY_DISPLAY_ORDER]
    for category in ordered_cats + remaining_cats:
        category_results = by_cat[category]
        lines.append(f"## {category} 测试结果")
        lines.append("")
        if category == "Determinism":
            append_determinism_rows(lines, category_results)
        elif category == "Boundary":
            append_boundary_rows(lines, category_results)
        else:
            append_regular_rows(lines, category_results)
        lines.append("")


def append_dtype_summary(lines, results):
    by_dtype = defaultdict(list)
    for result in results:
        by_dtype[result["dtype"]].append(result)
    lines.extend([
        "## 按 dtype 汇总统计",
        "",
        "| dtype | 用例数 | 通过数 | 失败数 | MERE 范围 | MARE 范围 | CosSim 范围 |",
        "|-------|--------|--------|--------|----------|----------|-------------|",
    ])
    for dtype_name in [DTYPE_NAMES.get(d, str(d)) for d in SUPPORTED_DTYPES]:
        dtype_results = by_dtype.get(dtype_name, [])
        if not dtype_results:
            continue
        n = len(dtype_results)
        n_pass = sum(1 for r in dtype_results if r["passed"])
        n_fail = n - n_pass
        meres = [r["MERE"] for r in dtype_results if r["MERE"] >= 0]
        mares = [r["MARE"] for r in dtype_results if r["MARE"] >= 0]
        coss = [r["cosine_sim"] for r in dtype_results if r["cosine_sim"] >= 0]
        mere_range = f"[{min(meres):.2e}, {max(meres):.2e}]" if meres else "N/A"
        mare_range = f"[{min(mares):.2e}, {max(mares):.2e}]" if mares else "N/A"
        cos_range = f"[{min(coss):.6f}, {max(coss):.6f}]" if coss else "N/A"
        lines.append(f"| {dtype_name} | {n} | {n_pass} | {n_fail} | {mere_range} | {mare_range} | {cos_range} |")
    lines.append("")


def append_key_findings(lines, results):
    lines.extend(["## 关键发现", ""])
    failed_results = [r for r in results if not r["passed"]]
    if failed_results:
        lines.append(f"1. **存在 {len(failed_results)} 个失败用例**，需排查精度问题")
        failed_dtypes = set(r["dtype"] for r in failed_results)
        lines.append(f"2. **失败 dtype**: {', '.join(sorted(failed_dtypes))}")
        for r in failed_results[:5]:
            lines.append(f"3. Case {r['case_id']:02d} ({r['category']}/{r['description']}): "
                        f"MERE={r['MERE']:.2e} MARE={r['MARE']:.2e}")
    else:
        lines.append("1. **全部用例通过**，精度验证达标")
        lines.append("2. **各 dtype 精度特征**: 均在阈值范围内")
        det_results = [r for r in results if r["category"] == "Determinism"]
        if det_results and all(r["passed"] for r in det_results):
            lines.append("3. **确定性验证通过**: 多次运行结果完全一致，无随机性")
        lines.append("4. **生产可用性**: 精度验证通过，可进入生产环境")
    lines.append("")


def generate_markdown_report(results, op_name, pass_count, fail_count):
    lines = []
    append_report_overview(lines, op_name, pass_count, fail_count)
    append_threshold_section(lines)
    append_category_sections(lines, results)
    append_dtype_summary(lines, results)
    append_key_findings(lines, results)
    return "\n".join(lines)


def log_evaluation_header():
    log(f"{'='*80}")
    log(f"  {{OP_NAME}} Precision Evaluation (ascend950)")
    log(f"  Total cases to run: {TOTAL_CASES}")
    log(f"{'='*80}")


def run_regular_shape_tests(results, case_id):
    pass_count = 0
    fail_count = 0
    log(f"\n--- [1/{NUM_STAGES}] Regular Shape Tests ({len(TEST_SHAPES) * len(SUPPORTED_DTYPES)} cases) ---")
    for cat, desc, shape in TEST_SHAPES:
        for dtype in SUPPORTED_DTYPES:
            case_id += 1
            x = make_random(shape, dtype)
            r, ok = run_one(case_id, cat, desc, x, dtype)
            results.append(r)
            if ok:
                pass_count += 1
            else:
                fail_count += 1
    return case_id, pass_count, fail_count


def run_boundary_tests(results, case_id):
    pass_count = 0
    fail_count = 0
    log(f"\n--- [2/{NUM_STAGES}] Boundary Tests ({len(BOUNDARY_VALUES) * len(SUPPORTED_DTYPES)} cases) ---")
    for desc, value in BOUNDARY_VALUES:
        for dtype in SUPPORTED_DTYPES:
            case_id += 1
            x = make_constant(BOUNDARY_SHAPE, value, dtype)
            r, ok = run_one(case_id, "Boundary", desc, x, dtype)
            results.append(r)
            if ok:
                pass_count += 1
            else:
                fail_count += 1
    return case_id, pass_count, fail_count


def run_determinism_tests(results, case_id):
    pass_count = 0
    fail_count = 0
    log(f"\n--- [3/{NUM_STAGES}] Determinism Tests ({len(SUPPORTED_DTYPES)} cases) ---")
    for dtype in SUPPORTED_DTYPES:
        case_id += 1
        x = make_random(DETERMINISM_SHAPE, dtype)
        try:
            run_results = []
            for _ in range(DETERMINISM_RUNS):
                run_results.append({{NPU_CALL}}.clone())
            det_pass = all(torch.equal(run_results[0], run_results[i]) for i in range(1, len(run_results)))
            npu_result = run_results[0]
            cpu_ref = {{CPU_REF}}
            max_abs, mean_abs, mare, mere, cos = compute_metrics(npu_result, cpu_ref)
        except Exception as e:
            max_abs = mean_abs = mare = mere = -1.0
            cos = -1.0
            det_pass = False
            LOGGER.error("  [CASE %02d] ERROR: %s", case_id, e)
        dtype_name = DTYPE_NAMES.get(dtype, str(dtype))
        thresh = THRESHOLD.get(dtype, 2**-10)
        status = "PASS" if det_pass else "FAIL"
        log(f"  [{status}] Case {case_id:02d}/{TOTAL_CASES} Determinism/{DETERMINISM_RUNS} runs | "
            f"shape={list(DETERMINISM_SHAPE)} dtype={dtype_name} | consistent={det_pass}")
        results.append({
            "case_id": case_id, "category": "Determinism",
            "description": f"{DETERMINISM_RUNS} runs consistency",
            "shape": str(list(DETERMINISM_SHAPE)), "dtype": dtype_name,
            "numel": int(np.prod(DETERMINISM_SHAPE)),
            "max_abs_err": max_abs, "mean_abs_err": mean_abs,
            "MARE": mare, "MERE": mere, "cosine_sim": cos,
            "threshold": thresh, "passed": det_pass,
        })
        if det_pass:
            pass_count += 1
        else:
            fail_count += 1
    return case_id, pass_count, fail_count


def write_reports(results, pass_count, fail_count):
    log(f"\n--- Generating reports ---")
    report_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(report_dir, "{{OP_NAME}}_precision_report.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"  JSON saved: {json_path}")

    md_report = generate_markdown_report(results, "{{OP_NAME}}", pass_count, fail_count)
    md_path = os.path.join(report_dir, "{{OP_NAME}}_precision_report.md")
    with open(md_path, "w") as f:
        f.write(md_report)
    log(f"  MD saved:   {md_path}")


def log_evaluation_summary(pass_count, fail_count):
    total = pass_count + fail_count
    log(f"\n{'='*80}")
    log(f"  {{OP_NAME}} Precision Evaluation Summary (ascend950)")
    log(f"  Total: {total} | Passed: {pass_count} | Failed: {fail_count}")
    log(f"  Pass Rate: {pass_count/total*100:.1f}%")
    log(f"{'='*80}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    log_evaluation_header()

    results = []
    pass_count = 0
    fail_count = 0
    case_id = 0

    case_id, stage_pass, stage_fail = run_regular_shape_tests(results, case_id)
    pass_count += stage_pass
    fail_count += stage_fail
    case_id, stage_pass, stage_fail = run_boundary_tests(results, case_id)
    pass_count += stage_pass
    fail_count += stage_fail
    case_id, stage_pass, stage_fail = run_determinism_tests(results, case_id)
    pass_count += stage_pass
    fail_count += stage_fail

    {{EXTRA_TEST_RUN_BLOCKS}}

    write_reports(results, pass_count, fail_count)
    log_evaluation_summary(pass_count, fail_count)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
