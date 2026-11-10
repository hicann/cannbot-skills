#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compute per-shape theoretical lower bounds and rank shapes.

The model intentionally follows the user's rule:

    T_vector = vector instruction time
    T_copy   = GM/HBM transfer time
    T_hw     = max(T_vector, T_copy)
    headroom = T_candidate / T_hw

A Cube term is optional.  The script is operator-agnostic: all operator-specific
information lives in golden_flow.json.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    emit,
    ExprError,
    case_variables,
    canonical_dtype,
    die,
    dtype_bytes,
    eval_number,
    load_json,
    load_performance,
    read_cases,
    write_json,
)


def _dtype_for_instruction(inst: Dict[str, Any], case: Dict[str, Any]) -> str:
    for key in ("dtype", "src_dtype", "dst_dtype"):
        value = inst.get(key)
        if value and str(value) != "@in":
            return canonical_dtype(value)
    dtypes = case.get("dtype") or []
    return canonical_dtype(dtypes[0] if dtypes else "float32")


def _instruction_dtype_bytes(inst: Dict[str, Any], case: Dict[str, Any]) -> int:
    values = []
    for key in ("dtype", "src_dtype", "dst_dtype"):
        value = inst.get(key)
        if not value:
            continue
        if str(value) == "@in":
            dtypes = case.get("dtype") or []
            value = dtypes[0] if dtypes else "float32"
        values.append(dtype_bytes(value))
    return max(values) if values else dtype_bytes(_dtype_for_instruction(inst, case))


def _elements_per_cycle(
    inst: Dict[str, Any], dtype: str, hardware: Dict[str, Any]
) -> float:
    table = hardware.get("vector_elements_per_cycle") or {}
    key = canonical_dtype(dtype)
    if key in table:
        return float(table[key])
    if str(dtype) in table:
        return float(table[str(dtype)])
    width_bits = float(hardware.get("vector_width_bits", 2048))
    return max(1.0, width_bits / (8.0 * dtype_bytes(dtype)))


def _vector_instruction_row(idx, inst, costs, case, hardware, variables):
    if not isinstance(inst, dict):
        die(f"compute[{idx}] 必须是对象")
    kind = str(inst.get("kind", "vector"))
    if kind not in ("vector", "scalar"):
        return None
    op = str(inst.get("op", inst.get("name", "op")))
    try:
        elements = float(eval_number(inst.get("elements", 0), variables))
    except ExprError as exc:
        die(f"compute[{idx}].elements 求值失败: {exc}")
    if elements <= 0:
        return None
    dtype = _dtype_for_instruction(inst, case)
    epc = _elements_per_cycle(inst, dtype, hardware)
    cost = float(inst.get("cost", costs.get(op, costs.get("default", 1.0))))
    count = float(inst.get("count", 1) or 1)
    cycles = math.ceil(elements / epc) * cost * count
    return {
        "op": op,
        "kind": kind,
        "dtype": dtype,
        "elements": elements,
        "elements_per_cycle": epc,
        "cost": cost,
        "count": count,
        "unit": str(inst.get("unit", "vector")),
        "cycles": cycles,
    }


def _vector_time_us(
    flow: Dict[str, Any],
    case: Dict[str, Any],
    hardware: Dict[str, Any],
    variables: Dict[str, Any],
) -> Dict[str, Any]:
    compute = flow.get("compute")
    if compute is None:
        compute = (flow.get("vector") or {}).get("instructions") or []
    if not isinstance(compute, list):
        die("golden_flow.compute 必须是数组")
    costs = hardware.get("vector_instruction_costs") or {}
    units: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []
    total_cycles = 0.0
    for idx, inst in enumerate(compute):
        row = _vector_instruction_row(idx, inst, costs, case, hardware, variables)
        if row is None:
            continue
        unit = row["unit"]
        units[unit] = units.get(unit, 0.0) + row["cycles"]
        total_cycles += row["cycles"]
        rows.append(row)
    cores = float(variables.get("aiv_core_num", 1) or 1)
    freq = float(variables.get("frequency_hz", 1.0) or 1.0)
    if not units:
        return {"t_vector_us": 0.0, "vector_cycles": 0.0, "vector_rows": rows}
    unit_times = {u: (c / (cores * freq)) * 1e6 for u, c in units.items()}
    return {
        "t_vector_us": max(unit_times.values()),
        "vector_cycles": total_cycles,
        "vector_unit_times_us": unit_times,
        "vector_rows": rows,
    }


def _cube_time_us(
    flow: Dict[str, Any], variables: Dict[str, Any], hardware: Dict[str, Any]
) -> Dict[str, Any]:
    compute = flow.get("cube") or []
    if not isinstance(compute, list):
        die("golden_flow.cube 必须是数组")
    total = 0.0
    rows = []
    peaks = hardware.get("cube_peak_flops") or {}
    for idx, inst in enumerate(compute):
        if not isinstance(inst, dict):
            die(f"cube[{idx}] 必须是对象")
        dtype = canonical_dtype(inst.get("dtype", "float16"))
        try:
            flops = float(eval_number(inst.get("flops", 0), variables))
        except ExprError as exc:
            die(f"cube[{idx}].flops 求值失败: {exc}")
        if flops <= 0:
            continue
        peak = float(peaks.get(str(inst.get("dtype", "")), peaks.get(dtype, 0)) or 0)
        if peak <= 0:
            die(f"硬件档案缺少 cube 峰值算力: {dtype}")
        t_us = flops / peak * 1e6
        total = max(total, t_us)
        rows.append({"flops": flops, "dtype": dtype, "peak_flops": peak, "t_us": t_us})
    return {"t_cube_us": total, "cube_rows": rows}


def _memory_bytes(memory, variables):
    try:
        read_bytes = float(eval_number(memory.get("gm_read_bytes", 0), variables))
        write_bytes = float(eval_number(memory.get("gm_write_bytes", 0), variables))
        reread_bytes = float(eval_number(memory.get("gm_reread_bytes", 0), variables))
        working_set = float(
            eval_number(
                memory.get("working_set_bytes", read_bytes + write_bytes),
                dict(
                    variables,
                    gm_read_bytes=read_bytes,
                    gm_write_bytes=write_bytes,
                    gm_reread_bytes=reread_bytes,
                    total_bytes=read_bytes + write_bytes,
                ),
            )
        )
    except ExprError as exc:
        die(f"memory 表达式求值失败: {exc}")
    return read_bytes, write_bytes, reread_bytes, working_set


def _effective_bandwidth(model, hardware, working_set):
    hbm = float(hardware.get("hbm_bandwidth_bytes_s", 1.8e12))
    l2 = float(hardware.get("l2_bandwidth_bytes_s", hbm))
    l2_cap = float(hardware.get("l2_capacity_bytes", 0))
    if model == "hbm":
        return hbm
    if model == "l2":
        return l2
    if model == "auto":
        return l2 if (l2_cap > 0 and working_set <= l2_cap) else hbm
    die(f"不支持的 bandwidth_model: {model}")
    return hbm


def _engine_time_us(read_bytes, write_bytes, hardware, variables):
    cores = float(variables.get("aiv_core_num", 1) or 1)
    freq = float(variables.get("frequency_hz", 1.0) or 1.0)
    mte2_bpc = float(hardware.get("mte2_bytes_per_cycle", 0) or 0)
    mte3_bpc = float(hardware.get("mte3_bytes_per_cycle", 0) or 0)
    t_engine2 = read_bytes / (cores * mte2_bpc * freq) * 1e6 if mte2_bpc > 0 else 0.0
    t_engine3 = write_bytes / (cores * mte3_bpc * freq) * 1e6 if mte3_bpc > 0 else 0.0
    return t_engine2, t_engine3, max(t_engine2, t_engine3)


def _copy_time_us(
    flow: Dict[str, Any], variables: Dict[str, Any], hardware: Dict[str, Any]
) -> Dict[str, Any]:
    memory = flow.get("memory") or {}
    read_bytes, write_bytes, reread_bytes, working_set = _memory_bytes(
        memory, variables
    )
    model = str(flow.get("bandwidth_model", memory.get("bandwidth_model", "auto")))
    bw = _effective_bandwidth(model, hardware, working_set)
    total_bytes = read_bytes + write_bytes
    t_memory_us = total_bytes / bw * 1e6 if bw > 0 else 0.0
    t_engine2, t_engine3, t_engine_us = _engine_time_us(
        read_bytes, write_bytes, hardware, variables
    )
    return {
        "t_copy_us": max(t_memory_us, t_engine_us),
        "t_memory_us": t_memory_us,
        "t_mte2_engine_us": t_engine2,
        "t_mte3_engine_us": t_engine3,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "reread_bytes": reread_bytes,
        "working_set_bytes": working_set,
        "bandwidth_model": model,
        "effective_bandwidth_bytes_s": bw,
    }


def _perf_key(cid, perf):
    if cid in perf:
        return cid
    tail = cid.rsplit("_", 1)[-1]
    return tail if tail in perf else cid


def _case_record(case, perf, flow, hardware):
    cid = str(case["case_id"])
    key = _perf_key(cid, perf)
    if key not in perf:
        die(f"profiling 中找不到 case {cid}")
    variables = case_variables(case, hardware)
    vec = _vector_time_us(flow, case, hardware, variables)
    copy = _copy_time_us(flow, variables, hardware)
    cube = _cube_time_us(flow, variables, hardware)
    t_hw = max(vec["t_vector_us"], copy["t_copy_us"], cube["t_cube_us"])
    if t_hw <= 0:
        die(f"case {cid} 的理论耗时为 0，请检查 golden_flow")
    t_candidate = float(perf[key]["t_candidate_us"])
    headroom = t_candidate / t_hw
    if copy["t_copy_us"] >= max(vec["t_vector_us"], cube["t_cube_us"]):
        bound = "copy"
    elif cube["t_cube_us"] >= vec["t_vector_us"]:
        bound = "cube"
    else:
        bound = "vector"
    t_floor = max(t_hw, float(hardware.get("launch_floor_us", 0.0) or 0.0))
    return {
        "case_id": cid,
        "shape": case.get("input_shape", []),
        "dtype": case.get("dtype", []),
        "attrs": case.get("attrs", {}),
        "t_candidate_us": t_candidate,
        "t_vector_us": round(vec["t_vector_us"], 6),
        "t_copy_us": round(copy["t_copy_us"], 6),
        "t_cube_us": round(cube["t_cube_us"], 6),
        "t_hw_us": round(t_hw, 6),
        "t_hw_floor_us": round(t_floor, 6),
        "headroom": round(headroom, 6),
        "bound_kind": bound,
        "vector_cycles": round(vec.get("vector_cycles", 0.0), 6),
        "vector_unit_times_us": vec.get("vector_unit_times_us", {}),
        "vector_rows": vec.get("vector_rows", []),
        "copy": copy,
        "cube_rows": cube.get("cube_rows", []),
        "perf": perf[key],
    }


def _summary(out_cases):
    return {
        "n_cases": len(out_cases),
        "worst_case_id": out_cases[0]["case_id"] if out_cases else None,
        "worst_headroom": out_cases[0]["headroom"] if out_cases else None,
        "bound_distribution": {
            b: sum(1 for r in out_cases if r["bound_kind"] == b)
            for b in ("vector", "copy", "cube")
        },
    }


def compute_limits(
    cases: List[Dict[str, Any]],
    perf: Dict[str, Dict[str, float]],
    flow: Dict[str, Any],
    hardware: Dict[str, Any],
    op_name: str = "",
) -> Dict[str, Any]:
    out_cases = []
    for case in cases:
        out_cases.append(_case_record(case, perf, flow, hardware))
    out_cases.sort(key=lambda r: (-r["headroom"], -r["t_candidate_us"], r["case_id"]))
    for index, record in enumerate(out_cases, 1):
        record["rank"] = index
    return {
        "schema_version": 1,
        "operator": op_name,
        "hardware": {
            "name": hardware.get("name", hardware.get("chip", "unknown")),
            "chip": hardware.get("chip", hardware.get("name", "unknown")),
            "aiv_core_num": hardware.get("aiv_core_num"),
            "frequency_hz": hardware.get("frequency_hz"),
            "ub_size_bytes": hardware.get("ub_size_bytes"),
        },
        "flow": flow,
        "cases": out_cases,
        "summary": _summary(out_cases),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="计算 shape 理论下限并排名")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--perf", required=True, help="profiling performance.json")
    ap.add_argument("--flow", required=True, help="golden_flow.json")
    ap.add_argument("--hardware", required=True, help="hardware profile JSON")
    ap.add_argument("--operator", default="")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    cases = read_cases(args.cases, args.operator)
    perf = load_performance(args.perf)
    if not perf:
        die(f"未从 profiling 文件解析到任何 case: {args.perf}")
    flow = load_json(args.flow)
    if not isinstance(flow, dict):
        die(f"golden_flow 必须是 JSON 对象: {args.flow}")
    hardware = load_json(args.hardware)
    if not isinstance(hardware, dict):
        die(f"hardware profile 必须是 JSON 对象: {args.hardware}")
    op = args.operator or flow.get("operator") or (cases[0] or {}).get("operator", "")
    result = compute_limits(cases, perf, flow, hardware, op)
    write_json(args.output, result)
    s = result["summary"]
    emit(
        f"[shape_limits] {s['n_cases']} 个 shape，最差 {s['worst_case_id']} "
        f"headroom={s['worst_headroom']} → {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
