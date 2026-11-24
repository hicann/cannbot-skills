# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""骨架测量驱动器：复制到 `tools/skeleton_profile.py` 后只改 `skeleton_spec.json`。

脚本已内置 develop SKILL 的骨架约束，不再由 agent 现写：

* `--dry-run`：读设备容量 / 核数 / L2（读取失败回退 90% 保守常量并标注来源），按硬约束与
  时间模型给出候选组、预测 ms、预测比值，不上板；
* 正常：按 `--tile` 编译 `tools/skeleton_gemm.py` 的 kernel，跑 device event 计时、与同 shape
  `torch.mm` 配对、按 `Step Id` 分轮解析 profile；只有对应管线利用率 ≥0.95 时才由实测反推
  并写入标定缓存，未饱和只留占位常数用于排序；
* `--ledger-row`：按 `ITERATIONS.md` 模板表头输出可直接粘贴的两行（性能行 + 名义搬运行）。

三组候选的来历（不枚举扫描）：按设备容量反推出 base tile 作为预测最优组，再由它派生 `BK`
减半方向对照与块映射对照（`nn` 可整除时取另一个因子，否则 `BM` 减半）。被硬约束淘汰的组合
只打印原因，不占实跑名额。

本脚本不绑定具体 SoC 与具体算子：容量、核数、L2 从设备读，峰值与带宽来自标定缓存，tile 由
容量反推（`--tile` 可覆盖）；算子相关的只有 `tools/skeleton_spec.json` 的 stage 与
`tools/skeleton_gemm.py` 的 kernel 体。输入 shape 从 spec 读取，例如：

    {"dtype": "bf16",
     "stages": [{"name": "gemm1", "m": 4096, "n": 8192, "k": 4096,
                 "layout_a": "row", "layout_b": "col", "source": "<用例 uuid>"}]}

batch 维不进 spec：按单 batch 的最大 shape 测，总预算按 batch 数外推，并在 ITERATIONS 写明
外推口径；每个 stage 可按不同 shape 分别跑一次。

`layout_a` / `layout_b` 取 `row` 或 `col`，对应 host 端 `from_dlpack` 的 `layout_tag`：`col`
表示物理按转置存（A 为 K×M、B 为 N×K），逻辑形状仍是 `[M, K]` 与 `[K, N]`，kernel 内不需要
任何转置分支；`C` 固定 row（逻辑 `[M, N]`）。

标定常数按 `(SoC, CANN, dtype, 映射档)` 缓存在 `.catlass-python/skeleton_calibration.json`，
命中时直接用于排序；标定不成立时保留占位值，只排序不淘汰。
"""

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch_npu  # noqa: F401

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

sys.path.insert(0, str(Path(__file__).resolve().parent))

from skeleton_gemm import Tiling, gemm_pipelined  # noqa: E402

# 占位常数：只在没有标定缓存时用于排序与 dry-run 展示，不参与淘汰。
# 平台相关的容量、核数、L2 一律从设备读取（见 device_limits），这里不绑定具体 SoC；
# 占位值的绝对量级不可跨平台引用，pred_ms 只用于同一设备内比较候选。
PEAK_TFLOPS = 400.0
BW_MTE2_TBS = 2.85
BW_MTE1_TBS = 2.85
BW_FIX_TBS = 2.0

# tile 候选阶梯：由设备容量从大到小筛，命中即用，不写死平台值。
# 阶梯起点取常见大核容量；L0C/L1 明显更大的设备可在副本里向上加档，并在 ITERATIONS 记原因。
TILE_LADDER = ((256, 256), (192, 256), (128, 256), (128, 128), (64, 128))
BK_LADDER = (128, 64, 32)

# 正确性冒烟容差按 dtype 取，避免用 bf16 的松容差去判 fp32。
ATOL_RTOL = {"bf16": (0.2, 0.02), "fp16": (0.05, 0.01), "fp32": (1e-3, 1e-4)}

ELEM_BYTES = {
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "fp32": 4,
    "float32": 4,
}

CACHE_PATH = Path(".catlass-python/skeleton_calibration.json")


@dataclass(frozen=True)
class Cand:
    name: str
    bm: int
    bn: int
    bk: int
    sk: int
    sm: int
    sn: int


def soc_info(device):
    info = {"device": "npu:{}".format(device)}
    try:
        info["soc"] = torch.npu.get_device_name(device)
    except Exception:  # noqa: BLE001
        info["soc"] = "unknown"
    # 部分版本 torch_npu.version.cann 为 None，退到包版本，避免缓存键退化成 unknown。
    info["cann"] = (
        getattr(getattr(torch_npu, "version", None), "cann", None)
        or getattr(torch_npu, "__version__", None)
        or "unknown"
    )
    return info


def device_limits(device):
    """容量与核数优先从设备读取；失败时回退保守常量并记录来源。"""
    info = {"source": "device"}
    try:
        props = torch.npu.get_device_properties(device)
        info["cube_core_num"] = None
        for attr in ("cube_core_num", "ai_core_num", "multi_processor_count"):
            value = int(getattr(props, attr, 0) or 0)
            if value:
                info["cube_core_num"] = value
                break
        info["vector_core_num"] = int(getattr(props, "vector_core_num", 0)) or None
    except Exception:  # noqa: BLE001
        info["cube_core_num"] = None
        info["vector_core_num"] = None

    for key, space in (
        ("l1_bytes", tla.AddressSpace.l1),
        ("l0a_bytes", tla.AddressSpace.l0a),
        ("l0b_bytes", tla.AddressSpace.l0b),
        ("l0c_bytes", tla.AddressSpace.l0c),
        ("ub_bytes", tla.AddressSpace.ub),
    ):
        try:
            info[key] = int(tla.arch.get_capacity_in_bytes(space))
        except Exception:  # noqa: BLE001
            info[key] = None

    l2 = None
    try:
        props = torch.npu.get_device_properties(device)
        for attr in ("l2_cache_size", "L2_cache_size"):
            if hasattr(props, attr):
                l2 = int(getattr(props, attr))
    except Exception:  # noqa: BLE001
        l2 = None
    info["l2_bytes"] = l2

    if any(
        info.get(k) is None for k in ("l1_bytes", "l0a_bytes", "l0b_bytes", "l0c_bytes")
    ):
        # 回退值只是一类设备的兜底容量，正常路径必须命中设备读取；命中回退时
        # source 标成 fallback，预筛结论要按目标平台重核。
        info["source"] = "fallback"
        info["l1_bytes"] = info.get("l1_bytes") or 512 * 1024
        info["l0a_bytes"] = info.get("l0a_bytes") or 64 * 1024
        info["l0b_bytes"] = info.get("l0b_bytes") or 64 * 1024
        info["l0c_bytes"] = info.get("l0c_bytes") or 256 * 1024
        info["ub_bytes"] = info.get("ub_bytes") or 256 * 1024
    if info.get("cube_core_num") is None:
        info["cube_core_num"] = 1
        info["source"] = "fallback"
    return info


def cache_key(info, dtype, tile):
    return "{soc}|{cann}|{dtype}|{tile}".format(
        soc=info.get("soc"), cann=info.get("cann"), dtype=dtype, tile=tile
    )


def load_cache():
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def divisors(n, cap=16):
    """能整除 n 的块映射候选，降序，只取不大于 cap 的因子。"""
    return sorted(
        (d for d in range(1, min(n, cap) + 1) if n and n % d == 0), reverse=True
    )


def mapping(nm, nn):
    """块映射按整除规则算，不占实跑名额；返回 (sm, sn, 候选 sn 列表)。"""

    def pick(n):
        if n <= 0:
            return 8, []
        if math.ceil(n / 8) * 8 / n - 1 > 0.25:
            cands = divisors(n, cap=16)
            if len(cands) <= 1:
                return 8, []
            return cands[0], cands
        return 8, []

    sm, _ = pick(nm)
    sn, sn_cands = pick(nn)
    return sm, sn, sn_cands


def sk_for(bk):
    """SK 跟随 BK：BK 较大时取 32，BK 很小时减半，保证 BK > SK 且整除。"""
    return 32 if bk >= 64 else bk // 2


def col_a_k_phase_ok(k, bk, sk):
    """列主序 A 的 K 必须让 SK 子块数按 BK/SK 闭合。

    实测（950PR，bf16，tile 256/256/128/32）：ceil(K/SK) 不是 BK/SK 的整数倍时，
    LOAD2D 写 L0A 越界（aicore error 5,131）或直接挂死。K=51350（1605 % 4 = 1）
    必挂，51456（1608 % 4 = 0）通过；换 tile 到 BK/SK=2 时同一 K 的通过性随
    ceil(K/64) % 2 变化，证明约束是子块相位而非 128/256 倍数或元素对齐。
    """
    if bk <= 0 or sk <= 0 or bk % sk:
        return True
    return ((k + sk - 1) // sk) % (bk // sk) == 0


def pad_k_for_col_a(k, bk):
    """补到 BK 倍数即可满足相位约束（BK 倍数是最省事的充分条件）。"""
    return (k + bk - 1) // bk * bk


def fits(bm, bn, bk, sk, limits, elem):
    """模板固定为双缓冲 + fp32 L0C 累加，这里只按设备容量判可行性。"""
    return (
        bm % 16 == 0
        and bn % 16 == 0
        and bm * bn * 4 <= limits["l0c_bytes"]
        and 2 * bm * sk * elem <= limits["l0a_bytes"]
        and 2 * sk * bn * elem <= limits["l0b_bytes"]
        and 2 * (bm * bk + bk * bn) * elem <= 0.9 * limits["l1_bytes"]
        and bk > sk
        and bk % sk == 0
    )


def base_tile(limits, elem):
    """按设备容量反推默认 tile，不写死某一平台的固定值。"""
    for bm, bn in TILE_LADDER:
        for bk in BK_LADDER:
            sk = sk_for(bk)
            if fits(bm, bn, bk, sk, limits, elem):
                return bm, bn, bk, sk
    return None


def candidates(m, n, limits, elem):
    base = base_tile(limits, elem)
    if base is None:
        return []
    bm, bn, bk, sk = base
    nm, nn = math.ceil(m / bm), math.ceil(n / bn)
    sm, sn, sn_cands = mapping(nm, nn)
    out = [Cand("predicted-best", bm, bn, bk, sk, sm, sn)]
    hb, hs = bk // 2, sk_for(bk // 2)
    if fits(bm, bn, hb, hs, limits, elem):
        out.append(Cand("bk-half", bm, bn, hb, hs, sm, sn))
    other = next((d for d in sn_cands if d != sn), None)
    if other:
        out.append(Cand("map-contrast", bm, bn, bk, sk, sm, other))
    elif fits(bm // 2, bn, bk, sk, limits, elem):
        out.append(Cand("bm-half", bm // 2, bn, bk, sk, sm, sn))
    return out


def prescreen(
    m, n, k, limits, elem, peak, bw_mte1, bw_mte2, bw_fix, extra=None, calibrated=False
):
    rows = []
    cands = candidates(m, n, limits, elem)
    if extra:
        cands += [c for c in extra if c.name not in {e.name for e in cands}]
    if not cands:
        return rows
    # 流水强度门槛依赖峰值与带宽：只有标定成立（实测饱和）时才用于淘汰，
    # 占位常数只参与排序，否则会按别的平台的比值误杀可行配置。
    required_intensity = peak / bw_mte2 if calibrated else None
    for cand in cands:
        bm, bn, bk, sk = cand.bm, cand.bn, cand.bk, cand.sk
        reason = None
        if bm % 16 or bn % 16:
            reason = "BM/BN 不是 16 的倍数"
        elif bm * bn * 4 > limits["l0c_bytes"]:
            reason = "L0C 超限"
        elif 2 * bm * sk * elem > limits["l0a_bytes"]:
            reason = "L0A 超限"
        elif 2 * sk * bn * elem > limits["l0b_bytes"]:
            reason = "L0B 超限"
        elif 2 * (bm * bk + bk * bn) * elem > 0.9 * limits["l1_bytes"]:
            reason = "L1 超限"
        elif not (bk > sk and bk % sk == 0):
            reason = "BK/SK 关系非法"
        elif (
            required_intensity is not None
            and 2 * bm * bn / (bm + bn) < required_intensity
        ):
            reason = "流水强度不足"
        mp, np_, kp = math.ceil(m / bm), math.ceil(n / bn), math.ceil(k / bk)
        bytes_a = np_ * mp * bm * kp * bk * elem
        bytes_b = mp * np_ * bk * bn * kp * elem
        bytes_l0 = mp * np_ * kp * (bm * bk + bk * bn) * elem
        t_mac = 2 * mp * bm * np_ * bn * kp * bk / (peak * 1e12) * 1e3
        t_mte1 = bytes_l0 / (bw_mte1 * 1e12) * 1e3
        t_mte2 = (bytes_a + bytes_b) / (bw_mte2 * 1e12) * 1e3
        t_fix = mp * bm * np_ * bn * elem / (bw_fix * 1e12) * 1e3
        rows.append(
            {
                "name": cand.name,
                "tile": "{}/{}/{}/{}".format(bm, bn, bk, sk),
                "map": "sm={},sn={}".format(cand.sm, cand.sn),
                "nm": mp,
                "nn": np_,
                "nk": kp,
                "a_reread": np_,
                "b_reread": mp,
                "nominal_GB": round((bytes_a + bytes_b) / 1e9, 3),
                "pred_ms": round(max(t_mac, t_mte1, t_mte2, t_fix), 3),
                "pred_mac_ms": round(t_mac, 3),
                "pred_mte1_ms": round(t_mte1, 3),
                "pred_mte2_ms": round(t_mte2, 3),
                "pred_fix_ms": round(t_fix, 3),
                "rejected": reason,
            }
        )
    best = min(
        (r for r in rows if not r["rejected"]), key=lambda r: r["pred_ms"], default=None
    )
    for row in rows:
        row["pred_ratio"] = round(row["pred_ms"] / best["pred_ms"], 3) if best else None
        row["selected"] = bool(best is not None and row is best)
    return rows


def make_tensors(m, n, k, layout_a, layout_b, device, dtype, dummy):
    torch.npu.set_device(device)
    dev = "npu:{}".format(device)
    tdtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        dtype
    ]

    def make(shape):
        if dummy:
            return torch.full(shape, 1.0, dtype=tdtype, device=dev)
        return torch.randn(shape, dtype=tdtype, device=dev)

    a_phys = make((m, k) if layout_a == "row" else (k, m))
    b_phys = make((k, n) if layout_b == "row" else (n, k))
    c = torch.empty((m, n), dtype=tdtype, device=dev)
    tag_a = tla.arch.RowMajor if layout_a == "row" else tla.arch.ColumnMajor
    tag_b = tla.arch.RowMajor if layout_b == "row" else tla.arch.ColumnMajor
    a_t = from_dlpack(a_phys, layout_tag=tag_a, origin_shape=(m, k))
    b_t = from_dlpack(b_phys, layout_tag=tag_b, origin_shape=(k, n))
    c_t = from_dlpack(c, layout_tag=tla.arch.RowMajor, origin_shape=(m, n))
    return a_phys, b_phys, c, a_t, b_t, c_t


def timed(fn, warmup, trials):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(trials):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        ts.append(start.elapsed_time(end))
    ts.sort()
    return ts[len(ts) // 2], ts


def profile_kernel(launch, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_cfg = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        export_type=[torch_npu.profiler.ExportType.Text],
        data_simplification=True,
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=1, active=1, repeat=1, skip_first=1
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out_dir)),
        experimental_config=exp_cfg,
    ) as prof:
        # schedule(skip_first=1, warmup=1, active=1, repeat=1) 共 3 个 step。
        for _ in range(3):
            launch()
            prof.step()
    torch.npu.synchronize()
    found = list(out_dir.rglob("kernel_details.csv"))
    return str(found[0]) if found else None


def _pick_round(rows, step_col, start_col, dur_col):
    """按 Step Id 分组取目标 kernel 齐全的一轮；无 Step Id 时按起始时间聚类。

    不得对全部行求和：同一形状会被跑多次，求和会得到成倍的假耗时。
    """
    if step_col:
        groups = {}
        for row in rows:
            key = (row.get(step_col) or "").strip()
            if key:
                groups.setdefault(key, []).append(row)
        if groups:
            return max(
                groups.values(),
                key=lambda g: (len(g), sum(float(r[dur_col] or 0) for r in g)),
            )
    if start_col:
        ordered = sorted(rows, key=lambda r: float(r[start_col] or 0))
        clusters, cur = [], [ordered[0]] if ordered else []
        for row in ordered[1:]:
            gap = float(row[start_col] or 0) - float(cur[-1][start_col] or 0)
            if gap > 2000.0:
                clusters.append(cur)
                cur = [row]
            else:
                cur.append(row)
        if cur:
            clusters.append(cur)
        if clusters:
            return max(
                clusters, key=lambda g: (len(g), sum(float(r[dur_col] or 0) for r in g))
            )
    return rows


def parse_kernel_csv(path, hint):
    import csv

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    cols = reader.fieldnames or []
    name_col = (
        "Name" if "Name" in cols else ("Kernel Name" if "Kernel Name" in cols else None)
    )
    step_col = next((c for c in cols if c.strip() in ("Step Id", "StepID")), None)
    start_col = next((c for c in cols if c.startswith("Start Time")), None)
    dur_col = next((c for c in cols if c.startswith("Duration")), None)
    mac_col = next((c for c in cols if "mac_ratio" in c or "cube_ratio" in c), None)
    mte1_col = next((c for c in cols if "mte1_ratio" in c), None)
    mte2_col = next((c for c in cols if "mte2_ratio" in c), None)
    fix_col = next((c for c in cols if "fix_ratio" in c or "fixpipe_ratio" in c), None)

    matched = [r for r in rows if name_col and hint in (r[name_col] or "")]
    hint_hit = bool(matched)
    hits = matched or rows
    if not dur_col:
        # 列名对不上时不猜，直接标出，避免拿错误列算时间与标定。
        return {
            "csv": path,
            "hint_hit": hint_hit,
            "matched_rows": len(hits),
            "round_rows": 0,
            "duration_us": None,
        }
    chosen = _pick_round(hits, step_col, start_col, dur_col)

    def pick_max(col):
        vals = []
        for row in chosen:
            try:
                vals.append(float(row.get(col) or ""))
            except (TypeError, ValueError):
                continue
        return max(vals) if vals else None

    return {
        "csv": path,
        "hint_hit": hint_hit,
        "matched_rows": len(hits),
        "round_rows": len(chosen),
        "duration_us": pick_max(dur_col),
        "mac_ratio": pick_max(mac_col) if mac_col else None,
        "mte1_ratio": pick_max(mte1_col) if mte1_col else None,
        "mte2_ratio": pick_max(mte2_col) if mte2_col else None,
        "fix_ratio": pick_max(fix_col) if fix_col else None,
    }


def load_spec(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("dtype", "bf16"), data.get("stages", [])


def ledger_rows(result):
    perf = (
        "| {gemm} / gemm_pipelined | {uuid} / {m},{n},{k},{dtype} | "
        "{bm}/{bn}/{bk}/{sk} × sm={sm},sn={sn} | {pred} | {pred_ratio} | {mac} | "
        "{mte2} | {ov} | {ms} | {mm} | {decision} | {ev} |"
    ).format(
        gemm=result["gemm"],
        uuid=result.get("source") or "-",
        m=result["shape"]["M"],
        n=result["shape"]["N"],
        k=result["shape"]["K"],
        dtype=result.get("dtype"),
        bm=result["tile"]["bm"],
        bn=result["tile"]["bn"],
        bk=result["tile"]["bk"],
        sk=result["tile"]["sk"],
        sm=result["tile"]["sm"],
        sn=result["tile"]["sn"],
        pred=result.get("pred_ms"),
        pred_ratio=result.get("pred_ratio"),
        mac=result.get("mac_ratio"),
        mte2=result.get("mte2_ratio"),
        ov=result.get("mac_plus_mte2"),
        ms=result.get("gemm_ms"),
        mm=result.get("torch_mm_ms"),
        decision=result.get("decision"),
        ev=result.get("kernel_csv") or result.get("output") or "-",
    )
    nominal = "| {gemm} | {bm}/{bn}/{bk}/{sk} | {nm} | {nn} | {ar} | {br} | {ideal} | {nom} | {conv} |".format(
        gemm=result["gemm"],
        bm=result["tile"]["bm"],
        bn=result["tile"]["bn"],
        bk=result["tile"]["bk"],
        sk=result["tile"]["sk"],
        nm=result["nm"],
        nn=result["nn"],
        ar=result["a_reread"],
        br=result["b_reread"],
        ideal=result.get("ideal_GB"),
        nom=result.get("nominal_GB"),
        conv=result.get("nominal_ms"),
    )
    return perf, nominal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="tools/skeleton_spec.json")
    parser.add_argument("--gemm", default=None, help="spec 中的 stage name，默认第一个")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--blocks", type=int, default=0, help="block_num；0 表示取设备核数"
    )
    parser.add_argument("--dtype", default=None, help="覆盖 spec 里的 dtype")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tile", default=None, help="BM/BN/BK/SK；留空按设备容量反推")
    parser.add_argument("--map", default=None, help="sm,sn；默认按整除规则")
    parser.add_argument(
        "--shape", default=None, help="M,N,K 覆盖（小 shape 正确性探针）"
    )
    parser.add_argument(
        "--dummy", action="store_true", help="常数填充，只用于排序，不做正确性"
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--compile-opts",
        default="",
        help="额外编译选项；留空走运行时默认目标（当前版本默认 3510，"
        "可被 CATLASS_NPU_ARCH 覆盖），不要硬编码",
    )
    parser.add_argument(
        "--kernel-hint",
        default="gemm_pipelined",
        help="profile 里匹配的 kernel 名片段，用于分轮取数",
    )
    parser.add_argument("--ledger-row", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dtype, stages = load_spec(args.spec)
    dtype = args.dtype or dtype
    if not stages:
        print("spec 里没有 stages：{}".format(args.spec))
        return 2
    if args.gemm is None:
        stage = stages[0]
    else:
        stage = next((s for s in stages if s.get("name") == args.gemm), None)
        if stage is None:
            print(
                "spec 里没有 stage：{}，可选 {}".format(
                    args.gemm, [s.get("name") for s in stages]
                )
            )
            return 2
    m, n, k = int(stage["m"]), int(stage["n"]), int(stage["k"])
    if args.shape:
        m, n, k = (int(x) for x in args.shape.split(","))
    layout_a = stage.get("layout_a", "row")
    layout_b = stage.get("layout_b", "row")
    for key, value in (("layout_a", layout_a), ("layout_b", layout_b)):
        if value not in ("row", "col"):
            print("{} 只支持 row/col，当前为 {}".format(key, value))
            return 2
    elem = ELEM_BYTES.get(dtype, 2)

    info = soc_info(args.device)
    limits = device_limits(args.device)
    limits.update(info)
    if args.tile:
        bm, bn, bk, sk = (int(x) for x in args.tile.split("/"))
    else:
        base = base_tile(limits, elem)
        if base is None:
            print("设备容量下没有可行 tile（L0C/L0A/L0B/L1 全受限），请显式传 --tile")
            return 2
        bm, bn, bk, sk = base
    tile_key = "{}/{}/{}/{}".format(bm, bn, bk, sk)
    k_raw = k
    if layout_a == "col" and not col_a_k_phase_ok(k, bk, sk):
        k = pad_k_for_col_a(k, bk)
        print(
            "WARN: layout_a=col 且 K={}（ceil(K/SK)={}，BK/SK={}）不满足子块相位，"
            "按实测会触发 LOAD2D 写 L0A 越界（err 5,131）或挂死；"
            "已自动补到 K={}".format(k_raw, (k_raw + sk - 1) // sk, bk // sk, k),
            flush=True,
        )
    k_pad = k
    cache = {} if args.no_cache else load_cache()
    key = cache_key(limits, dtype, tile_key)
    calib = cache.get(key, {})
    peak = float(calib.get("peak_tflops") or PEAK_TFLOPS)
    bw_mte1 = float(calib.get("bw_mte1_tbs") or BW_MTE1_TBS)
    bw_mte2 = float(calib.get("bw_mte2_tbs") or BW_MTE2_TBS)
    bw_fix = float(calib.get("bw_fix_tbs") or BW_FIX_TBS)

    if args.dry_run:
        rows = prescreen(
            m,
            n,
            k,
            limits,
            elem,
            peak,
            bw_mte1,
            bw_mte2,
            bw_fix,
            calibrated=bool(calib),
        )
        print("limits:", json.dumps(limits, ensure_ascii=False, sort_keys=True))
        print(
            "calibration:",
            json.dumps(
                {
                    "key": key,
                    "hit": bool(calib),
                    "peak_tflops": peak,
                    "bw_mte1_tbs": bw_mte1,
                    "bw_mte2_tbs": bw_mte2,
                    "bw_fix_tbs": bw_fix,
                },
                ensure_ascii=False,
            ),
        )
        print(
            "stage={} M={} N={} K={} K_raw={} dtype={} layout_a={} layout_b={} source={}".format(
                stage["name"],
                m,
                n,
                k,
                k_raw,
                dtype,
                layout_a,
                layout_b,
                stage.get("source", "-"),
            )
        )
        if k_pad != k_raw:
            print(
                "k_phase: violated -> padded K {} -> {} (BK={}, SK={})".format(
                    k_raw, k_pad, bk, sk
                )
            )
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
        return 0

    if args.map:
        sm, sn = (int(x) for x in args.map.split(","))
    else:
        sm, sn, _ = mapping(math.ceil(m / bm), math.ceil(n / bn))
    tiling = Tiling(bm=bm, bn=bn, bk=bk, sk=sk, sm=sm, sn=sn)

    a_phys, b_phys, c, a_t, b_t, c_t = make_tensors(
        m, n, k, layout_a, layout_b, args.device, dtype, args.dummy
    )
    blocks = args.blocks or max(1, int(limits["cube_core_num"]))
    t0 = time.time()
    if args.compile_opts:
        artifact = tla.compile(
            gemm_pipelined, a_t, b_t, c_t, tiling, options=args.compile_opts
        )
    else:
        artifact = tla.compile(gemm_pipelined, a_t, b_t, c_t, tiling)
    compile_s = time.time() - t0

    def launch():
        artifact(a_t, b_t, c_t, block_num=blocks)

    if args.compile_only:
        print(
            json.dumps(
                {"compile_s": round(compile_s, 2), "cache_key": artifact.cache_key}
            )
        )
        return 0

    launch()
    torch.npu.synchronize()

    a_logical = a_phys if layout_a == "row" else a_phys.t()
    b_logical = b_phys if layout_b == "row" else b_phys.t()
    if args.no_check or args.dummy:
        ok, exact = None, None
    else:
        ref = torch.mm(a_logical, b_logical)
        torch.npu.synchronize()
        atol, rtol = ATOL_RTOL.get(dtype, (0.2, 0.02))
        ok = bool(torch.allclose(c.float(), ref.float(), atol=atol, rtol=rtol))
        exact = float((c == ref).float().mean())
        del ref
        torch.npu.empty_cache()

    skel_ms, all_ms = timed(launch, args.warmup, args.trials)
    mm_ms, _ = timed(lambda: torch.mm(a_logical, b_logical), args.warmup, args.trials)
    torch.npu.synchronize()

    nm, nn = math.ceil(m / bm), math.ceil(n / bn)
    mp, np_, kp = nm, nn, math.ceil(k / bk)
    # 名义搬运量按 padded 尺寸算：nn·Mp·Kp·b + nm·Kp·Np·b。
    mte1_bytes = mp * np_ * kp * (bm * bk + bk * bn) * elem
    out_bytes = mp * bm * np_ * bn * elem
    nominal_bytes = (
        np_ * (mp * bm) * (kp * bk) * elem + mp * (kp * bk) * (np_ * bn) * elem
    )
    nominal_gb = nominal_bytes / 1e9
    ideal_gb = (m * k + k * n + m * n) * elem / 1e9
    pred_rows = prescreen(
        m,
        n,
        k,
        limits,
        elem,
        peak,
        bw_mte1,
        bw_mte2,
        bw_fix,
        extra=[Cand("chosen", bm, bn, bk, sk, sm, sn)],
        calibrated=bool(calib),
    )
    pred_row = (
        next((r for r in pred_rows if r["tile"] == tile_key), None) or pred_rows[0]
    )
    pred_ms = round(pred_row["pred_ms"], 3)
    result = {
        "gemm": stage["name"],
        "source": stage.get("source"),
        "dtype": dtype,
        "shape": {"M": m, "N": n, "K": k},
        "k_raw": k_raw,
        "k_pad": k_pad,
        "k_phase_padded": k_pad != k_raw,
        "layout_a": layout_a,
        "device": "npu:{}".format(args.device),
        "limits": limits,
        "calibration": {
            "key": key,
            "hit": bool(calib),
            "peak_tflops": peak,
            "bw_mte1_tbs": bw_mte1,
            "bw_mte2_tbs": bw_mte2,
            "bw_fix_tbs": bw_fix,
        },
        "tile": {"bm": bm, "bn": bn, "bk": bk, "sk": sk, "sm": sm, "sn": sn},
        "nm": nm,
        "nn": nn,
        "nk": math.ceil(k / bk),
        "a_reread": nn,
        "b_reread": nm,
        "nominal_GB": round(nominal_gb, 3),
        "ideal_GB": round(ideal_gb, 3),
        "nominal_ms": round(nominal_gb / bw_mte2, 3),
        "pred_ms": pred_ms,
        "pred_ratio": pred_row.get("pred_ratio"),
        "pred_breakdown": {
            k: pred_row.get(k)
            for k in ("pred_mac_ms", "pred_mte1_ms", "pred_mte2_ms", "pred_fix_ms")
        },
        "compile_s": round(compile_s, 2),
        "blocks": blocks,
        "allclose": ok,
        "bitwise_equal_frac": exact,
        "gemm_ms": skel_ms,
        "all_ms": all_ms,
        "torch_mm_ms": mm_ms,
        "ratio": round(skel_ms / mm_ms, 3) if mm_ms else None,
        "tflops": round(2 * m * n * k / (skel_ms * 1e-3) / 1e12, 1),
    }

    if not args.no_profile:
        out_dir = Path("evidence/skeleton") / stage["name"] / "profile"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        csv_path = profile_kernel(launch, out_dir)
        if csv_path:
            parsed = parse_kernel_csv(csv_path, args.kernel_hint)
            result.update(parsed)
            result["kernel_csv"] = csv_path
            if (
                parsed.get("mac_ratio") is not None
                and parsed.get("mte2_ratio") is not None
            ):
                result["mac_plus_mte2"] = round(
                    parsed["mac_ratio"] + parsed["mte2_ratio"], 3
                )

            # 用本次实测反推标定常数，按 (SoC, CANN, dtype, 映射档) 缓存。
            # kernel 名没匹配上时不标定：落到全部行会拿别的 kernel 反推，常数直接错。
            # 只用饱和实跑反推：mac≈1 定峰值，各带宽须该管 ≥0.95。
            # 未饱和的实跑反推会系统性偏低/偏高，宁可留占位常数只用于排序。
            entry = {"map": tile_key, "source": "measured"}
            dur_s = (parsed.get("duration_us") or 0) * 1e-6
            if parsed.get("hint_hit") and dur_s:
                if (parsed.get("mac_ratio") or 0) >= 0.95:
                    entry["peak_tflops"] = round(
                        2 * m * n * k / dur_s / parsed["mac_ratio"] / 1e12, 3
                    )
                if (parsed.get("mte1_ratio") or 0) >= 0.95:
                    entry["bw_mte1_tbs"] = round(
                        mte1_bytes / (dur_s * parsed["mte1_ratio"]) / 1e12, 3
                    )
                if (parsed.get("mte2_ratio") or 0) >= 0.95:
                    entry["bw_mte2_tbs"] = round(
                        nominal_gb * 1e9 / (dur_s * parsed["mte2_ratio"]) / 1e12, 3
                    )
                if (parsed.get("fix_ratio") or 0) >= 0.95:
                    entry["bw_fix_tbs"] = round(
                        out_bytes / (dur_s * parsed["fix_ratio"]) / 1e12, 3
                    )
            if len(entry) > 2:
                # 合并写入：同一映射档可能先由饱和跑测到峰值、后由非饱和跑补带宽，
                # 整体覆盖会把先前字段丢掉。
                merged = dict(cache.get(key, {}))
                merged.update(entry)
                cache[key] = merged
                save_cache(cache)
                result["calibration_measured"] = entry
            else:
                result["calibration_skipped"] = "各管线均未饱和（mac/各带宽 < 0.95）"

    ratio = result.get("ratio")
    mac = result.get("mac_ratio")
    mte2 = result.get("mte2_ratio")
    overlap = result.get("mac_plus_mte2")
    serial = (
        mac is not None
        and mte2 is not None
        and mac < 0.45
        and 0.80 < mte2 < 0.95
        and (overlap or 0) < 1.30
    )
    if ok is False:
        result["decision"] = "failed (allclose)"
    elif ratio is None:
        result["decision"] = "failed (no timing)"
    elif (
        mte2 is not None and mte2 >= 0.95 and result["gemm_ms"] <= result["torch_mm_ms"]
    ):
        result["decision"] = "bandwidth-bound"
    elif serial:
        result["decision"] = "failed (serial pipeline)"
    elif mac is not None and mac >= 0.99 and ratio <= 1.15:
        # 计算管已饱和时，与 torch.mm 的 5% 差距可能来自标定峰值本身偏高
        # （实测标定值比 CANN 可达算力高 4% 左右）与交接空洞，不构成串行流水。
        result["decision"] = "passed"
        result["decision_note"] = (
            "计算饱和（mac >= 0.99）且 ratio <= 1.15，按计算受限等价通过"
        )
    elif ratio <= 1.08:
        result["decision"] = "passed"
    else:
        result["decision"] = "failed (time)"
        result["decision_note"] = "需按 SKILL 写明豁免理由后再改 passed (waived)"

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    if args.ledger_row:
        perf, nominal = ledger_rows(result)
        print("\n" + perf)
        print(nominal)
    return 0 if ok is None or ok else 1


if __name__ == "__main__":
    sys.exit(main())
