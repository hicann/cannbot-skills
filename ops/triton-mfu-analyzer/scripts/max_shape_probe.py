#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""找出一个算子在本机能承载的最大 shape，并生成对应的单 case 用例文件。

为什么需要这一步
----------------
benchmark 数据集里的用例是**正确性用例**，不是性能 shape（常见 [2,3,7,16] 这种）。
在这种 shape 上测 MFU 只能测出 kernel 启动开销，数值没有意义，也测不出真实瓶颈——
数据全在 cache 里，放大后才会打回 HBM，访存墙这才显形。

判据（缺一不可，顺序即代价从低到高）
------------------------------------
1. 输入张量总字节 <= --max-bytes
2. 参考实现 class Model 能接受该 shape（FakeTensor 追踪成功且 FLOPs > 0）
   —— 参考实现拒绝的 shape 语义上就不合法，被测实现"能跑"不代表结果有意义
3. 被测实现 class ModelNew 在device上跑通
4. 单次前向 <= --max-ms（再大 PMU 采集不现实）
5. 有效带宽 = 输入字节 / 耗时 <= HBM 峰值
   —— 超过说明 kernel 根本没消费这些输入，缩放未生效

缩放规则
--------
对每个张量维度做 new = max(1, round(d * s))。这是 d 的**纯函数**，
所以"维度必须相等"的约束（Q/K 的 head_dim、K/V 的 seq_k）自动保持；
又是单调的，seq_k > seq_q 这类不等式也基本保持。
不需要为每个算子手写语义化的 shape 规则。

已知不适用的情形见 references/max-shape-search.md「attr 锁死」一节。

用法
----
    python3 max_shape_probe.py --dir <已 prepare 的算子目录> \
        [--max-bytes 10e9] [--max-ms 4000] [--out probe.json]
"""

import argparse
import fnmatch
import glob
import importlib.util as iu
import json
import os
import subprocess
import sys
import tempfile
import time

BYTES_PER = {
    "float32": 4,
    "float": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}
LADDER = [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
HBM_PEAK = 1.3e12  # 910B2C 实测 HBM 上限，用于物理合理性闸


def case_bytes(case, s=1.0):
    tot = 0
    for it in case["inputs"]:
        if it.get("type") != "tensor" or not it.get("shape"):
            continue
        p = 1
        for d in it["shape"]:
            p *= max(1, round(d * s))
        tot += p * BYTES_PER.get(it.get("dtype", "float32"), 4)
    return tot


def scaled(case, s):
    c = json.loads(json.dumps(case))
    for it in c["inputs"]:
        if it.get("type") == "tensor" and it.get("shape"):
            it["shape"] = [max(1, round(d * s)) for d in it["shape"]]
    return c


# 流程产物，绝不能当用例覆盖：verify_result*.json（精度校验）、perf_result*.json（性能结果）
EXCLUDE_PATTERNS = (
    "verify_result*.json",
    "perf_result*.json",
    "mfu_decompose.json",
    "performance.json",
    "probe.json",
    "model_new.json",
)


def is_excluded(path):
    base = os.path.basename(path)
    return any(fnmatch.fnmatch(base, pat) for pat in EXCLUDE_PATTERNS)


def is_case_file(path):
    """双重判定：先按文件名排除流程产物，再按内容确认是用例。

    只靠内容判定不够稳——产物文件的结构会变；只靠文件名黑名单也不够，
    下一个批次多出个新产物就漏了。两道一起用。
    误当用例写入会直接毁掉校验/性能结果。
    """
    if is_excluded(path):
        return False
    try:
        with open(path, encoding="utf-8-sig") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                return isinstance(obj, dict) and isinstance(obj.get("inputs"), list)
    except Exception:
        return False
    return False


def case_files(d):
    """算子目录里所有可能被 model.py 读取的用例文件。

    各算子 model.py 读用例的方式不统一：有的用 os.path.splitext(__file__)[0]+'.json'
    （即 model.json），有的硬编码 '<OpName>.json'。只写其中一个会**静默**让算子
    继续跑原始小 shape——耗时不随数据增长，是这个坑最典型的症状。
    """
    # 必须同时匹配 .json 与 .jsonl：采集脚本优先读 *_perf_cases.jsonl，
    # 只写 .json 会让它静默拿到原始小 shape（坑 7 的另一种形态）。
    cand = glob.glob(os.path.join(d, "*.json")) + glob.glob(os.path.join(d, "*.jsonl"))
    out = [p for p in cand if is_case_file(p)]
    mj = os.path.join(d, "model.json")
    if mj not in out:  # 目标文件可能还不存在，但仍要写
        out.append(mj)
    return out


def write_case(d, case):
    for fp in case_files(d):
        with open(fp, "w") as fh:
            fh.write(json.dumps(case) + "\n")


def load_modules(d):
    ms = iu.spec_from_file_location("mfu_ref", os.path.join(d, "model.py"))
    mm = iu.module_from_spec(ms)
    sys.modules["mfu_ref"] = mm
    ms.loader.exec_module(mm)
    cand = sorted(glob.glob(os.path.join(d, "model_new*.py")))
    if not cand:
        raise FileNotFoundError("目录里没有 model_new*.py")
    ns_ = iu.spec_from_file_location("mfu_impl", cand[0])
    nm = iu.module_from_spec(ns_)
    sys.modules["mfu_impl"] = nm
    ns_.loader.exec_module(nm)
    return mm, nm


def to_device(x):
    if hasattr(x, "npu"):
        return x.npu()
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(y) for y in x)
    if isinstance(x, dict):
        return {k: to_device(v) for k, v in x.items()}
    return x


def ref_accepts(d, flops_tool):
    """参考实现能否接受当前写入的 shape：FakeTensor 追踪成功且 FLOPs > 0。纯 CPU。"""
    if not flops_tool or not os.path.exists(flops_tool):
        return True, 0.0  # 没有 tracer 时跳过这一闸（但会在结果里标注）
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out = tf.name
    try:
        subprocess.run(
            [sys.executable, flops_tool, os.path.join(d, "model.py"), out],
            capture_output=True,
            text=True,
            timeout=900,
        )
        with open(out) as fp:
            fl = json.load(fp).get("flops", {})
        v = max(fl.values()) if fl else 0.0
        return v > 0, v
    except Exception:
        return False, 0.0
    finally:
        if os.path.exists(out):
            os.unlink(out)


def run_impl(d, iters=3):
    """被测实现在 device 上跑通并计时（单次 ms）。"""
    import torch

    for m in ("mfu_ref", "mfu_impl"):
        sys.modules.pop(m, None)
    mm, nm = load_modules(d)
    g = [to_device(x) for x in mm.get_input_groups()[0]]
    net = nm.ModelNew()
    if hasattr(net, "npu"):
        net = net.npu()
    with torch.no_grad():
        net(*g)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            net(*g)
        torch.npu.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1000
    del g, net
    torch.npu.empty_cache()
    return ms


def probe(d, cases_path, max_bytes, max_ms, flops_tool):
    with open(cases_path, encoding="utf-8-sig") as fp:
        orig = [json.loads(line) for line in fp if line.strip()]
    base = max(orig, key=case_bytes)
    best, s_fail, reason = None, None, "扫完整个梯度仍未受限"

    def try_scale(s):
        """返回 (ok, info)；info 里带 ms / flops / 拒绝原因。"""
        if case_bytes(base, s) > max_bytes:
            return False, f"输入 {case_bytes(base, s) / 1e9:.1f}GB 超预算"
        write_case(d, scaled(base, s))
        ok, fl = ref_accepts(d, flops_tool)
        if not ok:
            return False, "参考实现拒绝该 shape（FLOPs 追踪失败）"
        try:
            ms = run_impl(d)
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:110]}"
        if ms > max_ms:
            return False, f"单次 {ms:.0f}ms 超时限"
        bw = case_bytes(base, s) / (ms / 1000)
        if s > 1 and bw > HBM_PEAK:
            return False, f"有效带宽 {bw / 1e12:.1f}TB/s 超 HBM，kernel 未消费输入"
        return True, {"ms": ms, "flops": fl, "bw": bw}

    for s in LADDER:
        ok, info = try_scale(s)
        if not ok:
            s_fail, reason = s, f"s={s}: {info}"
            break
        best = {
            "s": s,
            "ms": round(info["ms"], 3),
            "flops": info["flops"],
            "bytes": case_bytes(base, s),
            "gbs": round(info["bw"] / 1e9, 1),
            "case": scaled(base, s),
        }

    # 梯度相邻档体积可差 8 倍，不细化会白白浪费预算
    if best and s_fail:
        lo, hi = best["s"], s_fail
        for _ in range(3):
            mid = (lo + hi) / 2
            if abs(mid - lo) < 0.05:
                break
            ok, info = try_scale(mid)
            if not ok:
                hi = mid
                continue
            best = {
                "s": round(mid, 3),
                "ms": round(info["ms"], 3),
                "flops": info["flops"],
                "bytes": case_bytes(base, mid),
                "gbs": round(info["bw"] / 1e9, 1),
                "case": scaled(base, mid),
            }
            lo = mid
        reason += f"；细化到 s={best['s']}"

    if best:  # 把选中的 shape 落到目录里，后续采集直接用
        write_case(d, best["case"])
    return {"best": best, "stop": reason}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dir", required=True, help="已 prepare_op_dir 的算子目录")
    ap.add_argument("--cases", default=None, help="原始用例 jsonl（默认取目录里的）")
    ap.add_argument("--max-bytes", type=float, default=10e9)
    ap.add_argument("--max-ms", type=float, default=4000.0)
    ap.add_argument(
        "--flops-tool",
        default=None,
        help="trace_flops.py 路径；不给则跳过参考实现门控（不推荐）",
    )
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cases = a.cases
    if not cases:
        cand = [
            p
            for p in case_files(a.dir)
            if os.path.basename(p) != "model.json" and is_case_file(p)
        ]
        if not cand:
            sys.exit("找不到用例文件（需为 JSONL，每行含 inputs 列表）")
        cases = cand[0]
    r = probe(a.dir, cases, a.max_bytes, a.max_ms, a.flops_tool)
    r["dir"] = a.dir
    r["ref_gate"] = bool(a.flops_tool)
    out = a.out or os.path.join(a.dir, "probe.json")
    with open(out, "w") as fp:
        json.dump(r, fp, ensure_ascii=False, indent=1)
    b = r["best"]
    if b:
        print(
            f"最大 shape: s={b['s']}  单次 {b['ms']}ms  输入 {b['bytes'] / 1e6:.1f}MB  "
            f"{b['gbs']}GB/s\n停因: {r['stop']}\n已写入: {out}"
        )
    else:
        print(f"未找到可用 shape。停因: {r['stop']}")
        sys.exit(3)


if __name__ == "__main__":
    main()
