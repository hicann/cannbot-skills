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
"""在 (case, 倍率) 候选空间里搜索极限 MFU：墙钟初筛 -> msprof 精测。

为什么是二维
------------
**极限 MFU 不等于最大 shape 下的 MFU。**

- **倍率维**：MFU 对 shape 不单调。小 shape 时数据在 cache 里，放大后被打回 HBM，
  `aic_mte2` 变成墙，MFU 反而下降。实测一批算子里约 2/3 放大后 MFU 降低，
  峰值常落在中间档而非最大档。
- **case 维**：同一算子的不同 case 可能走**不同的代码路径**。
  实测某 OutlookAttention 实现：case A 只执行 `_linear_kernel`（MFU 18%），
  case B 执行完整的 `_linear/_pool/_softmax/_attn/_fold` 五个 kernel（MFU 2%）。
  只固定"FLOPs 最大的 case"会系统性漏掉上界。

两级流程与参数依据
------------------
全 case 逐点采集太贵（约 50 case x 89 算子），所以分两级：

1. **初筛（墙钟，单进程遍历）**：候选池 = 按 FLOPs 降序的前 K 个**原始 case**
   （默认 K=8）+ 它们通过合法性闸的**放大变体**。原始 case 无条件入池。
2. **精测（msprof）**：只对初筛排名前 M（默认 3）的点采集。
   **最终 MFU 一律取自精测的 kernel 时间**，墙钟只用于排序。

K=8 的依据（5 个算子全 50 case 实测标定）：最高 MFU 的 case 在 FLOPs 降序里
分别排第 4/2/5/3/2 位 —— **top-3 有 2/5 漏掉最高点，top-5 全部命中**，
取 8 是在 top-5 基础上留冗余，吸收墙钟排序可能的翻转。
另外三种 dtype（fp16/bf16/fp32）都出现过冠军，**不能只按 dtype 挑**。

墙钟开销（实测）：每次前向固定 30~70us（均值 43us）的 host 侧 dispatch+同步成本，
**与 kernel 时长无关**。所以低估幅度 = 开销/墙钟：
kernel ~70us 时低估 49%，~140us 时 17%，~860us 时 3.4%，>2ms 时 <2%。
危险区在 kernel < 200us，而极限 MFU 几乎总出现在大 case 上，
墙钟失真最严重的恰好是注定不会夺冠的小 case —— 这是初筛可用的前提。

用法
----
    python3 mfu_grid_search.py --dir <已 prepare 的算子目录> \
        --flops-tool <trace_flops.py> --runner <run_once.py> \
        [--cases-top-k 3] [--scales 1,2,4] [--peak 353.89e12]
"""

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from max_shape_probe import case_bytes, case_files, scaled

HBM_PEAK = 1.3e12
RATIOS = [
    "aic_mac_ratio",
    "aic_scalar_ratio",
    "aic_mte1_ratio",
    "aic_mte2_ratio",
    "aic_fixpipe_ratio",
    "aiv_vec_ratio",
    "aiv_scalar_ratio",
    "aiv_mte2_ratio",
    "aiv_mte3_ratio",
]
SHOW = {"aic_mac_ratio": "cube"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def write_case(d, case):
    for fp in case_files(d):
        with open(fp, "w") as fh:
            fh.write(json.dumps(case) + "\n")


def _limit_mem(nbytes):
    """给子进程设地址空间硬上限，超限抛 MemoryError 而不是把整机内存吃光。"""

    def _f():
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (int(nbytes), int(nbytes)))
        except Exception:
            pass

    return _f


def trace_flops(d, tool, mem_cap=8 * 1024**3):
    """对参考实现 class Model 做 aten 级 FLOPs 计数。

    **FakeTensor 失败即跳过，不回退真实张量。**

        回退路径会实例化 [B,H,Sq,Sk] 的稠密中间张量：输入几十 MB 时中间量可达数十 GB
        （Sq=Sk=8192、40 heads → score 矩阵 2.7e9 元素 = 10.7 GB），
        而按输入规模设的护栏挡不住 —— 护栏查的是输入，爆的是中间量。
        真实张量算出的 FLOPs 与 fake 一致，回退没有收益，只有爆内存的风险。

        两道强制：环境变量 MFU_FLOPS_NO_REAL=1 供 tracer 自行遵守；
        RLIMIT_AS 硬上限作为兜底（对不认该变量的 tracer 同样有效）。
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out = tf.name
    try:
        subprocess.run(
            [sys.executable, tool, os.path.join(d, "model.py"), out],
            capture_output=True,
            text=True,
            timeout=900,
            env=dict(os.environ, MFU_FLOPS_NO_REAL="1"),
            preexec_fn=_limit_mem(mem_cap),
        )
        with open(out) as fp:
            fl = json.load(fp).get("flops", {})
        return max(fl.values()) if fl else 0.0
    except Exception:
        return 0.0
    finally:
        if os.path.exists(out):
            os.unlink(out)


def period(names, n_iters):
    """检出前向的重复周期，用于剔除 setup 阶段并正确计入混合实现里的框架算子。"""
    n = len(names)
    for m in range(n_iters, 2, -1):
        for p in range(n // m, 0, -1):
            tail = names[n - m * p:]
            if all(tail[i * p:(i + 1) * p] == tail[:p] for i in range(m)):
                return p, m
    return None, None


def collect(prof_dir, iters):
    """从 msprof 产物解析 kernel 时间与流水占比。返回 None 表示采集不可用。"""
    rows = []
    for fp in glob.glob(
        os.path.join(prof_dir, "**", "op_summary_*.csv"), recursive=True
    ):
        with open(fp) as csv_fp:
            rows += list(csv.DictReader(csv_fp))
    if not rows:
        return None
    rows.sort(key=lambda r: _f(r.get("Task Start Time(us)", 0)))
    names = [r["Op Name"] for r in rows]
    p, _m = period(names, iters + 2)
    ker = (
        rows[len(rows) - iters * p:]
        if p
        else [r for r in rows if not r["Op Name"].startswith("aclnn")]
    )
    if not ker:
        return None
    tot = sum(_f(r["Task Duration(us)"]) for r in ker)
    if tot <= 0:
        return None
    agg = {k: 0.0 for k in RATIOS}
    act = 0.0
    cube_cap = 0.0  # cube 在各 kernel 上能产出的 FLOPs 上限之和
    per = defaultdict(float)
    for r in ker:
        du = _f(r["Task Duration(us)"])
        w = du / tot
        cube_cap += du / iters * 1e-6 * _f(r.get("aic_mac_ratio", 0))
        for k in RATIOS:
            agg[k] += _f(r.get(k, 0)) * w
        ac, av = _f(r.get("aicore_time(us)", 0)), _f(r.get("aiv_time(us)", 0))
        act += min(max(ac, av), _f(r["Task Duration(us)"]))
        per[r["Op Name"]] += _f(r["Task Duration(us)"])
    anomaly = [k for k in RATIOS if any(_f(r.get(k, 0)) > 1.05 for r in ker)]
    live = {k: v for k, v in agg.items() if v > 0}
    return {
        "kernel_us": tot / iters,
        "period_ok": bool(p),
        "active": round(act / tot, 3),
        "cube_secs": cube_cap,
        "ratios": live,
        "anomaly": anomaly,
        "kernels": {
            k: round(v / iters, 1)
            for k, v in sorted(per.items(), key=lambda x: -x[1])[:6]
        },
    }


PEAK_DTYPE = 319.9e12  # 实测 fp16/bf16 峰值（fp32 为 84.56e12）
HBM_BW = 1300e9  # 实测 HBM 上限


def roofline_pct(flops, byts, peak):
    """理论极限 MFU(%)：min(实测dtype峰值, HBM×算术强度) / 标称峰值。"""
    if not byts:
        return round(PEAK_DTYPE / peak * 100, 3)
    ai = flops / byts
    return round(min(PEAK_DTYPE, HBM_BW * ai) / peak * 100, 3)


def shape_tag(case, s=1.0):
    """极限/基线 shape 的可读写法，如 b2_q4096_nq32_nk8_d128。"""
    parts = []
    for it in case["inputs"]:
        if it.get("type") == "tensor" and it.get("shape"):
            sh = [max(1, round(d * s)) for d in it["shape"]]
            parts.append(f"{it['name']}[{'x'.join(map(str, sh))}]")
        elif it.get("type") != "tensor" and isinstance(it.get("value"), (int, float)):
            parts.append(f"{it['name']}={it['value']}")
    dt = "/".join(
        sorted(
            {
                it.get("dtype")
                for it in case["inputs"]
                if it.get("type") == "tensor" and it.get("dtype")
            }
        )
    )
    return "_".join(parts) + (f"_{dt}" if dt else "")


def write_md(path, op, base, best, peak):
    """产出 mfu.md：基线 / 极限 / 分子分母 / roofline / bound / profiling 细节。"""
    order = [
        "cube",
        "aic_scalar",
        "aic_mte1",
        "aic_mte2",
        "aic_fixpipe",
        "aiv_vec",
        "aiv_scalar",
        "aiv_mte2",
        "aiv_mte3",
    ]
    r = best.get("ratios", {})  # 键已是简写（grid 里做过映射）
    nine = ", ".join(f"{k} {r[k]:.3f}" for k in order if k in r)
    det = nine
    if best.get("active") is not None:
        det += f"；核活跃 {best['active'] * 100:.0f}%"
    if best.get("kernels"):
        det += f"；执行 kernel: {', '.join(list(best['kernels'])[:4])}"
    for w in best.get("suspect", []):
        det += "；⚠" + w
    cols = [
        "基线shape",
        "基线MFU",
        "极限shape",
        "极限kernel耗时（ms）",
        "极限分子Tflops",
        "分母Tflops",
        "实测极限MFU",
        "bound",
        "profiling细节",
    ]
    vals = [
        base["shape"] if base else "—",
        f"{base['mfu']:.4g}" if base else "—",
        best["shape"],
        f"{best['kernel_us'] / 1000:.4f}",
        f"{best['flops'] / (best['kernel_us'] * 1e-6) / 1e12:.4g}",
        f"{peak / 1e12:.2f}",
        f"{best['mfu']:.4g}",
        best["bound"],
        det,
    ]
    with open(path, "w") as f:
        f.write(f"# {op} MFU\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        f.write("| " + " | ".join(str(v).replace("|", "/") for v in vals) + " |\n")
    return dict(zip(cols, vals))


def verdict(live, anomaly, sat):
    if anomaly:
        return (
            "PMU 计数不自洽，不做 bound 判定",
            "计数异常（%s 比值>1）"
            % ",".join(a.replace("_ratio", "") for a in anomaly[:3]),
        )
    if not live:
        return "无流水数据", "N/A"
    top = max(live, key=live.get)
    name = SHOW.get(top, top.replace("_ratio", ""))
    if live[top] >= sat:
        return "达到 bound", f"{name} {live[top]:.3f}"
    return "未达 bound", f"未达；最高 {name} {live[top]:.3f}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dir", required=True, help="已 prepare_op_dir 的算子目录")
    ap.add_argument("--cases", default=None, help="原始用例 jsonl（默认取目录里的）")
    ap.add_argument("--flops-tool", required=True, help="trace_flops.py 路径")
    ap.add_argument(
        "--runner",
        required=True,
        help="执行脚本，契约：runner <算子目录> <iters> [--wall]；"
        "普通模式跑完打印 [RUN] ok（供 msprof 包裹）；"
        "带 --wall 时额外打印 [WALL] <单次毫秒>（供初筛，不开 msprof）",
    )
    ap.add_argument(
        "--screen-top-k",
        type=int,
        default=8,
        help="初筛池：按 FLOPs 降序取前 K 个原始 case（标定建议 >=5，默认 8）",
    )
    ap.add_argument(
        "--measure-top-k",
        type=int,
        default=3,
        help="精测：对初筛排名前 M 的点用 msprof 采集",
    )
    ap.add_argument("--scales", default="1", help="逗号分隔的倍率，务必包含 1")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--peak", type=float, default=353.89e12, help="芯片峰值算力")
    ap.add_argument("--sat", type=float, default=0.70, help="判定 bound 的流水占比阈值")
    ap.add_argument("--device", default="0")
    ap.add_argument(
        "--flops-mem-cap-gb",
        type=float,
        default=8.0,
        help="FLOPs 追踪子进程的地址空间硬上限（GB）。兜底用："
        "正常路径下 FakeTensor 失败即跳过，不会分配真实张量",
    )
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cases_path = a.cases
    if not cases_path:
        cand = [p for p in case_files(a.dir) if os.path.basename(p) != "model.json"]
        if not cand:
            sys.exit("找不到用例文件")
        cases_path = cand[0]
    with open(cases_path, encoding="utf-8-sig") as fp:
        cs = [json.loads(line) for line in fp if line.strip()]

    # 先按原始 shape 的 FLOPs 排序挑候选 case
    order = []
    for i, c in enumerate(cs):
        write_case(a.dir, c)
        order.append(
            (trace_flops(a.dir, a.flops_tool, a.flops_mem_cap_gb * 1024**3), i)
        )
    order.sort(reverse=True)
    top = [i for _, i in order[: a.screen_top_k]]
    scales = [float(x) for x in a.scales.split(",")]
    if 1.0 not in scales:
        scales.insert(0, 1.0)  # 原始 case 必须在池里

    # ---- 初筛：墙钟遍历候选池 ----
    screen = []
    for ci in top:
        for s in scales:
            by = case_bytes(cs[ci], s)
            write_case(a.dir, scaled(cs[ci], s))
            fl = trace_flops(a.dir, a.flops_tool, a.flops_mem_cap_gb * 1024**3)
            if fl <= 0:
                screen.append(
                    {
                        "case": ci,
                        "s": s,
                        "skip": "参考实现拒绝该 shape / FLOPs 追踪为 0 或超内存上限",
                    }
                )
                continue
            r = subprocess.run(
                [sys.executable, a.runner, a.dir, "3", "--wall"],
                capture_output=True,
                text=True,
                timeout=3600,
            )
            # CANN 的 [Warning] 不带换行，会和输出粘成一行，
            # 因此必须全文搜索而不是 startswith —— 用后者会把每个点都误判成"没返回"。
            m = re.search(r"\[WALL\]\s+([0-9.eE+-]+)", r.stdout + r.stderr)
            if not m:
                tail = (r.stderr or r.stdout).strip().splitlines()
                why = tail[-1][:100] if tail else "runner 无输出"
                screen.append(
                    {"case": ci, "s": s, "skip": f"runner 未返回 [WALL]：{why}"}
                )
                continue
            ms = float(m.group(1))
            if s > 1 and by / (ms / 1000) > HBM_PEAK:
                screen.append(
                    {
                        "case": ci,
                        "s": s,
                        "skip": f"有效带宽 {by / (ms / 1000) / 1e12:.1f}TB/s 超 HBM，缩放未生效",
                    }
                )
                continue
            screen.append(
                {
                    "case": ci,
                    "s": s,
                    "flops": fl,
                    "bytes": by,
                    "wall_ms": round(ms, 4),
                    "mfu_est": round(fl / (ms / 1000) / a.peak * 100, 4),
                }
            )
    live = [x for x in screen if "skip" not in x]
    live.sort(key=lambda x: -x["mfu_est"])
    picked = {(g["case"], g["s"]) for g in live[: a.measure_top_k]}
    if not live:  # 全被跳过时把原因摊开，便于定位
        from collections import Counter

        why = Counter(x["skip"].split("：")[0] for x in screen)
        print(
            "初筛无一入池，跳过原因分布："
            + ", ".join(f"{k}×{v}" for k, v in why.most_common())
        )

    # ---- 精测：只对入围点跑 msprof ----
    grid = []
    for ci in top:
        for s in scales:
            if (ci, s) not in picked:
                continue
            by = case_bytes(cs[ci], s)
            case = scaled(cs[ci], s)
            write_case(a.dir, case)
            fl = trace_flops(a.dir, a.flops_tool, a.flops_mem_cap_gb * 1024**3)
            if fl <= 0:
                grid.append(
                    {"case": ci, "s": s, "skip": "参考实现拒绝该 shape / FLOPs=0"}
                )
                continue
            pd = tempfile.mkdtemp(prefix="prof_")
            env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=a.device)
            subprocess.run(
                [
                    "msprof",
                    f"--output={pd}",
                    "--ai-core=on",
                    "--task-time=on",
                    "--aic-metrics=PipeUtilization",
                    "--ascendcl=on",
                    sys.executable,
                    a.runner,
                    a.dir,
                    str(a.iters),
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=3600,
            )
            m = collect(pd, a.iters)
            if not m:
                grid.append({"case": ci, "s": s, "skip": "msprof 无有效 kernel 数据"})
                continue
            bw = by / (m["kernel_us"] * 1e-6)
            if s > 1 and bw > HBM_PEAK:
                grid.append(
                    {"case": ci, "s": s, "skip": f"有效带宽 {bw / 1e12:.1f}TB/s 超 HBM"}
                )
                continue
            v, b = verdict(m["ratios"], m["anomaly"], a.sat)
            # ---- cube 一致性：分子不可能超过 cube 在这段时间里能产出的 FLOPs ----
            cap = m["cube_secs"] * a.peak
            over = fl / cap if cap > 0 else float("inf")
            suspect = []
            if fl > cap * 1.05:
                suspect.append(
                    f"分子超 cube 上限 {over:.2f}x —— cube 流水占比 "
                    f"{m['ratios'].get('aic_mac_ratio', 0):.3f} 决定了这段时间内 cube 最多只能"
                    f"产出 {cap:.3g} FLOPs，而分子是 {fl:.3g}。成因：参考实现用稠密表达了"
                    f"稀疏/滑窗/causal/topk 语义（算完整矩阵再 mask），"
                    f"triton kernel 利用了稀疏结构、实际没做这么多。"
                    f"【MFU 数值仍按参考实现分子口径给出，未做修正】"
                )
            if m["ratios"].get("aic_mac_ratio", 0) <= 0.001:
                suspect.append(
                    "cube 未参与（纯 vector 实现），cube 判据不适用，MFU 需另行核实"
                )
            grid.append(
                {
                    "case": ci,
                    "s": s,
                    "flops": fl,
                    "bytes": by,
                    "shape": shape_tag(cs[ci], s),
                    "roof": roofline_pct(fl, by, a.peak),
                    "cube_cap": cap,
                    "flops_over_cube": round(over, 2),
                    "suspect": suspect,
                    "kernel_us": round(m["kernel_us"], 2),
                    "mfu": round(fl / (m["kernel_us"] * 1e-6) / a.peak * 100, 4),
                    "gbs": round(bw / 1e9, 1),
                    "active": m["active"],
                    "verdict": v,
                    "bound": b,
                    "kernels": m["kernels"],
                    "ratios": {
                        SHOW.get(k, k.replace("_ratio", "")): round(x, 3)
                        for k, x in m["ratios"].items()
                    },
                }
            )

    ok = [g for g in grid if "skip" not in g]
    # 寻优目标：MFU 最大化
    best = max(ok, key=lambda g: g["mfu"]) if ok else None
    # 基线 = 数据集原始 shape（s=1）里 FLOPs 最大的 case —— 固定参照点，
    # 不随寻优结果变化，这样"基线 -> 极限"的增益才有意义。
    s1 = [g for g in ok if g["s"] == 1.0]
    base = max(s1, key=lambda g: g["flops"], default=None)
    # 分子口径：统一用参考实现追踪（FlopCounterMode 对 class Model 做 aten 级计数）。
    # 不再做硬件计数核算 —— MFU 的分子取语义 FLOPs，硬件计数含 tiling 补齐与
    # 被 mask 丢弃的无效算力，属"算力利用率"而非 MFU。

    res = {
        "dir": a.dir,
        "screen": screen,
        "screen_live": len(live),
        "grid": grid,
        "best": best,
        "baseline": base,
    }
    if best:
        md = os.path.join(os.path.dirname(a.out or os.path.join(a.dir, "x")), "mfu.md")
        res["mfu_md"] = write_md(
            md, os.path.basename(a.dir.rstrip("/")), base, best, a.peak
        )
        print(f"  已写 {md}")
    out = a.out or os.path.join(a.dir, "mfu_grid.json")
    with open(out, "w") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=1)
    if best:
        if best.get("suspect"):
            print("⚠ 该点未通过一致性检查：" + "；".join(best["suspect"]))
        print(
            f"极限 MFU {best['mfu']:.4f}%  @ case#{best['case']} s={best['s']}  "
            f"kernel {best['kernel_us']:.1f}us\n  bound: {best['bound']}\n"
            f"  执行的 kernel: {list(best['kernels'])}\n"
            f"  初筛 {len(live)}/{len(screen)} 点入池 -> 精测 {len(grid)} 点\n  明细: {out}"
        )
    else:
        print(f"网格内无有效点，明细: {out}")


if __name__ == "__main__":
    main()
