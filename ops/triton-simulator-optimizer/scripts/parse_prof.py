#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
"""解析 msprof op simulator 产出，按核汇总 pipe 占比与热源码行。

供 triton-simulator-optimizer 「全 kernel 采集覆盖门禁」步骤 3/4 使用：
给定 msprof 产出的 prof_out 目录，自动定位各核的 *_instr_exe.csv + *_code_exe.csv，
按 Cube / Vector 核分别汇总 pipe 占比、top 指令、热源码行，并打印关键瓶颈信号。

用法:
    python3 parse_prof.py <prof_out_dir> [--top 8]

CSV 路径用 glob 定位（不同 msprof 版本层级可能不同，见 references/msprof-simulator.md）：
    *_instr_exe.csv  字段: instr, addr, pipe, call_count, cycles, running_time(us), detail
    *_code_exe.csv   字段: code, call_count, cycles, running_time(us)
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

# 关键瓶颈信号 pipe 名（与 references/bottleneck-diagnosis.md 对齐）
SIGNAL_PIPES = {
    "WAIT_FLAG_DEVI": "Cube 空等 Vector（FLOWCTRL）",
    "MMAD": "Cube 矩阵乘加（真 dot 算力）",
    "SCALAR": "标量降级（i32 比较/算术）",
    "MTE2": "Vector load（访存入）",
    "MTE3": "Vector store（访存出）",
    "VECTOR": "向量化计算",
    "BAR": "跨核/跨 pipe 同步",
}


def _read_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _core_kind(csv_path: str) -> str:
    """从路径推断核类型：cubecore / veccore。"""
    p = csv_path.lower()
    if "cubecore" in p:
        return "Cube"
    if "veccore" in p:
        return "Vector"
    return "Other"


def parse_instr_exe(path: str) -> dict:
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["cycles"] = _read_int(r.get("cycles"))
        r["call_count"] = _read_int(r.get("call_count"))
    rows.sort(key=lambda r: r["cycles"], reverse=True)
    total = sum(r["cycles"] for r in rows) or 1
    pipe = defaultdict(int)
    for r in rows:
        pipe[r.get("pipe", "?")] += r["cycles"]
    pipe_pct = sorted(
        ((p, c, c / total) for p, c in pipe.items()),
        key=lambda x: x[1], reverse=True,
    )
    return {"rows": rows, "total": total, "pipe_pct": pipe_pct}


def parse_code_exe(path: str) -> dict:
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["cycles"] = _read_int(r.get("cycles"))
    rows.sort(key=lambda r: r["cycles"], reverse=True)
    total = sum(r["cycles"] for r in rows) or 1
    return {"rows": rows, "total": total}


def fmt_pct(x: float) -> str:
    return f"{x*100:5.1f}%"


def main() -> int:
    p = argparse.ArgumentParser(description="解析 msprof simulator 产出 → pipe 占比 + 热源码行")
    p.add_argument("prof_dir", help="msprof 产出的 prof_out 目录")
    p.add_argument("--top", type=int, default=8, help="每核展示的 top 指令/行数（默认 8）")
    args = p.parse_args()

    instr_csvs = sorted(glob.glob(os.path.join(args.prof_dir, "**", "*_instr_exe.csv"), recursive=True))
    code_csvs = sorted(glob.glob(os.path.join(args.prof_dir, "**", "*_code_exe.csv"), recursive=True))

    if not instr_csvs:
        print(f"[ERROR] 未找到 *_instr_exe.csv 于 {args.prof_dir}", file=sys.stderr)
        print("  以两张 CSV 是否存在且非空为采集成功判据（见 references/msprof-simulator.md）",
              file=sys.stderr)
        return 2

    # 按 (core_kind, 核目录) 分组
    cores = {}
    for c in instr_csvs:
        key = (_core_kind(c), os.path.dirname(c))
        cores.setdefault(key, {"instr": None, "code": None})
        cores[key]["instr"] = c
    for c in code_csvs:
        key = (_core_kind(c), os.path.dirname(c))
        cores.setdefault(key, {"instr": None, "code": None})
        cores[key]["code"] = c

    print(f"# 覆盖表：{len(cores)} 个核目录，prof_dir={args.prof_dir}")
    print(f"| core | 核目录 | instr_csv | code_csv | total_cycles | top_pipe(占比) |")
    print("|------|--------|-----------|----------|---------------|---------------|")
    for (kind, cdir), paths in sorted(cores.items()):
        has_i = "✓" if paths["instr"] else "✗"
        has_c = "✓" if paths["code"] else "✗"
        if paths["instr"]:
            d = parse_instr_exe(paths["instr"])
            top_pipes = ", ".join(f"{p}({fmt_pct(pc)})" for p, _, pc in d["pipe_pct"][:3])
            total_cyc = d["total"]
        else:
            top_pipes, total_cyc = "(无 instr_exe)", 0
        print(f"| {kind} | {os.path.basename(cdir)} | {has_i} | {has_c} | {total_cyc} | {top_pipes} |")

    # 每核详细
    for (kind, cdir), paths in sorted(cores.items()):
        print(f"\n===== {kind}  {os.path.basename(cdir)} =====")
        if not paths["instr"]:
            print("  (无 *_instr_exe.csv) 跳过")
            continue
        d = parse_instr_exe(paths["instr"])
        print(f"  total cycles: {d['total']}")
        print("  pipe 占比（按 cycles 降序）:")
        for pname, cyc, pct in d["pipe_pct"]:
            tag = f"  ← {SIGNAL_PIPES[pname]}" if pname in SIGNAL_PIPES else ""
            print(f"    {pname:18s} {cyc:>10d}  {fmt_pct(pct)}{tag}")
        # 关键信号速报
        pipe_map = {p: pct for p, _, pct in d["pipe_pct"]}
        signals = []
        if pipe_map.get("WAIT_FLAG_DEVI", 0) > 0.5 and pipe_map.get("MMAD", 0) < 0.05:
            signals.append("⚠️ Cube 空等 Vector（WAIT_FLAG_DEVI>50% 且 MMAD<5%）→ 修复方向: latency-optimizer 优化点 19 / 21")
        if pipe_map.get("MMAD", 0) > 0.5:
            signals.append("⚠️ MMAD>50% → 计算bound（真·硬件极限判据，无现成优化点：增大 tile / bf16 化，均不可行回 4.6）")
        if pipe_map.get("SCALAR", 0) > 0.3:
            signals.append("⚠️ SCALAR>30% → 标量降级 → 修复方向: latency-optimizer 优化点 6 / 5 / 17")
        if signals:
            print("  信号:")
            for s in signals:
                print(f"    {s}")
        print(f"  top {args.top} 指令（按 cycles 降序）:")
        for r in d["rows"][:args.top]:
            print(f"    {r.get('instr','?'):20s} pipe={r.get('pipe','?'):14s} "
                  f"calls={r['call_count']:>6d} cyc={r['cycles']:>8d}")
        if paths["code"]:
            cd = parse_code_exe(paths["code"])
            print(f"  top {args.top} 热源码行（code_exe，按 cycles 降序）:")
            for r in cd["rows"][:args.top]:
                code = (r.get("code", "") or "").strip().replace("\n", " ")[:80]
                print(f"    cyc={r['cycles']:>8d}  {code}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
