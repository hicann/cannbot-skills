#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""提取静态图（V1 task sink）算子实际下发地址（dfx args 表 + D2H 回读）。

用法: python3 static_addr_extract.py <日志文件> [算子名过滤]
前提: 日志需开启 GE_DAVINCI_MODEL_PROFILING=2 + ASCEND_GLOBAL_LOG_LEVEL=1（未开启时脚本会提示重跑）
原理（见 references/addr-reference.md 路径二）:
  ① [IMAS] GetInput/GetOutputDataAddrs 行给出算子 io 明细（name[input/output[N]]/size/memaddr=logical_addr）
  ② 两级 [Args][Init]: args_io_addrs_updater 补 id/refreshable，
     GenModelArgsRefreshInfosForTask 给算子段基址与 io_index→args_offset
  ③ 槽位公式: host table index = (算子段基址 - model_args_table_addr)/8 + io_index
  ④ Print model args host table 给各槽多轮实值；Print different args 为 D2H 回读差异
  ⑤ ConstructZeroCopyIoActiveBaseAddrs 的 user_addr 为零拷贝实际地址（经 active_mem_base[id] 匹配关联）
  ⑥ F 区实际申请（need N/M）判定 logical_addr 已分配/未分配（零拷贝占位）
输出: 按 skill 展示规则——实际下发地址 + logical_addr（含 refreshable）+ user_addr（先 logical 后 user）
"""

import logging
import re
import signal
import sys
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def hsz(b):
    """字节数转人类可读字符串。"""
    if b >= 1048576:
        return f"{b / 1048576:.0f}M"
    if b >= 1024:
        return f"{b / 1024:.0f}K"
    return f"{b}B"


@dataclass
class Regions:
    """内存区范围（F 区实际申请 / W 区常量区 / 申请量）。"""

    f_start: int = 0
    f_end: int = 0
    actual_sz: int = 0
    total_sz: int = 0
    w_start: int = 0
    w_end: int = 0

    def alloc_state(self, address):
        """判定地址落在已分配区（F 区 / W 区常量）还是未分配（零拷贝占位）。"""
        if self.f_start and self.f_start <= address < self.f_end:
            return "已分配（F 区）"
        if self.w_start and self.w_start <= address < self.w_end:
            return "已分配（W 区常量）"
        return None


@dataclass
class ParsedLog:
    """日志解析结果（算子 io / id / 段 / 槽值 / user_addr / 区范围 / 表基址）。"""

    table_base: int = 0
    regions: Regions = field(default_factory=Regions)
    io_lines: dict = field(default_factory=dict)
    id_info: dict = field(default_factory=dict)
    segs: dict = field(default_factory=dict)
    slot_vals: dict = field(default_factory=dict)
    user_addrs: dict = field(default_factory=dict)
    id2user: dict = field(default_factory=dict)


def parse_regions(text):
    """解析 F 区实际申请范围与 W 区（常量区）范围。"""
    rg = Regions()
    m = re.search(
        r"need (\d+)/(\d+) for feature-map without zero-copyable memory", text
    )
    rg.actual_sz, rg.total_sz = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    offs = []
    for off, a in re.findall(
        r"\[IMAS\]\S* ?type\[F\] name\[\S+?\] (?:input|output)\[\d+\] offset\[(\d+)\] size\[\d+\] "
        r"memaddr\[(0x[0-9a-f]+)\]",
        text,
    ):
        offs.append(int(a, 16) - int(off))
    if offs:
        rg.f_start = min(offs)
        rg.f_end = rg.f_start + rg.actual_sz
    m = re.search(
        r"InitWeightMem \w+ MallocMemory type\[W\] memaddr\[(0x[0-9a-f]+)\] mem_size\[(\d+)\]",
        text,
    )
    if m:
        rg.w_start = int(m.group(1), 16)
        rg.w_end = rg.w_start + int(m.group(2))
    return rg


def parse_ops(text):
    """解析算子 io 明细、id/refreshable、段基址（两级 [Args][Init] + [IMAS] 行）。"""
    io_lines = {}
    for m in re.finditer(
        r"\[IMAS\][^[]*?type\[(\w)\] name\[(\S+?)\] (input|output)\[(\d+)\](?: offset\[\d+\])? "
        r"size\[(\d+)\] memaddr\[(0x[0-9a-f]+)\]",
        text,
    ):
        key = (m.group(2), m.group(3), int(m.group(4)))
        if key not in io_lines:
            io_lines[key] = {
                "type": m.group(1),
                "size": int(m.group(5)),
                "logical": int(m.group(6), 16),
            }
    id_info = {}
    for m in re.finditer(
        r"\[Args\]\[Init\] op_name:(\S+), op_type:\S+, logical_addr\[\d+\]:(0x[0-9a-f]+), id:(\d+), "
        r"offset:(0x[0-9a-f]+), refreshable:(\d)",
        text,
    ):
        id_info[(m.group(1), int(m.group(2), 16))] = {
            "id": int(m.group(3)),
            "refr": m.group(5),
        }
    segs = {}
    for m in re.finditer(
        r"GenModelArgsRefreshInfosForTask:\[Args\]\[Init\] op_name:(\S+), op_type:\S+, pls:\d+, "
        r".*?pls dev addr:(0x[0-9a-f]+), task args refresh info:\[id:\d+, offset:\S+, "
        r"io_index:(\d+), args_offset:(0x[0-9a-f]+)",
        text,
    ):
        op, seg, iidx = m.group(1), int(m.group(2), 16), int(m.group(3))
        d = segs.setdefault(op, {"seg": seg, "io": {}})
        d["seg"] = min(d["seg"], seg)
        d["io"][iidx] = int(m.group(4), 16)
    return io_lines, id_info, segs


def parse_slot_vals(text):
    """解析槽值多轮实值（Print model args host table）。"""
    slot_vals = {}
    for m in re.finditer(
        r"Print model args host table, model args index is:(\d+), "
        r"model args host tensor data addr is:(0x[0-9a-f]+)",
        text,
    ):
        slot_vals.setdefault(int(m.group(1)), []).append(m.group(2))
    return slot_vals


def match_user_addr(user_addrs, bval):
    """在 user_addrs 中查找与 active mem base 值匹配的 (kind, index) 集合。"""
    matched = set()
    for kind, idx_map in user_addrs.items():
        for idx, vals in idx_map.items():
            if bval in vals:
                matched.add((kind, idx))
    return matched


def link_id2user(user_addrs, text):
    """解析 active mem base 表，把 id 关联到 user_addr 的 (kind, index)。"""
    id2user = {}
    for m in re.finditer(
        r"Print Kernel Launch Args, host active mem base Index is:(\d+), "
        r"active mem base addr is:(0x[0-9a-f]+)",
        text,
    ):
        bid, bval = int(m.group(1)), m.group(2)
        for pair in match_user_addr(user_addrs, bval):
            id2user.setdefault(bid, set()).add(pair)
    return id2user


def parse_user_addrs(text):
    """解析零拷贝 user_addr（Input/Output，多轮保序去重）。"""
    user_addrs = {"Input": {}, "Output": {}}
    for m in re.finditer(
        r"ConstructZeroCopyIoActiveBaseAddrs:\[(Input|Output)\] index:(\d+), "
        r"user_addr:(0x[0-9a-f]+)",
        text,
    ):
        kind_map = user_addrs.setdefault(m.group(1), {})
        kind_map.setdefault(int(m.group(2)), []).append(m.group(3))
    return user_addrs


def parse_all(text):
    """汇总全部解析步骤，返回 ParsedLog。"""
    m = re.search(r"model_args_table_addr:(0x[0-9a-f]+)", text)
    p = ParsedLog(table_base=int(m.group(1), 16) if m else 0)
    p.regions = parse_regions(text)
    p.io_lines, p.id_info, p.segs = parse_ops(text)
    p.slot_vals = parse_slot_vals(text)
    p.user_addrs = parse_user_addrs(text)
    p.id2user = link_id2user(p.user_addrs, text)
    return p


def print_op(op, p):
    """输出单个算子的全部 io 地址明细（实际下发 + logical_addr + user_addr）。"""
    seg = p.segs.get(op, {}).get("seg", p.table_base)
    base_idx = (seg - p.table_base) // 8
    ios = sorted(
        [(k, v) for k, v in p.io_lines.items() if k[0] == op],
        key=lambda kv: (kv[0][1] != "input", kv[0][2]),
    )
    n_in = sum(1 for k, _ in ios if k[1] == "input")
    logger.info(
        f"\n[{op}]  ({n_in} in / {len(ios) - n_in} out，槽位 {base_idx}~{base_idx + len(ios) - 1})"
    )
    for io_i, ((_, kind, idx), v) in enumerate(ios):
        e = p.id_info.get((op, v["logical"]), {})
        slot = base_idx + io_i
        vals = list(dict.fromkeys(p.slot_vals.get(slot, [])))
        tag = f"{kind}_{idx}"
        logger.info(
            f"    {tag:9s} {' / '.join(vals) if vals else '（无槽值记录）'}  ({hsz(v['size'])})"
        )
        refr = e.get("refr", "?")
        st = p.regions.alloc_state(v["logical"])
        if refr == "0":
            logger.info(
                f"              ← logical_addr 0x{v['logical']:x}（{st or '常量'}，refreshable=0 直传，"
                f"id:{e.get('id', '?')}）"
            )
        else:
            logger.info(
                f"              ← logical_addr 0x{v['logical']:x}"
                f"（{st or '未分配，零拷贝占位'}，id:{e.get('id', '?')}，refreshable={refr}）"
            )
        for ukind, uidx in sorted(p.id2user.get(e.get("id", -1), set())):
            uv = list(dict.fromkeys(p.user_addrs.get(ukind, {}).get(uidx, [])))
            logger.info(
                f"              ← user_addr [{ukind}] index:{uidx} = {' / '.join(uv)}"
                f"（用户 buffer，即实际下发地址）"
            )


def main():
    """解析 dfx 日志并输出算子下发地址，返回退出码。"""
    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    if len(sys.argv) < 2:
        logger.error(f"用法: python3 {sys.argv[0]} <日志文件> [算子名过滤]")
        return 1

    log = sys.argv[1]
    op_filter = sys.argv[2] if len(sys.argv) > 2 else None
    with open(log, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # ① 开关判定（未开启提示重跑）
    m = re.search(r"get device model args table flag:(\d)", text)
    if not m or m.group(1) != "1":
        logger.error("错误: 日志未开启 GE_DAVINCI_MODEL_PROFILING=2（无 dfx args 表）")
        logger.error(
            "请设置后重新运行取新日志: export GE_DAVINCI_MODEL_PROFILING=2 && export ASCEND_GLOBAL_LOG_LEVEL=1"
        )
        return 1
    logger.info("开关: get device model args table flag=1 ✓")

    # ② 表基址
    if not re.search(r"model_args_table_addr:(0x[0-9a-f]+)", text):
        logger.error(
            "错误: 未找到 model_args_table_addr（需 Print kernelLaunch Op args 日志）"
        )
        return 1

    # ③~⑨ 解析与输出
    p = parse_all(text)
    d2h_diff = len(re.findall(r"Print different args", text))
    r = p.regions
    logger.info(
        f"表基址 model_args_table_addr=0x{p.table_base:x}；F 区实际申请 {r.actual_sz}/{r.total_sz}"
        f"（0x{r.f_start:x}~0x{r.f_end:x}）；D2H 差异 {d2h_diff} 条"
    )
    logger.info("")
    logger.info("==== 静态图算子实际下发地址（input_N/output_N）====")

    # 算子聚合（有段信息的才算子 launch 段，含常量等无段算子跳过）
    for op in sorted(
        {k[0] for k in p.io_lines}, key=lambda o: p.segs.get(o, {}).get("seg", 1 << 62)
    ):
        if op_filter and op_filter not in op:
            continue
        if op in p.segs:
            print_op(op, p)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:  # 管道下游提前关闭（如 | head）时静默退出
        sys.exit(0)
