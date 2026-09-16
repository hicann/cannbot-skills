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

"""提取 RT2.0 动态执行日志中算子实际下发地址（input_N/output_N 格式，按执行序）。

用法: python3 addr_extract.py <日志文件>
原理（见 references/addr-reference.md 路径三）:
  ① [KernelTrace][LaunchKernelWithHandle_<算子名>_<id>]Input/Output addresses: <地址...>  launch 时 device 地址实值
  ② 同算子 Launch args-args addresses info 中 ArgsInputsAddr/ArgsOutputsAddr 的 length/8 = 输入/输出指针数
     据此切分 ① 的地址列表：前 N 个为 input_0..N-1，其余为 output_0..M-1
输出: 按 step 执行序（日志时间序）逐条列出，每次执行可直接对应地址；档位由尺寸标注体现
"""

import logging
import re
import signal
import sys

logger = logging.getLogger(__name__)


def hsz(b):
    """字节数转人类可读字符串。"""
    if b >= 1048576:
        return f"{b / 1048576:.0f}M"
    if b >= 1024:
        return f"{b / 1024:.0f}K"
    return f"{b}B"


def main():
    """解析日志并按执行轮次输出算子下发地址，返回退出码。"""
    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
    # 管道下游提前关闭（如 | head）时静默退出，避免 logging flush 报错刷屏
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    if len(sys.argv) != 2:
        logger.error(f"用法: python3 {sys.argv[0]} <日志文件>")
        return 1

    with open(sys.argv[1], encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # 每次 launch 记录：时间戳、算子、地址列表、尺寸列表（与地址同序：input 在前 output 在后）
    recs = []
    for m in re.finditer(
        r"\):(\d{4}-\d{2}-\d{2}-[\d.:]+) \[[\w.]+:\d+\]\d+ OnExecuteEvent:"
        r"\[KernelTrace\]\[LaunchKernelWithHandle_(\S+?)\]Input/Output addresses: ([^,]+), "
        r"Input/Output sizes: ([^,\n]+)",
        text,
    ):
        recs.append(
            (
                m.group(1),
                m.group(2),
                tuple(m.group(3).split()),
                tuple(int(s) for s in m.group(4).split()),
            )
        )

    # input/output 指针数（args 结构长度）
    counts = {}
    for m in re.finditer(
        r"\[LaunchKernelWithHandle_(\S+?)\]Launch args-args addresses info.*?"
        r"ArgsInputsAddr\(address/length\): \S+/(\d+), ArgsOutputsAddr\(address/length\): \S+/(\d+)",
        text,
    ):
        counts[m.group(1)] = (int(m.group(2)) // 8, int(m.group(3)) // 8)

    logger.info("==== 算子实际下发地址（按执行轮次 step，input_N/output_N）====")
    # 轮次划分：同一算子再次 launch（且属于一轮内已见集合）= 新一轮图执行（多轮循环执行的通用判定；
    # 时间间隔不可靠——轮内算子真实计算耗时可大于轮间间隔）
    step = 0
    seen = set()
    for ts, op_id, addrs, sizes in recs:
        if op_id in seen:
            step += 1
            seen = set()
        if len(seen) == 0:
            step = max(step, 1)
            logger.info(
                f"\n==== step {step}（{ts.split('-', 3)[-1]} 起，档位 {hsz(max(sizes))}）===="
            )
        seen.add(op_id)
        logger.info(f"[{op_id}]")
        n_in, _ = counts.get(op_id, (0, 0))
        for i, a in enumerate(addrs):
            tag = f"input_{i}" if i < n_in else f"output_{i - n_in}"
            sz = sizes[i] if i < len(sizes) else 0
            logger.info(f"    {tag:9s} {a}  ({hsz(sz)})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:  # 管道下游提前关闭（如 | head）时静默退出
        sys.exit(0)
