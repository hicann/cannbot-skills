#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""评测进度显示。

tqdm 可用时使用 tqdm 进度条; 否则回退到内置文本进度行
(每算子完成时打印 [###---] 百分比/已耗时/ETA)。不引入强制依赖。
"""

import logging
import sys
import time


class _StdoutLogHandler(logging.StreamHandler):
    """emit 时再解析 sys.stdout, 兼容 pytest capsys 的 stdout 替换。"""

    def emit(self, record):
        self.stream = sys.stdout
        super().emit(record)


_log = logging.getLogger("eval_progress")
if not any(isinstance(h, _StdoutLogHandler) for h in _log.handlers):
    _handler = _StdoutLogHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)
    _log.propagate = False


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds or 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class EvalProgress:
    """跨算子进度跟踪: tqdm 进度条或文本进度行。"""

    BAR_WIDTH = 20

    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.start_time = time.time()
        self._durations: list[float] = []
        self._bar = None
        try:
            from tqdm import tqdm
            self._bar = tqdm(total=total, unit="op", desc="评测进度",
                             dynamic_ncols=True)
        except ImportError:
            self._bar = None

    @property
    def use_tqdm(self) -> bool:
        return self._bar is not None

    def start_op(self, index: int, op_name: str):
        """第 index 个算子 (0 基) 开始。"""
        if self._bar is not None:
            self._bar.set_description(f"评测 {index + 1}/{self.total}")
            self._bar.set_postfix_str(op_name.split("/")[-1], refresh=True)
        else:
            _log.info("\n[%d/%d] 评测: %s", index + 1, self.total, op_name)

    def finish_op(self, status: str, duration_s: float):
        """算子完成: 更新进度条或打印文本进度行。"""
        self.done += 1
        self._durations.append(duration_s)
        if self._bar is not None:
            self._bar.update(1)
            self._bar.set_postfix_str(
                f"last={status} {_fmt_duration(duration_s)}", refresh=True)
            return
        elapsed = time.time() - self.start_time
        filled = int(self.BAR_WIDTH * self.done / max(self.total, 1))
        bar = "#" * filled + "-" * (self.BAR_WIDTH - filled)
        pct = 100.0 * self.done / max(self.total, 1)
        _log.info("  [PROGRESS] [%s] %d/%d (%.0f%%) | 已耗时 %s | ETA %s",
                  bar, self.done, self.total, pct,
                  _fmt_duration(elapsed), _fmt_duration(self._eta()))

    def write(self, msg: str):
        """打印一行消息: tqdm 模式下走 tqdm.write 避免打断进度条。"""
        if self._bar is not None:
            from tqdm import tqdm
            tqdm.write(msg)
        else:
            _log.info("%s", msg)

    def close(self):
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def _eta(self) -> float:
        """基于已完成算子平均耗时估算剩余时间。"""
        if not self._durations:
            return 0.0
        avg = sum(self._durations) / len(self._durations)
        return avg * (self.total - self.done)
