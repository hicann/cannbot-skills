#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
计算 Chrome Trace 文件的时间跨度。

跨度定义：所有 ph == "X" 的事件中，最早 ts 到最晚 (ts + dur) 的时间差。
"""

import argparse
import json
from pathlib import Path
from typing import Iterable

from op_autoresearch.utils.console import emit

TraceEvent = dict[str, object]


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="计算完整事件覆盖的时间跨度")
    parser.add_argument("input", type=Path, metavar="TRACE_JSON")
    return parser


def parse_args():
    return _argument_parser().parse_args()


def calc_span(trace_events: Iterable[TraceEvent]) -> float:
    bounds = None
    for event in trace_events:
        if event.get("ph") != "X":
            continue
        started = float(event.get("ts") or 0)
        ended = started + float(event.get("dur") or 0)
        if bounds is None:
            bounds = [started, ended]
        else:
            bounds[0] = min(bounds[0], started)
            bounds[1] = max(bounds[1], ended)
    if bounds is None:
        raise ValueError("未找到 ph==X 的事件")
    return bounds[1] - bounds[0]


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"文件不存在: {args.input}")
    with args.input.open(encoding="utf-8") as stream:
        document = json.load(stream)
    emit(calc_span(document.get("traceEvents", ())))


if __name__ == "__main__":
    main()
