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
"""生成 ATC 单算子描述文件（JSON）并提示转换命令。

用法:
    python3 gen_singleop_json.py --op Add \\
        --input x1:16,16:float16 \\
        --input x2:16,16:float16 \\
        --output y:16,16:float16 \\
        [-a strides:list_int:1,1,1,1] \\
        [--format ND] [--name my_add] \\
        -o op_list.json

    --input/--output 条目格式: <name>:<dim,dim,...>:<dtype>，name 可省略（x0/x1.../y0...）
    --attr 条目格式: <name>:<type>:<value>（value 按逗号拆为 list，单个值保持标量）

单算子描述文件官方说明见 references/atc_tools_guide.md（单算子描述文件转换成离线模型）与 references/atc_official_digest.md（Ascend IR JSON 描述文件格式）。
"""

import argparse
import json
import sys

KNOWN_DTYPES = {
    "float32",
    "float16",
    "bfloat16",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
    "double",
}
KNOWN_FORMATS = {"ND", "NCHW", "NHWC", "NC1HWC0", "NDHWC", "HWCN", "FRACTAL_NZ"}


def _coerce_scalar(v: str):
    if v.lstrip("-").replace(".", "", 1).isdigit():
        return float(v) if "." in v else int(v)
    if v in ("true", "false"):
        return v == "true"
    return v


def parse_tensor(spec: str, default_name: str, fmt: str) -> dict:
    parts = spec.split(":")
    if len(parts) == 3:
        name, shape_s, dtype = parts
    elif len(parts) == 2:
        name, (shape_s, dtype) = default_name, parts
    else:
        raise ValueError(
            f"张量描述 '{spec}' 应为 <name>:<dims>:<dtype> 或 <dims>:<dtype>"
        )
    try:
        shape = [int(d) for d in shape_s.split(",") if d.strip()]
    except ValueError as e:
        raise ValueError(f"shape '{shape_s}' 含非整数维度") from e
    if not shape:
        raise ValueError(f"shape '{shape_s}' 为空")
    if dtype not in KNOWN_DTYPES:
        raise ValueError(
            f"dtype '{dtype}' 不在已知集合 {sorted(KNOWN_DTYPES)}（请核对算子支持列表）"
        )
    if fmt not in KNOWN_FORMATS:
        raise ValueError(f"format '{fmt}' 不在已知集合 {sorted(KNOWN_FORMATS)}")
    return {"format": fmt, "type": dtype, "name": name, "shape": shape}


def parse_attr(spec: str) -> dict:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"属性描述 '{spec}' 应为 <name>:<type>:<value>")
    name, atype, value = parts
    if atype.startswith("list_"):
        val = [_coerce_scalar(v) for v in value.split(",")]
    else:
        val = _coerce_scalar(value)
    return {"name": name, "type": atype, "value": val}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--op",
        required=True,
        help="算子类型，须为运行环境已注册算子（内置或 vendor 自定义）",
    )
    p.add_argument("--input", action="append", default=[], help="<name>:<dims>:<dtype>")
    p.add_argument(
        "--output",
        action="append",
        default=[],
        dest="outputs",
        help="<name>:<dims>:<dtype>",
    )
    p.add_argument("--attr", action="append", default=[], help="<name>:<type>:<value>")
    p.add_argument("--format", default="ND", help="统一 format，默认 ND")
    p.add_argument("--name", default=None, help="算子实例名（可选）")
    p.add_argument("-o", "--out", required=True, help="输出 JSON 路径")
    args = p.parse_args()

    if not args.input or not args.outputs:
        print("✗ 至少需要一个 --input 和一个 --output", file=sys.stderr)
        return 2

    try:
        op_desc = {
            "op": args.op,
            "input_desc": [
                parse_tensor(s, f"x{i}", args.format) for i, s in enumerate(args.input)
            ],
            "output_desc": [
                parse_tensor(s, f"y{i}", args.format)
                for i, s in enumerate(args.outputs)
            ],
        }
        if args.attr:
            op_desc["attr"] = [parse_attr(s) for s in args.attr]
        if args.name:
            op_desc["name"] = args.name
    except ValueError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump([op_desc], f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"✓ 已生成 {args.out}")
    print("  转换命令:")
    print(f"    atc --singleop={args.out} --soc_version=<soc> --output=<目录> \\")
    print(
        "        [--precision_mode=must_keep_origin_dtype]  # fp32 输入建议加，避免默认 force_fp16 插 Cast"
    )
    print("  注意: 单算子产物文件名含 shape/dtype 枚举，修改描述文件后产物名会变化，")
    print("        设备侧引用的模型文件名需同步更新")
    return 0


if __name__ == "__main__":
    sys.exit(main())
