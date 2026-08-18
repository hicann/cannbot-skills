#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Skill script: 算子源码目录定位。

属 ops-knowledge-optimization-ingest skill。支持两种目录布局（无需仓库根目录）：
  - 标准两级：<dataset_dir>/<category>/<op>
  - 三级：<dataset_dir>/<ops-pkg>/<category>/<op>（自动通配发现）

用法::

    python resolve_source.py --dataset_dir <d> --category <c> --op <o>
        -> stdout: 解析出的目录绝对路径（找不到则空串，退出码 1）
"""
import argparse
import glob
import logging
import os
import sys

# Dedicated stdout channel for the resolved path (the script's CLI contract):
# a bare "%(message)s" formatter keeps the output byte-identical to a plain print.
_stdout_logger = logging.getLogger(__name__ + ".stdout")
_stdout_logger.setLevel(logging.INFO)
_stdout_logger.propagate = False
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(logging.Formatter("%(message)s"))
_stdout_logger.addHandler(_stdout_handler)


def resolve_source_dir(dataset_dir: str, category: str, op: str) -> str:
    """返回首个存在的算子源码目录，找不到则返回空串。"""
    direct = os.path.join(dataset_dir, category, op)
    if os.path.isdir(direct):
        return direct
    for cand in sorted(glob.glob(os.path.join(dataset_dir, "*", category, op))):
        if os.path.isdir(cand):
            return cand
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", required=True, help="算子数据集根目录")
    parser.add_argument("--category", required=True, help="算子类别，如 norm")
    parser.add_argument("--op", required=True, help="算子名，如 add_rms_norm_quant")
    args = parser.parse_args()

    result = resolve_source_dir(args.dataset_dir, args.category, args.op)
    _stdout_logger.info(result)
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
