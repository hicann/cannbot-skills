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

"""隔离检查脚本：验证 example 目录是否干净, 无残留生成物。

用法:
  python runner/isolation_check.py
"""

import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINE_OPS = {
    "aclnn_launch_example": {"add", "sqrt", "mish", "_common"},
    "direct_launch_example": {"add", "sqrt"},
}


def verify_isolation(cann_bench_root: str = None) -> bool:
    if cann_bench_root is None:
        from setup_cann_bench import ensure_cann_bench
        cann_bench_root = ensure_cann_bench()

    examples_dir = os.path.join(cann_bench_root, "examples")
    violations = []

    for example_name, baseline in BASELINE_OPS.items():
        ops_dir = os.path.join(examples_dir, example_name, "csrc", "ops")
        if not os.path.isdir(ops_dir):
            continue
        all_ops = {d for d in os.listdir(ops_dir)
                   if os.path.isdir(os.path.join(ops_dir, d))}
        extra = all_ops - baseline
        if extra:
            violations.append(f"  {example_name}/csrc/ops/ 存在非 baseline 目录: {extra}")

        init_py = os.path.join(examples_dir, example_name, "cann_bench", "__init__.py")
        if os.path.exists(init_py):
            result = subprocess.run(
                ["git", "diff", "--name-only", "--", init_py],
                cwd=cann_bench_root,
                capture_output=True, text=True,
            )
            if result.stdout.strip():
                violations.append(f"  {example_name}/cann_bench/__init__.py 已被修改")

    for root, dirs, files in os.walk(examples_dir):
        for d in ["build", "build_py", "dist", "__pycache__"]:
            if d in dirs:
                violations.append(
                    f"  {os.path.relpath(os.path.join(root, d), cann_bench_root)} 残留")

    tgz = os.path.join(cann_bench_root, "examples.tgz")
    if os.path.exists(tgz):
        violations.append(f"  examples.tgz 残留 (可能包含生成源码)")

    if violations:
        print("\n[隔离检查 FAILED] 检测到以下污染源:")
        for v in violations:
            print(v)
        print("\n请先运行 cleanup.py 清理。")
        return False

    print("[隔离检查 OK] 所有 example 目录干净。")
    return True


if __name__ == "__main__":
    sys.exit(0 if verify_isolation() else 1)
