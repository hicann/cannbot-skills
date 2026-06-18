#!/bin/bash

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# 启动 opencode serve + 跑单个算子 bench
# 用法: bash run_bench.sh [op_name] [timeout_sec]

set -e

OP_NAME="${1:-level1/exp}"
TIMEOUT="${2:-3600}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$BENCH_DIR"

# serve 生命周期完全交给 run_eval.py 的 ServeManager 管理
# 它会为每个算子的隔离工作目录启动正确 cwd 的 serve，并在 cwd 变化时自动重启

echo "=== 开始评测: $OP_NAME ==="
OP_TIMEOUT="$TIMEOUT" OPS_FILTER="$OP_NAME" python3 -u runner/run_eval.py --all --skip-isolation-check
RC=$?

echo "=== 完成 (exit=$RC) ==="
exit $RC
