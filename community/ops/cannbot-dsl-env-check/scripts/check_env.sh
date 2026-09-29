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

# CANNBotDSL 环境检查入口。业务逻辑由 env_check.py 实现。
set -eu

if [ "$#" -ne 2 ] || [ "$1" != "--output-dir" ] || [ -z "$2" ]; then
    echo "用法: $0 --output-dir <artifact-root>" >&2
    exit 2
fi
OUTPUT_DIR=$2

PY=${PYTHON:-python3}
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "错误: 找不到 Python 解释器：$PY" >&2
    exit 2
fi
if ! "$PY" -c 'import yaml' >/dev/null 2>&1; then
    echo "错误: 当前解释器缺少 PyYAML：$PY" >&2
    echo "安装: $PY -m pip install pyyaml" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PY" "$SCRIPT_DIR/env_check.py" \
    --output-dir "$OUTPUT_DIR"
