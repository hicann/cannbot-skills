#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This example requires the target Linux CANN environment" >&2
  exit 1
fi
if [[ -z "${ASCEND_HOME_PATH:-}" || ! -d "$ASCEND_HOME_PATH" ]]; then
  echo "Source the target CANN set_env.sh first" >&2
  exit 1
fi
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_BUILD_DIR="${EXAMPLE_DIR}/build"
cmake -S "$EXAMPLE_DIR" -B "$EXAMPLE_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$EXAMPLE_BUILD_DIR" --target build_es_custom --parallel
EXAMPLE_WHEEL="${EXAMPLE_BUILD_DIR}/es_output/whl/es_custom-1.0.0-py3-none-any.whl"
test -s "$EXAMPLE_WHEEL"
python3 -m pip install --no-deps --upgrade --target "${EXAMPLE_BUILD_DIR}/whl_package" "$EXAMPLE_WHEEL"
export PYTHONPATH="${EXAMPLE_BUILD_DIR}/whl_package:${EXAMPLE_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${EXAMPLE_BUILD_DIR}/es_output/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# 使用本例的有效插件条目，不带入未知的旧OPP，避免Python扫描失效路径。
export ASCEND_CUSTOM_OPP_PATH="${EXAMPLE_BUILD_DIR}/opp:${EXAMPLE_DIR}/src/ge"
DEVICE_ID="${DEVICE_ID:-0}" python3 "${EXAMPLE_DIR}/src/run.py"
