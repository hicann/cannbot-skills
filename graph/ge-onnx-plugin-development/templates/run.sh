#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 导出 → atc 编译 → ACL 执行一键脚本模板。
# TODO(填充): MY_OP/model 文件名按实际替换。

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_DIR="${SCRIPT_DIR}/output"
SOC_VERSION="${SOC_VERSION:-Ascend910B1}"   # TODO(填充): 完整芯片名（npu-smi info 查询）

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ERROR: please source CANN set_env.sh first" >&2
    exit 1
fi
if ! command -v atc >/dev/null 2>&1; then
    echo "ERROR: atc is not available in PATH" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
python3 "${SCRIPT_DIR}/export_onnx.py" --output "${OUTPUT_DIR}/my_model.onnx"

# 插件目录单独存放：GE 扫描 ASCEND_CUSTOM_OPP_PATH 下的一层 Python 文件，
# 导出器/执行器（依赖 torch/numpy/acl）不应作为插件被加载。
export ASCEND_CUSTOM_OPP_PATH="${SCRIPT_DIR}/plugin:${ASCEND_CUSTOM_OPP_PATH:-}"
atc \
    --model="${OUTPUT_DIR}/my_model.onnx" \
    --framework=5 \
    --output="${OUTPUT_DIR}/my_model" \
    --soc_version="${SOC_VERSION}"

python3 "${SCRIPT_DIR}/run_model.py" --model "${OUTPUT_DIR}/my_model.om"
