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
# 设备验证特有项预检（步骤 0）
# 常规 host/CANN 环境检查委托 ascendc-kirin-env-check；hdc 定位/连接自检委托 tool-hdc。
# 设备链路可独立暂停/恢复（见 references/device_verify_examples.md P8）
# 用法:
#   bash check_onboard_env.sh                # vendor 装入 + HAP 就绪 + hdc 自检
#   bash check_onboard_env.sh --skip-device  # 跳过 hdc 自检
#   bash check_onboard_env.sh --hap <hap文件># 附带指定 HAP 文件检查（位置参数同样支持）
# 退出码: 0=全部通过  1=存在 FAIL
# 说明: 本脚本只检查设备侧链路特有项（vendor 装入、HAP 就绪、hdc 连接）；
#       host/CANN 环境请运行 ascendc-kirin-env-check 的 check_kirin_env.sh；
#       hdc 命令细节以 tool-hdc 为准。

PASS=0; FAIL=0; WARN=0
ok()   { echo "[PASS] $1"; PASS=$((PASS+1)); }
bad()  { echo "[FAIL] $1"; FAIL=$((FAIL+1)); }
warn() { echo "[WARN] $1"; WARN=$((WARN+1)); }

HAP_FILE=""
SKIP_DEVICE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-device) SKIP_DEVICE=1 ;;
        -h|--hap)
            [ -n "$2" ] || { echo "[ERROR] $1 需跟文件路径"; exit 2; }
            HAP_FILE="$2"; shift ;;
        -h=*|--hap=*)
            HAP_FILE="${1#*=}" ;;
        *)
            HAP_FILE="$1" ;;
    esac
    shift
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "=== 设备验证特有项检查 ==="
echo "（host/CANN 环境见 ascendc-kirin-env-check；hdc 命令见 tool-hdc）"

# 1. vendor 已装入 opp（需先 source set_env.sh）
if [ -n "${ASCEND_OPP_PATH}" ] && [ -d "${ASCEND_OPP_PATH}" ]; then
    if [ -d "${ASCEND_OPP_PATH}/vendors" ]; then
        vn=$(find "${ASCEND_OPP_PATH}/vendors" -mindepth 1 -maxdepth 1 -printf '%f ' 2>/dev/null)
        if [ -n "${vn}" ]; then
            ok "已安装 vendor: ${vn}（改码后须 uninstall→重编→重装，安装器不覆盖旧文件）"
        else
            warn "opp/vendors 为空（步骤 1~2 未完成或未装到当前工具链）"
        fi
    else
        warn "opp/vendors 目录不存在（步骤 2 未执行）"
    fi
else
    warn "ASCEND_OPP_PATH 未设置或目录不存在，无法检查 vendor 装入；请先运行 ascendc-kirin-env-check 或 source set_env.sh"
fi

# 2. hdc 定位与连接（委托 tool-hdc/find_hdc.sh）
if [ "${SKIP_DEVICE}" -eq 1 ]; then
    echo "=== 已跳过 hdc 连接自检（--skip-device） ==="
else
    FIND_HDC="$SCRIPT_DIR/../../../tools/tool-hdc/scripts/find_hdc.sh"
    if [ -f "$FIND_HDC" ]; then
        bash "$FIND_HDC" || warn "hdc 未就绪（定位/连接/错误码见 tool-hdc skill）"
    else
        warn "未找到 tool-hdc 自检脚本；hdc 定位与连接检查请见 tool-hdc skill（scripts/find_hdc.sh）"
    fi
fi

# 3. HAP 文件
if [ -n "${HAP_FILE}" ] && [ -f "${HAP_FILE}" ]; then
    ok "HAP 文件: ${HAP_FILE}"
elif [ -n "${HAP_FILE}" ]; then
    bad "HAP 文件不存在: ${HAP_FILE}（下载方式见 references/device_verify_examples.md P9）"
else
    warn "未传入 HAP 文件路径（步骤 5 前需准备）"
fi

echo "=== 汇总: PASS=${PASS} WARN=${WARN} FAIL=${FAIL} ==="
[ "${FAIL}" -eq 0 ]
