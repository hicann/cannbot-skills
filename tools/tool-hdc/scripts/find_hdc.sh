#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 定位 hdc 可执行文件并自检连接（对齐 skill 验证要求：hdc list targets 确认有可用设备）
# 用法：bash find_hdc.sh [-v]
#   -v  额外执行 hdc list targets -v 查看设备详情
set -u

VERBOSE=false
[ "${1:-}" = "-v" ] && VERBOSE=true

print_ok()  { echo "✓ $*"; }
print_warn(){ echo "⚠ $*"; }
print_err() { echo "✗ $*" >&2; }

# 1. 候选路径：PATH 优先，其次常见 SDK toolchains 路径（官方文档口径）
CANDIDATES=(
    "$(command -v hdc 2>/dev/null || true)"
    "$HOME/DevEco-Studio/sdk/default/openharmony/toolchains/hdc"
    "/opt/command-line-tools/sdk/default/openharmony/toolchains/hdc"
    "/usr/local/bin/hdc"
)

HDC=""
for c in "${CANDIDATES[@]}"; do
    [ -n "$c" ] && [ -x "$c" ] && HDC="$c" && break
done

if [ -z "$HDC" ]; then
    print_err "未找到 hdc 可执行文件"
    echo "  获取方式（二选一，见官方文档）："
    echo "    1. DevEco Studio 内嵌 SDK：{DevEco Studio}/sdk/default/openharmony/toolchains/"
    echo "    2. Command Line Tools：{CLT}/sdk/default/openharmony/toolchains/"
    echo "  独立运行需 hdc + libusb_shared 同目录，并将该目录加入 PATH"
    exit 1
fi
print_ok "hdc: $HDC"

# 2. 版本（判断 -b/远程转发等特性可用性）
ver=$("$HDC" version 2>/dev/null | head -1)
if [ -n "$ver" ]; then
    print_ok "版本: $ver"
else
    print_warn "无法获取版本（检查依赖 libusb_shared）"
fi

# 3. 服务与设备（验证要求：确认有可用设备）
if "$HDC" list targets >/dev/null 2>&1; then
    targets=$("$HDC" list targets 2>/dev/null)
    if [ -z "$targets" ] || [ "$targets" = "[Empty]" ]; then
        print_warn "服务正常，但未发现设备（排查：USB 调试开关 / 数据线 / 授权弹窗）"
    else
        print_ok "已连接设备:"
        while IFS= read -r line; do
            echo "    $line"
        done <<< "$targets"
    fi
else
    print_warn "hdc 服务异常，可尝试: hdc kill -r"
fi

if [ "$VERBOSE" = true ]; then
    echo "--- hdc list targets -v ---"
    "$HDC" list targets -v 2>/dev/null || true
fi
