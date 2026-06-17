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

# 构建前 hook：在核函数源文件中 grep 仅 host 侧可用的辅助函数，
# 这些函数在 __aicore__ device 侧编译上下文中会编译失败。
#
# 假定当前工作目录为项目根目录（Claude Code hook 的标准行为）。
#
# Exit 0 = 允许构建
# Exit 2 = 阻断构建（发现仅 host 侧可用的辅助函数）

KERNEL_DIRS=""
# 待扫描的核函数源码目录候选。可通过 KERNEL_SRC_DIRS 环境变量覆盖（空格分隔），
# 以适配不同工程布局。新布局中算子为扁平的 `operators/{OP}/{OP}.asc`，hook 在算子
# 工程根目录触发，故默认扫描当前目录（递归，排除 build/）。
CANDIDATE_DIRS="${KERNEL_SRC_DIRS:-.}"
for dir in $CANDIDATE_DIRS; do
  if [ -d "$dir" ]; then
    KERNEL_DIRS="$KERNEL_DIRS $dir"
  fi
done

if [ -z "$KERNEL_DIRS" ]; then
  exit 0
fi

# 已知仅 host 侧可用、在 device 侧编译会失败的辅助函数（历史上已失败 3 次以上）。
# 发现新的违规项时，请扩展此列表。
HOST_ONLY_PATTERN='ceil_div|align_down|align_up'

MATCHES=$(grep -rn -E "\b($HOST_ONLY_PATTERN)\b" --include='*.cpp' --include='*.asc' --exclude-dir=build --exclude-dir=.git $KERNEL_DIRS 2>/dev/null || true)

if [ -n "$MATCHES" ]; then
  echo "【已阻断】：核函数代码中发现仅 host 侧可用的辅助函数（将在 __aicore__ 编译时失败）：" >&2
  echo "$MATCHES" >&2
  echo "" >&2
  echo "请替换为内联算术：例如使用 (a + b - 1) / b 代替 ceil_div(a, b)" >&2
  exit 2
fi

# Ascend950 Reg API 防护栏。这些检查刻意采用词法层面匹配：在缓慢的构建
# 进入 device 侧编译之前，提前捕获常见的禁用 API。
FORBIDDEN_PATTERN='AscendC::MicroAPI|MicroAPI|Membase::|Membase|membase'
FORBIDDEN_MATCHES=$(grep -rn -E "$FORBIDDEN_PATTERN" --include='*.cpp' --include='*.asc' --exclude-dir=build --exclude-dir=.git $KERNEL_DIRS 2>/dev/null || true)

if [ -n "$FORBIDDEN_MATCHES" ]; then
  echo "【已阻断】：核函数代码中发现禁用的 Ascend950/Reg API 系列：" >&2
  echo "$FORBIDDEN_MATCHES" >&2
  echo "" >&2
  echo "算子计算请使用 AscendC::Reg；不要使用 MicroAPI 或 Membase。" >&2
  exit 2
fi

RAW_ASC_MATCHES=$(grep -rn -E '\basc_[A-Za-z0-9_]+\s*\(' --include='*.cpp' --include='*.asc' --exclude-dir=build --exclude-dir=.git $KERNEL_DIRS 2>/dev/null | grep -v 'asc_vf_call' || true)

if [ -n "$RAW_ASC_MATCHES" ]; then
  echo "【已阻断】：核函数代码中发现原始的 asc_* C API 调用：" >&2
  echo "$RAW_ASC_MATCHES" >&2
  echo "" >&2
  echo "在 Reg 路径中，asc_vf_call 是唯一允许的原始 asc_* 入口。" >&2
  exit 2
fi

CLASSIC_COMPUTE_PATTERN='AscendC::(Mul|Muls|Add|Adds|Sub|Div|Exp|Sqrt|Sigmoid|Cast|ReduceSum|Duplicate)\s*\('
REG_MODE_FILES=$(
  {
    grep -rl -E 'AscendC::Reg::|__simd_vf__|asc_vf_call' --include='*.cpp' --include='*.asc' --exclude-dir=build --exclude-dir=.git $KERNEL_DIRS 2>/dev/null
    find $KERNEL_DIRS -type f \( -name '*.cpp' -o -name '*.asc' \) -path '*dav-3510*' -not -path '*/build/*' 2>/dev/null
  } | sort -u
)
CLASSIC_COMPUTE_MATCHES=""
if [ -n "$REG_MODE_FILES" ]; then
  CLASSIC_COMPUTE_MATCHES=$(grep -n -E "$CLASSIC_COMPUTE_PATTERN" $REG_MODE_FILES 2>/dev/null | grep -v 'AscendC::Reg::' || true)
fi

if [ -n "$CLASSIC_COMPUTE_MATCHES" ]; then
  echo "【已阻断】：核函数代码中发现经典 AscendC 的 compute/cast/reduce 调用：" >&2
  echo "$CLASSIC_COMPUTE_MATCHES" >&2
  echo "" >&2
  echo "对于 Ascend950 Reg 路径，请使用 __simd_vf__ + AscendC::Reg + asc_vf_call 封装向量 compute/cast/reduce。" >&2
  exit 2
fi

exit 0
