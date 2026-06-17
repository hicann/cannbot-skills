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

# 核函数 .asc 文件的编辑后钩子：检查常见陷阱。
#
# 该钩子只给出警告（exit 0），而非阻断（exit 2），
# 因为编辑可能是增量进行的，相关问题会在后续步骤中修复。

if ! command -v jq &>/dev/null; then
  echo "WARNING: jq not found; post-edit kernel checks skipped" >&2
  exit 0
fi

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null)

if [ -z "$FILE_PATH" ]; then
  exit 0
fi

WARNINGS=""

# 检查 1：TBuf::Get() 缺少 if constexpr 守卫
# 启发式判断：可能误报。仍需人工核对。
if grep -q 'TBuf.*Get\|\.Get<' "$FILE_PATH" 2>/dev/null; then
  GET_COUNT=$(grep -c 'TBuf.*Get\|\.Get<' "$FILE_PATH" 2>/dev/null || echo "0")
  GUARD_COUNT=$(grep -c 'if constexpr' "$FILE_PATH" 2>/dev/null || echo "0")
  if [ "$GET_COUNT" -gt "$GUARD_COUNT" ]; then
    WARNINGS="${WARNINGS}【警告】发现 $GET_COUNT 处 TBuf::Get() 调用，但只有 $GUARD_COUNT 处 if-constexpr 守卫。每个 Get() 都必须由与其 InitBuffer() 相同的条件守卫。\n"
  fi
fi

# 检查 2：device 侧代码中使用了 host 侧专用辅助函数
if grep -qE '\b(ceil_div|align_down|align_up)\b' "$FILE_PATH" 2>/dev/null; then
  WARNINGS="${WARNINGS}【警告】检测到 host 侧专用辅助函数。构建前请替换为内联算术。\n"
fi

# 检查 3：DataCopyPad 缺少字节数注释
if grep -q 'DataCopyPad' "$FILE_PATH" 2>/dev/null; then
  if ! grep -q 'bytes\|sizeof' "$FILE_PATH" 2>/dev/null; then
    WARNINGS="${WARNINGS}【提醒】DataCopyPad 的 blockLen 以字节为单位，而非元素个数。请核对字节数。\n"
  fi
fi

# 检查 4：host 分发是否覆盖 fp16/half 路径
# 核函数模板 {OP}_kernel<T> 由 host 入口按 dtype 实例化（如 at::kHalf -> <half>）。
if grep -qE '_kernel\s*<|__global__ __aicore__' "$FILE_PATH" 2>/dev/null; then
  if ! grep -qE 'kHalf|<half>|\bhalf\b' "$FILE_PATH" 2>/dev/null; then
    WARNINGS="${WARNINGS}【提醒】host 分发可能缺少 fp16/half（at::kHalf -> {OP}_kernel<half>）路径。\n"
  fi
fi

# 检查 5：Ascend950 Reg API 禁用族
if grep -qE 'AscendC::MicroAPI|MicroAPI|Membase::|Membase|membase' "$FILE_PATH" 2>/dev/null; then
  WARNINGS="${WARNINGS}【警告】检测到禁用的 Ascend950/Reg API 族。请使用 AscendC::Reg；不要使用 MicroAPI 或 Membase。\n"
fi

# 检查 6：除 asc_vf_call 以外的裸 asc_* 调用
RAW_ASC=$(grep -n -E '\basc_[A-Za-z0-9_]+\s*\(' "$FILE_PATH" 2>/dev/null | grep -v 'asc_vf_call' || true)
if [ -n "$RAW_ASC" ]; then
  WARNINGS="${WARNINGS}【警告】检测到裸 asc_* 调用。在 Reg 路径中，asc_vf_call 是唯一允许的裸 asc_* 入口。\n"
fi

# 检查 7：在看起来使用 Reg 的文件中出现经典 AscendC 计算
if grep -q 'AscendC::Reg::\|__simd_vf__\|asc_vf_call' "$FILE_PATH" 2>/dev/null; then
  if grep -qE 'AscendC::(Mul|Muls|Add|Adds|Sub|Div|Exp|Sqrt|Sigmoid|Cast|ReduceSum|Duplicate)\s*\(' "$FILE_PATH" 2>/dev/null; then
    WARNINGS="${WARNINGS}【警告】在看起来使用 Reg 的文件中检测到经典 AscendC 的计算/类型转换/规约调用。请优先使用 __simd_vf__ + AscendC::Reg 封装。\n"
  fi
fi

# 检查 8：Reg 封装形态及常见尾块/类型转换冒险
if grep -q 'AscendC::Reg::' "$FILE_PATH" 2>/dev/null; then
  if ! grep -q '__simd_vf__' "$FILE_PATH" 2>/dev/null || ! grep -q 'asc_vf_call' "$FILE_PATH" 2>/dev/null; then
    WARNINGS="${WARNINGS}【提醒】Reg 计算通常应使用通过 asc_vf_call 封装调用的 __simd_vf__ 函数。\n"
  fi
  if grep -q 'StoreAlign' "$FILE_PATH" 2>/dev/null && ! grep -q 'UpdateMask\|CreateMask' "$FILE_PATH" 2>/dev/null; then
    WARNINGS="${WARNINGS}【提醒】Reg 存储应由显式的整块/尾块掩码保护；掩码为元素个数，而非字节数。\n"
  fi
  if grep -q 'Reg::Cast' "$FILE_PATH" 2>/dev/null; then
    if ! grep -q 'CastTrait' "$FILE_PATH" 2>/dev/null; then
      WARNINGS="${WARNINGS}【提醒】Reg 类型转换应指定 AscendC::Reg::CastTrait。\n"
    fi
    if grep -q 'bfloat16_t\|half\|uint16_t\|std::uint16_t' "$FILE_PATH" 2>/dev/null; then
      if ! grep -q 'DIST_UNPACK_B16\|DIST_PACK_B32' "$FILE_PATH" 2>/dev/null; then
        WARNINGS="${WARNINGS}【提醒】B16 的 Reg 类型转换通常需要 DIST_UNPACK_B16 和/或 DIST_PACK_B32。\n"
      fi
    fi
  fi
  if grep -q 'GetValue' "$FILE_PATH" 2>/dev/null; then
    WARNINGS="${WARNINGS}【提醒】避免通过 LocalTensor::GetValue() 取出 Reg 产生的标量值；请将标量的最终化与广播保留在 Reg 封装内部。\n"
  fi
  # 可选的 VF 融合上限：仅当环境变量 VF_FUSION_LIMIT 设为正整数时才生效。
  # 未设置时不检查任何上限（默认）。统计每个 __simd_vf__ ... { ... } 块内的
  # AscendC::Reg::{Add,Adds,Sub,Mul,Muls,Div,Max,Exp,Sqrt,Reduce,Cast,Duplicate} 调用；
  # load/store/掩码辅助函数不计入。
  if [ -n "${VF_FUSION_LIMIT:-}" ] && [ "${VF_FUSION_LIMIT}" -gt 0 ] 2>/dev/null; then
    OVER_FUSION=$(awk -v limit="$VF_FUSION_LIMIT" '
      /__simd_vf__/ { in_fn = 1; depth = 0; count = 0; saw_open = 0 }
      in_fn {
        s = $0
        while ((p = index(s, "{")) > 0) { depth++; saw_open = 1; s = substr(s, p + 1) }
        s = $0
        while ((p = index(s, "}")) > 0) { depth--; s = substr(s, p + 1) }
        s = $0
        while (match(s, /AscendC::Reg::(Add|Adds|Sub|Mul|Muls|Div|Max|Exp|Sqrt|Reduce|Cast|Duplicate)[^A-Za-z0-9_]/)) {
          count++
          s = substr(s, RSTART + RLENGTH)
        }
        if (saw_open && depth == 0) {
          if (count > limit) print count
          in_fn = 0
        }
      }
    ' "$FILE_PATH" 2>/dev/null)
    if [ -n "$OVER_FUSION" ]; then
      WARNINGS="${WARNINGS}【警告】某个 __simd_vf__ 函数融合的 VF 计算指令数超过了配置的 VF_FUSION_LIMIT=${VF_FUSION_LIMIT}。请拆分为多个封装，并通过各自独立的 asc_vf_call 调用串接。\n"
    fi
  fi
fi

if [ -n "$WARNINGS" ]; then
  printf "%b" "$WARNINGS" >&2
fi

exit 0
