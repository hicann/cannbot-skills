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

# mhc_post_fusion 专用诊断脚本
#
# 问题：Vector Core 超时（error 507034）
# 怀疑原因：事件ID复用 + 高频同步导致硬件状态机混乱
#
# 使用方法：
#   cd <CAKE2_ROOT>/output/mhc_post_fusion
#   bash diagnose_mhc_post_fusion.sh

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}mhc_post_fusion 超时诊断${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 项目路径（从脚本位置向上回溯到 CAKE2 根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output/mhc_post_fusion"
cd "${OUTPUT_DIR}"

# 1. 环境检查
echo -e "${YELLOW}[1/6] 环境检查${NC}"
echo "-----------------------------------"

# 检查 mssanitizer
if command -v mssanitizer &> /dev/null; then
    MSSAN_PATH=$(which mssanitizer)
    echo -e "${GREEN}✓${NC} mssanitizer: ${MSSAN_PATH}"
    mssanitizer --version | head -1
else
    # 尝试 CANN 路径
    for ver in "latest" "8.3.RC2" "8.3.RC1"; do
        MSSAN_CANDIDATE="/usr/local/Ascend/ascend-toolkit/${ver}/tools/mssanitizer/bin/mssanitizer"
        if [ -x "${MSSAN_CANDIDATE}" ]; then
            MSSAN_PATH="${MSSAN_CANDIDATE}"
            echo -e "${GREEN}✓${NC} mssanitizer: ${MSSAN_PATH}"
            break
        fi
    done
fi

if [ -z "${MSSAN_PATH}" ]; then
    echo -e "${RED}✗${NC} 未找到 mssanitizer"
    echo "请确保已安装 CANN 工具包"
    exit 1
fi

# 检查必要文件
if [ ! -f "${OUTPUT_DIR}/mhc_post_fusion_op_desc.json" ]; then
    echo -e "${RED}✗${NC} 未找到算子配置文件"
    exit 1
fi
echo -e "${GREEN}✓${NC} 算子配置文件存在"

# 关闭 PyTorch 内存池
export PYTORCH_NO_NPU_MEMORY_CACHING=1
echo -e "${GREEN}✓${NC} PyTorch 内存池已关闭"

echo ""

# 2. 同步检测（最优先，因为怀疑是同步问题）
echo -e "${YELLOW}[2/6] 同步检测 (synccheck)${NC}"
echo "-----------------------------------"
echo "检测目标：SetFlag/WaitFlag 配对问题"
echo ""

SYNC_LOG="${OUTPUT_DIR}/mssanitizer_synccheck.log"

${MSSAN_PATH} --tool=synccheck \
    --log-level=info \
    --log-file="${SYNC_LOG}" \
    bash -c "
        cd ${OUTPUT_DIR}
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        timeout 60 python3 -c \"
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from mhc_post_fusion_reference import MhcPostFusionReference
import torch

# 极小 shape 测试
B, S, D, N = 1, 1, 64, 4
op = MhcPostFusionReference(D=N, N2=N*N, OUT_DIM=24)
h_out = torch.randn(B*S, D, dtype=torch.bfloat16)
residual = torch.randn(B*S, N, D, dtype=torch.bfloat16)

# 执行算子
try:
    result = op.forward(h_out, residual)
    print('Test completed')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
\"
" 2>&1 || true

if [ -f "${SYNC_LOG}" ]; then
    echo -e "${BLUE}同步检测日志:${NC}"
    cat "${SYNC_LOG}"

    # 分析结果
    if grep -q "Unpaired set_flag" "${SYNC_LOG}"; then
        echo ""
        echo -e "${RED}❌ 发现未配对的 SetFlag 指令！${NC}"
        echo -e "${YELLOW}建议:${NC} 检查 ComputeXHat 等函数中的 SetFlag/WaitFlag 配对"
    elif grep -q "Redundant set_flag" "${SYNC_LOG}"; then
        echo ""
        echo -e "${RED}❌ 发现冗余的 SetFlag 指令！${NC}"
        echo -e "${YELLOW}建议:${NC} 移除重复的 SetFlag 调用"
    else
        echo ""
        echo -e "${GREEN}✓ 未发现同步配对问题${NC}"
    fi
else
    echo -e "${YELLOW}⚠ 未生成同步检测日志${NC}"
fi

echo ""

# 3. 内存检测
echo -e "${YELLOW}[3/6] 内存检测 (memcheck)${NC}"
echo "-----------------------------------"
echo "检测目标：内存越界、踩踏、泄漏"
echo ""

MEM_LOG="${OUTPUT_DIR}/mssanitizer_memcheck.log"

${MSSAN_PATH} --tool=memcheck \
    --log-level=info \
    --log-file="${MEM_LOG}" \
    --leak-check=yes \
    bash -c "
        cd ${OUTPUT_DIR}
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        timeout 60 python3 -c \"
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from mhc_post_fusion_reference import MhcPostFusionReference
import torch

B, S, D, N = 1, 1, 64, 4
op = MhcPostFusionReference(D=N, N2=N*N, OUT_DIM=24)
h_out = torch.randn(B*S, D, dtype=torch.bfloat16)
residual = torch.randn(B*S, N, D, dtype=torch.bfloat16)

try:
    result = op.forward(h_out, residual)
    print('Test completed')
except Exception as e:
    print(f'Error: {e}')
\"
" 2>&1 || true

if [ -f "${MEM_LOG}" ]; then
    echo -e "${BLUE}内存检测日志:${NC}"
    head -50 "${MEM_LOG}"

    # 快速分析
    if grep -q "illegal read\|illegal write" "${MEM_LOG}"; then
        echo ""
        echo -e "${RED}❌ 发现非法内存访问！${NC}"
    fi
    if grep -q "out of bounds" "${MEM_LOG}"; then
        echo ""
        echo -e "${YELLOW}⚠ 发现多核踩踏风险${NC}"
    fi
fi

echo ""

# 4. 竞争检测
echo -e "${YELLOW}[4/6] 竞争检测 (racecheck)${NC}"
echo "-----------------------------------"
echo "检测目标：RAW/WAR/WAW 数据竞争"
echo ""

RACE_LOG="${OUTPUT_DIR}/mssanitizer_racecheck.log"

${MSSAN_PATH} --tool=racecheck \
    --log-level=info \
    --log-file="${RACE_LOG}" \
    bash -c "
        cd ${OUTPUT_DIR}
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        timeout 60 python3 -c \"
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from mhc_post_fusion_reference import MhcPostFusionReference
import torch

B, S, D, N = 1, 1, 64, 4
op = MhcPostFusionReference(D=N, N2=N*N, OUT_DIM=24)
h_out = torch.randn(B*S, D, dtype=torch.bfloat16)
residual = torch.randn(B*S, N, D, dtype=torch.bfloat16)

try:
    result = op.forward(h_out, residual)
    print('Test completed')
except Exception as e:
    print(f'Error: {e}')
\"
" 2>&1 || true

if [ -f "${RACE_LOG}" ]; then
    echo -e "${BLUE}竞争检测日志:${NC}"
    head -50 "${RACE_LOG}"

    if grep -q "Potential.*hazard" "${RACE_LOG}"; then
        echo ""
        echo -e "${RED}❌ 发现数据竞争！${NC}"
    fi
fi

echo ""

# 5. 源码分析
echo -e "${YELLOW}[5/6] 源码分析${NC}"
echo "-----------------------------------"

KERNEL_CPP="${OUTPUT_DIR}/MhcPostFusionCustom/op_kernel/mhc_post_fusion_custom.cpp"

if [ -f "${KERNEL_CPP}" ]; then
    echo -e "${BLUE}ComputeXHat 函数分析:${NC}"

    # 统计 SetFlag/WaitFlag 数量
    SETFLAG_COUNT=$(grep -o "SetFlag" "${KERNEL_CPP}" | wc -l)
    WAITFLAG_COUNT=$(grep -o "WaitFlag" "${KERNEL_CPP}" | wc -l)

    echo "SetFlag 调用次数: ${SETFLAG_COUNT}"
    echo "WaitFlag 调用次数: ${WAITFLAG_COUNT}"

    # 检查事件ID复用
    echo ""
    echo -e "${BLUE}事件ID使用情况:${NC}"
    grep -n "eIdMte2V\|eIdVS" "${KERNEL_CPP}" | head -20

    # 分析 ComputeXHat 中的循环
    echo ""
    echo -e "${BLUE}ComputeXHat 循环结构:${NC}"
    sed -n '/__aicore__ inline void ComputeXHat/,/^}$/p' "${KERNEL_CPP}" | \
        grep -n "for.*uint32_t" | head -10

else
    echo -e "${YELLOW}⚠ 未找到 kernel 源码${NC}"
fi

echo ""

# 6. 诊断总结与建议
echo -e "${YELLOW}[6/6] 诊断总结${NC}"
echo "========================================"

# 汇总检测结果
HAS_SYNC_ISSUE=false
HAS_MEM_ISSUE=false
HAS_RACE_ISSUE=false

if [ -f "${SYNC_LOG}" ] && grep -q "Unpaired\|Redundant" "${SYNC_LOG}"; then
    HAS_SYNC_ISSUE=true
    echo -e "${RED}❌ 同步问题${NC}"
fi

if [ -f "${MEM_LOG}" ] && grep -q "illegal\|out of bounds" "${MEM_LOG}"; then
    HAS_MEM_ISSUE=true
    echo -e "${RED}❌ 内存问题${NC}"
fi

if [ -f "${RACE_LOG}" ] && grep -q "hazard" "${RACE_LOG}"; then
    HAS_RACE_ISSUE=true
    echo -e "${RED}❌ 竞争问题${NC}"
fi

if [ "$HAS_SYNC_ISSUE" = false ] && [ "$HAS_MEM_ISSUE" = false ] && [ "$HAS_RACE_ISSUE" = false ]; then
    echo -e "${GREEN}✓ 未发现明确的 msSanitizer 检测到的问题${NC}"
    echo ""
    echo -e "${YELLOW}可能的原因:${NC}"
    echo "1. 事件ID复用导致硬件状态机混乱（msSanitizer 可能检测不到）"
    echo "2. 循环边界条件错误导致死循环"
    echo "3. NaN/Inf 数据导致硬件陷阱"
    echo ""
    echo -e "${YELLOW}建议的下一步:${NC}"
    echo "1. 简化 ComputeXHat 函数，仅做初始化测试"
    echo "2. 使用独立的事件ID（不复用 eIdMte2V/eIdVS）"
    echo "3. 检查 tiling 参数（tileD, nTilesD）是否正确"
else
    echo ""
    echo -e "${YELLOW}修复建议:${NC}"

    if [ "$HAS_SYNC_ISSUE" = true ]; then
        echo ""
        echo "同步问题修复："
        echo "1. 检查每个 SetFlag 是否有对应的 WaitFlag"
        echo "2. 移除冗余的 SetFlag 调用"
        echo "3. 考虑使用 TQue 替代 TBuf 以减少手动同步"
    fi

    if [ "$HAS_MEM_ISSUE" = true ]; then
        echo ""
        echo "内存问题修复："
        echo "1. 检查 DataCopy 的偏移和大小"
        echo "2. 添加核间同步避免多核踩踏"
        echo "3. 确保地址满足对齐要求"
    fi

    if [ "$HAS_RACE_ISSUE" = true ]; then
        echo ""
        echo "竞争问题修复："
        echo "1. 添加适当的 SetFlag/WaitFlag 同步"
        echo "2. 调整内存访问顺序"
        echo "3. 使用 TQue 的 EnQue/DeQue 语义"
    fi
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}诊断完成${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "日志文件位置："
echo "  同步检测: ${SYNC_LOG}"
echo "  内存检测: ${MEM_LOG}"
echo "  竞争检测: ${RACE_LOG}"
