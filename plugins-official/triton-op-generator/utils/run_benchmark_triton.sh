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

# 批量调度 triton-ascend-coder，支持多 NPU 并行
#
# 支持两种模式：
# 1. 单 NPU 模式（--npu）：串行执行，向后兼容
# 2. 多 NPU 并行模式（--npu-list）：NPU 间并行，NPU 内串行
#
# 用法:
#   # 单 NPU 模式
#   bash utils/run_benchmark_triton.sh --benchmark-dir /path/to/KernelBench --level 1 --range 41-53 --npu 0 --output /path/to/output
#
#   # 多 NPU 并行模式
#   bash utils/run_benchmark_triton.sh --benchmark-dir /path/to/KernelBench --level 1 --range 1-30 --npu-list "0,1,2,3,4,5" --output /path/to/output

set -euo pipefail

# ── 默认值 ──
BENCHMARK_DIR=""
LEVEL=""
RANGE=""
IDS=""
NPU_ID=0
NPU_LIST=""
OUTPUT_DIR=""
ARCH="ascend910b2"
CLAUDE_PROJECT_DIR=""

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark-dir) BENCHMARK_DIR="$2"; shift 2 ;;
        --level)         LEVEL="$2"; shift 2 ;;
        --range)         RANGE="$2"; shift 2 ;;
        --ids)           IDS="$2"; shift 2 ;;
        --npu)           NPU_ID="$2"; shift 2 ;;
        --npu-list)      NPU_LIST="$2"; shift 2 ;;
        --output)        OUTPUT_DIR="$2"; shift 2 ;;
        --arch)          ARCH="$2"; shift 2 ;;
        --claude-project-dir) CLAUDE_PROJECT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "用法: bash utils/run_benchmark_triton.sh --benchmark-dir <path> --level <N> [--range <start-end> | --ids <id_list>] [--npu <id> | --npu-list <list>] --output <path>"
            echo ""
            echo "参数:"
            echo "  --benchmark-dir  KernelBench 根目录路径 (必填)"
            echo "  --level          Level 编号，如 1, 2, 3 (必填)"
            echo "  --range          算子范围，如 41-53 (与 --ids 二选一)"
            echo "  --ids            指定算子编号列表，逗号分隔，如 3,7,15 (与 --range 二选一)"
            echo "  --npu            单 NPU 设备 ID，如 0 (默认 0，与 --npu-list 互斥)"
            echo "  --npu-list       多 NPU 列表，逗号分隔，如 0,1,2,3,4,5 (与 --npu 互斥，优先级更高)"
            echo "  --output         输出目录 (必填)"
            echo "  --arch           目标设备架构，默认 ascend910b2"
            echo "  --claude-project-dir  Claude Code 项目目录，用于定位 .jsonl 会话文件"
            echo ""
            echo "示例:"
            echo "  # 单 NPU 串行模式"
            echo "  bash utils/run_benchmark_triton.sh --benchmark-dir /path/to/KernelBench --level 1 --range 1-30 --npu 0 --output /path/to/output"
            echo ""
            echo "  # 多 NPU 并行模式"
            echo "  bash utils/run_benchmark_triton.sh --benchmark-dir /path/to/KernelBench --level 1 --range 1-30 --npu-list \"0,1,2,3,4,5\" --output /path/to/output"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 参数校验 ──
if [[ -z "$BENCHMARK_DIR" ]]; then
    echo "错误: 必须指定 --benchmark-dir"
    exit 1
fi

if [[ -z "$LEVEL" ]]; then
    echo "错误: 必须指定 --level"
    exit 1
fi

if [[ -z "$RANGE" && -z "$IDS" ]]; then
    echo "错误: 必须指定 --range 或 --ids"
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "错误: 必须指定 --output"
    exit 1
fi

LEVEL_DIR="${BENCHMARK_DIR}/level${LEVEL}"
if [[ ! -d "$LEVEL_DIR" ]]; then
    echo "错误: 目录不存在: ${LEVEL_DIR}"
    exit 1
fi

# ── 确定执行模式 ──
USE_PARALLEL=false
if [[ -n "$NPU_LIST" ]]; then
    USE_PARALLEL=true
    # 解析 NPU 列表
    IFS=',' read -ra NPU_ARRAY <<< "$NPU_LIST"
    NPU_COUNT=${#NPU_ARRAY[@]}
    if [[ $NPU_COUNT -eq 0 ]]; then
        echo "错误: NPU 列表为空"
        exit 1
    fi
else
    # 单 NPU 模式
    NPU_ARRAY=("$NPU_ID")
    NPU_COUNT=1
fi

# ── 构建算子 ID 列表 ──
OP_IDS=()
if [[ -n "$RANGE" ]]; then
    START=$(echo "$RANGE" | cut -d'-' -f1)
    END=$(echo "$RANGE" | cut -d'-' -f2)
    for i in $(seq "$START" "$END"); do
        OP_IDS+=("$i")
    done
elif [[ -n "$IDS" ]]; then
    IFS=',' read -ra OP_IDS <<< "$IDS"
fi

# ── 扫描算子文件 ──
declare -A OP_FILES
for id in "${OP_IDS[@]}"; do
    # 匹配 {id}_{name}.py 格式
    matched=$(find "$LEVEL_DIR" -maxdepth 1 -name "${id}_*.py" -type f 2>/dev/null | head -1)
    if [[ -n "$matched" ]]; then
        OP_FILES[$id]="$matched"
    else
        echo "警告: 未找到算子 ${id} 的文件，跳过"
    fi
done

if [[ ${#OP_FILES[@]} -eq 0 ]]; then
    echo "错误: 未找到任何算子文件"
    exit 1
fi

# ── 创建输出目录 ──
mkdir -p "$OUTPUT_DIR"

# ── 创建文件锁 ──
touch "${OUTPUT_DIR}/.lock"

# ── 结果记录 ──
REPORT_FILE="${OUTPUT_DIR}/batch_report.md"
echo "# 批量执行报告" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "- benchmark: ${BENCHMARK_DIR}" >> "$REPORT_FILE"
echo "- level: ${LEVEL}" >> "$REPORT_FILE"
echo "- arch: ${ARCH}" >> "$REPORT_FILE"
if [[ "$USE_PARALLEL" == true ]]; then
    echo "- npu-list: ${NPU_LIST}" >> "$REPORT_FILE"
    echo "- 执行模式: 多 NPU 并行（NPU 间并行，NPU 内串行）" >> "$REPORT_FILE"
else
    echo "- npu: ${NPU_ID}" >> "$REPORT_FILE"
    echo "- 执行模式: 单 NPU 串行" >> "$REPORT_FILE"
fi
echo "- 开始时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| 算子ID | 文件 | 状态 | 耗时(s) |" >> "$REPORT_FILE"
echo "|--------|------|------|---------|" >> "$REPORT_FILE"

TOTAL=${#OP_FILES[@]}
SUCCESS=0
FAIL=0

# ── 执行模式选择 ──
if [[ "$USE_PARALLEL" == true ]]; then
    # ========== 多 NPU 并行模式 ==========
    echo ""
    echo "================================================================"
    echo "多 NPU 并行模式: ${NPU_COUNT} 个 NPU，${TOTAL} 个算子"
    echo "NPU 列表: ${NPU_LIST}"
    echo "================================================================"
    echo ""

    # 任务分配：轮询分配算子到各 NPU 队列
    declare -A npu_tasks
    npu_index=0
    for id in "${OP_IDS[@]}"; do
        if [[ -v OP_FILES[$id] ]]; then
            npu=${NPU_ARRAY[$((npu_index % NPU_COUNT))]}
            npu_tasks[$npu]+="${id} "
            npu_index=$((npu_index + 1))
        fi
    done

    # 为每个 NPU 启动 worker 进程
    for npu in "${NPU_ARRAY[@]}"; do
        # 检查该 NPU 是否有任务
        if [[ -n "${npu_tasks[$npu]:-}" ]]; then
            (
                # ========== Worker 进程开始 ==========
                # 若外部未传入，则自动推导 CLAUDE_PROJECT_DIR
                # Claude Code 项目目录命名规则：将绝对路径的 / 替换为 -，_ 替换为 -
                if [[ -z "${CLAUDE_PROJECT_DIR:-}" ]]; then
                    _PWD_DASH="$(pwd | sed 's|/|-|g; s|_|-|g')"
                    CLAUDE_PROJECT_DIR="/root/.claude/projects/${_PWD_DASH}"
                    echo "[NPU $npu] 自动推导 CLAUDE_PROJECT_DIR: ${CLAUDE_PROJECT_DIR}" >&2
                else
                    echo "[NPU $npu] 使用传入的 CLAUDE_PROJECT_DIR: ${CLAUDE_PROJECT_DIR}" >&2
                fi
                for id in ${npu_tasks[$npu]}; do
                    file="${OP_FILES[$id]}"
                    filename=$(basename "$file")
                    op_name="${filename%.*}"
                    TARGET_OP_DIR="${OUTPUT_DIR}/${op_name}"
                    json_file="${file%.py}.json"

                    mkdir -p "$TARGET_OP_DIR"

                    # 预生成 session-id，调用后按 SID 精确取 jsonl，避免 ls -t 竞态
                    SID=$(python3 -c 'import uuid;print(uuid.uuid4())')

                    START_TIME=$(date +%s)

                    if [[ -f "$json_file" ]]; then
                        PROMPT="生成一个基于 Triton-Ascend 框架的算子，参考${file}和${json_file}。目标设备架构为${ARCH}，使用NPU=${npu}，请将生成的代码文件输出至${TARGET_OP_DIR}/目录下。"
                    else
                        PROMPT="生成一个基于 Triton-Ascend 框架的算子，参考${file}。目标设备架构为${ARCH}，使用NPU=${npu}，请将生成的代码文件输出至${TARGET_OP_DIR}/目录下。"
                    fi

                    if claude -p "$PROMPT" \
                        --session-id "$SID" \
                        --allowedTools 'Bash(*)' 'Read(*)' 'Write(*)' 'Edit(*)' 'Glob(*)' 'Grep(*)' 'Skill(*)' \
                        >> "${OUTPUT_DIR}/npu_${npu}.log" 2>&1; then

                        END_TIME=$(date +%s)
                        ELAPSED=$((END_TIME - START_TIME))

                        # 立即输出到主终端
                        echo "[NPU $npu] ✅ 算子 ${id}: ${filename} 完成 (${ELAPSED}s)" >&2

                        # 加锁写入报告
                        {
                            flock -x 200
                            echo "| ${id} | ${filename} | ✅ 成功 | ${ELAPSED} |" >> "$REPORT_FILE"
                        } 200>"${OUTPUT_DIR}/.lock"

                        STATUS="success"
                    else
                        END_TIME=$(date +%s)
                        ELAPSED=$((END_TIME - START_TIME))

                        # 立即输出到主终端
                        echo "[NPU $npu] ❌ 算子 ${id}: ${filename} 失败 (${ELAPSED}s)" >&2

                        # 加锁写入报告
                        {
                            flock -x 200
                            echo "| ${id} | ${filename} | ❌ 失败 | ${ELAPSED} |" >> "$REPORT_FILE"
                        } 200>"${OUTPUT_DIR}/.lock"

                        STATUS="fail"
                    fi

                    # 按 session-id 精确搬运思维轨迹（无需 flock，无竞态）
                    SRC_JSONL="${CLAUDE_PROJECT_DIR}/${SID}.jsonl"
                    if [[ -f "$SRC_JSONL" ]]; then
                        mv "$SRC_JSONL" "${TARGET_OP_DIR}/session.jsonl"
                    else
                        echo "[NPU $npu] ⚠ 未找到 session jsonl: ${SRC_JSONL}" >&2
                    fi
                    if [[ -d "${CLAUDE_PROJECT_DIR}/${SID}" ]]; then
                        mv "${CLAUDE_PROJECT_DIR}/${SID}" "${TARGET_OP_DIR}/session_dir"
                    fi
                done
                # ========== Worker 进程结束 ==========
            ) &
        fi
    done

    # 等待所有 worker 完成
    wait

else
    # ========== 单 NPU 串行模式（原逻辑）==========
    echo ""
    echo "================================================================"
    echo "单 NPU 串行模式: NPU ${NPU_ID}，${TOTAL} 个算子"
    echo "================================================================"
    echo ""

    # 若外部未传入，则自动推导 CLAUDE_PROJECT_DIR
    # Claude Code 项目目录命名规则：将绝对路径的 / 替换为 -，_ 替换为 -
    if [[ -z "${CLAUDE_PROJECT_DIR:-}" ]]; then
        _PWD_DASH="$(pwd | sed 's|/|-|g; s|_|-|g')"
        CLAUDE_PROJECT_DIR="/root/.claude/projects/${_PWD_DASH}"
        echo "[NPU ${NPU_ID}] 自动推导 CLAUDE_PROJECT_DIR: ${CLAUDE_PROJECT_DIR}"
    else
        echo "[NPU ${NPU_ID}] 使用传入的 CLAUDE_PROJECT_DIR: ${CLAUDE_PROJECT_DIR}"
    fi

    CURRENT=0
    for id in $(echo "${!OP_FILES[@]}" | tr ' ' '\n' | sort -n); do
        file="${OP_FILES[$id]}"
        filename=$(basename "$file")
        op_name="${filename%.*}"
        TARGET_OP_DIR="${OUTPUT_DIR}/${op_name}"
        json_file="${file%.py}.json"

        mkdir -p "$TARGET_OP_DIR"

        CURRENT=$((CURRENT + 1))

        echo ""
        echo "================================================================"
        echo "[${CURRENT}/${TOTAL}] 算子 ${id}: ${filename} (输出至: ${op_name}/)"
        echo "================================================================"

        START_TIME=$(date +%s)

        # 预生成 session-id，调用后按 SID 精确取 jsonl
        SID=$(python3 -c 'import uuid;print(uuid.uuid4())')

        if [[ -f "$json_file" ]]; then
            PROMPT="生成一个基于 Triton-Ascend 框架的算子，参考${file}和${json_file}。目标设备架构为${ARCH}，使用NPU=${NPU_ID}，请将生成的代码文件输出至${TARGET_OP_DIR}/目录下。"
        else
            PROMPT="生成一个基于 Triton-Ascend 框架的算子，参考${file}。目标设备架构为${ARCH}，使用NPU=${NPU_ID}，请将生成的代码文件输出至${TARGET_OP_DIR}/目录下。"
        fi

        if claude -p "$PROMPT" \
            --session-id "$SID" \
            --allowedTools 'Bash(*)' 'Read(*)' 'Write(*)' 'Edit(*)' 'Glob(*)' 'Grep(*)' 'Skill(*)'; then
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            echo "| ${id} | ${filename} | ✅ 成功 | ${ELAPSED} |" >> "$REPORT_FILE"
            SUCCESS=$((SUCCESS + 1))
            echo "[${CURRENT}/${TOTAL}] ✅ 算子 ${id} 完成 (${ELAPSED}s)"
            STATUS="success"
        else
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            echo "| ${id} | ${filename} | ❌ 失败 | ${ELAPSED} |" >> "$REPORT_FILE"
            FAIL=$((FAIL + 1))
            echo "[${CURRENT}/${TOTAL}] ❌ 算子 ${id} 失败 (${ELAPSED}s)"
            STATUS="fail"
        fi

        # 按 session-id 精确搬运思维轨迹
        SRC_JSONL="${CLAUDE_PROJECT_DIR}/${SID}.jsonl"
        if [[ -f "$SRC_JSONL" ]]; then
            mv "$SRC_JSONL" "${TARGET_OP_DIR}/session.jsonl"
        else
            echo "[NPU ${NPU_ID}] ⚠ 未找到 session jsonl: ${SRC_JSONL}"
        fi
        if [[ -d "${CLAUDE_PROJECT_DIR}/${SID}" ]]; then
            mv "${CLAUDE_PROJECT_DIR}/${SID}" "${TARGET_OP_DIR}/session_dir"
        fi
    done
fi

# ── 写入汇总 ──
echo "" >> "$REPORT_FILE"
echo "## 汇总" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# 统计成功和失败数
SUCCESS=$(grep -c "✅ 成功" "$REPORT_FILE" 2>/dev/null || echo 0)
FAIL=$(grep -c "❌ 失败" "$REPORT_FILE" 2>/dev/null || echo 0)

echo "- 总数: ${TOTAL}" >> "$REPORT_FILE"
echo "- 成功: ${SUCCESS}" >> "$REPORT_FILE"
echo "- 失败: ${FAIL}" >> "$REPORT_FILE"
echo "- 结束时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"

if [[ "$USE_PARALLEL" == true ]]; then
    echo "- 执行模式: 多 NPU 并行" >> "$REPORT_FILE"
    echo "- NPU 日志: npu_0.log, npu_1.log, ... (在输出目录中)" >> "$REPORT_FILE"
fi

echo ""
echo "================================================================"
echo "批量执行完成: 成功 ${SUCCESS}/${TOTAL}, 失败 ${FAIL}/${TOTAL}"
echo "报告: ${REPORT_FILE}"
if [[ "$USE_PARALLEL" == true ]]; then
    echo "NPU 日志目录: ${OUTPUT_DIR}/"
fi
echo "================================================================"


