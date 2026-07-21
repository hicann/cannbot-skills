# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
FRAMEWORK_DIR="$REPO_ROOT/tests/system"
TARGET_BRANCH="${CI_MERGE_REQUEST_TARGET_BRANCH_NAME:-${BASE_BRANCH:-master}}"

# =========================================================================
# Phase 0: 平台参数解析
# =========================================================================
# 读取 --ascend-platform 和 ASCEND_PLATFORM 环境变量

ASCEND_PLATFORMS=()
REPEAT_COUNT=1
ALL_MODE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ascend-platform)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "ERROR: --ascend-platform requires a value (A2/A3/A5)"
                exit 1
            fi
            ASCEND_PLATFORMS+=("$2")
            shift 2
            ;;
        --repeat)
            if [[ -z "$2" || ! "$2" =~ ^[0-9]+$ ]] || [ "$2" -lt 1 ]; then
                echo "ERROR: --repeat requires a positive integer"
                exit 1
            fi
            REPEAT_COUNT="$2"
            shift 2
            ;;
        --all)
            ALL_MODE=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 若未通过 CLI 指定，fallback 到环境变量
if [ ${#ASCEND_PLATFORMS[@]} -eq 0 ] && [ -n "${ASCEND_PLATFORM:-}" ]; then
    IFS=', ' read -ra ASCEND_PLATFORMS <<< "$ASCEND_PLATFORM"
fi

# 若均未配置平台参数，默认使用 A2（兼容未配置 CI 环境的场景）
if [ ${#ASCEND_PLATFORMS[@]} -eq 0 ]; then
    echo "未指定 --ascend-platform 且 ASCEND_PLATFORM 环境变量未设置，默认使用 A2。"
    ASCEND_PLATFORMS=("A2")
fi

# 校验每个平台值为 A2/A3/A5
for p in "${ASCEND_PLATFORMS[@]}"; do
    case "$p" in
        A2|A3|A5) ;;
        *)
            echo "ERROR: 无效的平台值 '$p'，请使用 A2/A3/A5"
            exit 1
            ;;
    esac
done

echo "目标平台: ${ASCEND_PLATFORMS[*]}"

# =========================================================================
# Phase 1: 环境初始化
# =========================================================================
echo "=== Phase 1: Environment Setup ==="
echo "Repository root: $REPO_ROOT"

# ---------- PEP 668 兼容 ----------
# Ubuntu 24.04+ 默认启用 PEP 668，禁止系统级 pip install（报 externally-managed-environment）。
# cann-bench 评测管道内部通过 pip install 安装 whl 包，需要此环境变量绕过限制。
# 设置为全局导出，确保所有子进程（包括 cann-bench 的 staged_eval.py）均继承。
export PIP_BREAK_SYSTEM_PACKAGES=1

# ---------- 加载 ASCEND 环境 ----------
# CI 通过 sshpass/SSH 非交互式执行时，~/.bashrc 不会被加载，
# 需要主动 source ASCEND 工具链环境变量。
ASCEND_ENV_LOADED=0
for env_script in \
    "$HOME/Ascend/ascend-toolkit/set_env.sh" \
    "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
    "$HOME/Ascend/cann/set_env.sh" \
    "/usr/local/Ascend/cann/set_env.sh" \
    "/home/developer/Ascend/ascend-toolkit/set_env.sh"; do
    if [ -f "$env_script" ]; then
        echo "  Sourcing ASCEND env: $env_script"
        # shellcheck disable=SC1090
        # set_env.sh 内部引用 LD_LIBRARY_PATH 等可能未定义的变量，
        # 临时关闭 nounset 避免 -u 模式下报 unbound variable 错误。
        set +u
        source "$env_script"
        set -u
        ASCEND_ENV_LOADED=1
        break
    fi
done
if [ "$ASCEND_ENV_LOADED" -eq 0 ]; then
    echo "  WARNING: No ASCEND set_env.sh found. ST evals requiring ASCEND may fail."
    echo "  Searched: ~/Ascend/ascend-toolkit/, /usr/local/Ascend/ascend-toolkit/, ~/Ascend/cann/, /usr/local/Ascend/cann/, /home/developer/Ascend/ascend-toolkit/"
fi

# ---------- Conda 环境激活 ----------
# CI 通过 sshpass/SSH 非交互式执行时，~/.bashrc 不会被加载，
# conda 环境不会被激活，导致使用系统 Python 而非 conda Python。
# 系统 Python 的 torch_npu 可能版本过旧（如 2.6.0），不支持新芯片（如 Ascend950PR 958b），
# 而 conda 环境中的 torch_npu 通常更新（如 2.11.0rc1），支持更多 SoC 版本。
# 此处主动探测并激活 conda，确保使用正确的 Python 环境。
CONDA_ACTIVATED=0
if command -v conda &> /dev/null; then
    echo "  conda already in PATH: $(conda --version 2>&1)"
    CONDA_ACTIVATED=1
else
    # conda 不在 PATH 中（非交互式 SSH 常见情况），尝试从常见安装路径初始化
    for conda_sh in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "/opt/miniconda3/etc/profile.d/conda.sh" \
        "/opt/anaconda3/etc/profile.d/conda.sh" \
        "/root/miniconda3/etc/profile.d/conda.sh" \
        "/root/anaconda3/etc/profile.d/conda.sh"; do
        if [ -f "$conda_sh" ]; then
            echo "  Found conda init script: $conda_sh"
            # shellcheck disable=SC1090
            source "$conda_sh"
            if command -v conda &> /dev/null; then
                conda activate base 2>/dev/null || true
                echo "  conda activated: $(conda --version 2>&1)"
                CONDA_ACTIVATED=1
                break
            fi
        fi
    done
fi
if [ "$CONDA_ACTIVATED" -eq 0 ]; then
    echo "  WARNING: conda not found. Using system Python (torch_npu may be outdated)."
fi

# ---------- Python 兼容性检查 ----------
# cann-bench 的 run_evaluation.sh 使用 "python" 命令（非 python3），
# 部分系统仅有 python3 而无 python 符号链接，会导致评测脚本直接失败。
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        PYTHON3_PATH="$(command -v python3)"
        # 在临时 bin 目录创建 python → python3 符号链接并加入 PATH
        LOCAL_BIN="${REPO_ROOT}/.local_bin"
        mkdir -p "$LOCAL_BIN"
        ln -sf "$PYTHON3_PATH" "${LOCAL_BIN}/python"
        export PATH="${LOCAL_BIN}:${PATH}"
        echo "  python not found, created symlink: ${LOCAL_BIN}/python -> ${PYTHON3_PATH}"
    else
        echo "ERROR: Neither python nor python3 found in PATH."
        echo "cann-bench 评测需要 Python，请安装后再运行 gate_check。"
        exit 1
    fi
fi
echo "  python: $(python --version 2>&1)"
echo "  python3: $(python3 --version 2>&1)"
echo "  python path: $(command -v python 2>/dev/null || echo 'not found')"
echo "  python3 path: $(command -v python3 2>/dev/null || echo 'not found')"
# 打印 torch_npu 版本（cann-bench 评测依赖，SoC 兼容性由其决定）
TORCH_NPU_VER=$(python3 -c "import importlib.metadata; print(importlib.metadata.version('torch_npu'))" 2>/dev/null || echo "not installed")
echo "  torch_npu: ${TORCH_NPU_VER}"

# 预检 opencode CLI（Phase 2 AI 语义评测的前置依赖）
if ! command -v opencode &> /dev/null; then
    echo "ERROR: opencode CLI not found in PATH."
    echo "Phase 2 (AI 语义评测) 需要 opencode，请安装后再运行 gate_check。"
    exit 1
fi
echo "  opencode: $(opencode --version 2>&1 | head -1)"

# =========================================================================
# Phase 2: 检测变更文件（--all 模式下跳过）
# =========================================================================
if [ "$ALL_MODE" -eq 1 ]; then
    echo "=== Phase 2: --all mode, skip change detection ==="
else
    echo "=== Phase 2: Detect Changed Files ==="
    # 优先从 pr_filelist.txt 读取（CI 流水线在项目根目录生成）
    PR_FILELIST="$REPO_ROOT/pr_filelist.txt"
    if [ -f "$PR_FILELIST" ]; then
        CHANGED_FILES=$(grep -v '^\s*$' "$PR_FILELIST" || true)
        echo "[from pr_filelist.txt]"
    fi

    if [ -z "${CHANGED_FILES:-}" ]; then
        git fetch origin "$TARGET_BRANCH" --depth=1 2>/dev/null || true
        CHANGED_FILES=$(git diff --name-only "origin/$TARGET_BRANCH"...HEAD 2>/dev/null || true)
    fi

    if [ -z "$CHANGED_FILES" ]; then
        echo "No changed files detected. Exiting."
        exit 0
    fi
    echo "Changed files:"
    echo "$CHANGED_FILES"

    readarray -t changed_files_array <<< "$CHANGED_FILES"
fi

# =========================================================================
# Phase 3: 安装依赖
# =========================================================================
echo "=== Phase 3: Install Dependencies ==="
pip install -r "$FRAMEWORK_DIR/scripts/requirements.txt" --quiet --break-system-packages

# =========================================================================
# Phase 3.5: cann-bench 自动获取（按需）
# =========================================================================
CANN_BENCH_DEFAULT="${REPO_ROOT}/../../cann-bench"
CANN_BENCH_TARGET="${CANN_BENCH_PATH:-$CANN_BENCH_DEFAULT}"

NEED_CANN_BENCH=false
if [ "$ALL_MODE" -eq 1 ]; then
    # --all 模式：扫描所有 skill/team 的 evals.json 文件
    for evals_file in "$REPO_ROOT"/ops/*/evals/evals.json \
                     "$REPO_ROOT"/graph/*/evals/evals.json \
                     "$REPO_ROOT"/model/*/evals/evals.json \
                     "$REPO_ROOT"/infra/*/evals/evals.json; do
        if [ -f "$evals_file" ] && grep -q "cann_bench" "$evals_file" 2>/dev/null; then
            NEED_CANN_BENCH=true
            break
        fi
    done
else
    # 增量模式：从变更文件提取实体名，检查对应 evals.json
    ENTITIES=()
    for f in "${changed_files_array[@]}"; do
        entity=$(echo "$f" | cut -d'/' -f2)
        [ -n "$entity" ] && ENTITIES+=("$entity")
    done
    if [ ${#ENTITIES[@]} -gt 0 ]; then
        ENTITIES=($(printf '%s\n' "${ENTITIES[@]}" | sort -u))
        for entity in "${ENTITIES[@]}"; do
            # 在 scan dirs 中查找对应实体的 evals.json
            for base_dir in "$REPO_ROOT"/ops/*/ \
                           "$REPO_ROOT"/graph/*/ \
                           "$REPO_ROOT"/model/*/ \
                           "$REPO_ROOT"/infra/*/ \
                           "$REPO_ROOT"/plugins-official/*/; do
                evals_file="${base_dir}evals/evals.json"
                if [ -f "$evals_file" ] && grep -q "cann_bench" "$evals_file" 2>/dev/null; then
                    NEED_CANN_BENCH=true
                    break 2
                fi
            done
        done
    fi
fi

echo "=== Phase 3.5: cann-bench Setup ==="
if [ "$NEED_CANN_BENCH" = true ] && [ ! -d "$CANN_BENCH_TARGET" ]; then
    echo "cann_bench mode detected, cloning cann-bench..."
    git clone --branch master --depth 1 \
        https://gitcode.com/cann/cann-bench.git "$CANN_BENCH_TARGET"
    echo "cann-bench cloned to: $CANN_BENCH_TARGET"
elif [ "$NEED_CANN_BENCH" = true ]; then
    echo "cann-bench already exists at $CANN_BENCH_TARGET"
else
    echo "cann-bench not needed, skipping"
fi

# =========================================================================
# Phase 4: 执行门禁检查
# =========================================================================
echo "=== Phase 4: Run Gate Check (${REPEAT_COUNT} iteration(s)) ==="

OVERALL_PASS=0
OVERALL_FAIL=0

for ((iter=1; iter<=REPEAT_COUNT; iter++)); do
    echo ""
    echo "--- Iteration ${iter}/${REPEAT_COUNT} ---"

    ITER_EXIT_CODE=0
    MAIN_ARGS=("--repo-root" "$REPO_ROOT" "--parallel" "auto")
    if [ "$ALL_MODE" -eq 1 ]; then
        MAIN_ARGS+=("--all")
    else
        MAIN_ARGS+=("--changed-files" "${changed_files_array[@]}")
    fi
    for p in "${ASCEND_PLATFORMS[@]}"; do
        MAIN_ARGS+=("--ascend-platform" "$p")
    done
    python3 "$FRAMEWORK_DIR/scripts/main.py" "${MAIN_ARGS[@]}" \
        || ITER_EXIT_CODE=$?

    if [ $ITER_EXIT_CODE -eq 0 ]; then
        echo "Iteration ${iter}: PASSED"
        OVERALL_PASS=$((OVERALL_PASS + 1))
    else
        echo "Iteration ${iter}: FAILED (exit code: $ITER_EXIT_CODE)"
        OVERALL_FAIL=$((OVERALL_FAIL + 1))
    fi
done

# 输出汇总
echo ""
echo "=== Repeat Summary ==="
echo "Total: ${REPEAT_COUNT}, Passed: ${OVERALL_PASS}, Failed: ${OVERALL_FAIL}"

if [ $OVERALL_FAIL -gt 0 ]; then
    echo "Gate check FAILED (${OVERALL_FAIL} iteration(s) failed)"
    exit 1
else
    echo "Gate check PASSED (all ${REPEAT_COUNT} iteration(s))"
    exit 0
fi