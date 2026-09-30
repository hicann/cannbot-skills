#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
plugin_dir="$(cd "$script_dir/../../.." && pwd)"
if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [[ "$1" != /* ]]; }; then
    echo "Usage: prepare_framework.sh [absolute source directory]" >&2
    exit 2
fi
repo_dir="${1:-${TILELANG_DIR:-$plugin_dir/repositories/Ascend910/tilelang-ascend}}"
repo_dir=$(realpath -m -- "$repo_dir")
repo_key=$(printf '%s' "$repo_dir" | sha256sum | cut -d ' ' -f 1)
log_dir="$plugin_dir/.preparation/Ascend910/$repo_key"
mkdir -p "$log_dir"
exec 9>"$log_dir/prepare.lock"
if ! flock -n 9; then
    echo "Ascend910 preparation is already running for $repo_dir; logs: $log_dir" >&2
    exit 1
fi
log_file=$(mktemp "$log_dir/$(date +%Y%m%d-%H%M%S).XXXXXX.log")
exec > >(tee -a "$log_file") 2>&1
started=$SECONDS
stage="检查源码"
active_pid=""
clone_tmp=""
cleanup() {
    local code=$?
    trap - EXIT INT TERM
    if [ -n "$active_pid" ]; then
        kill -TERM -- "-$active_pid" 2>/dev/null || true
        wait "$active_pid" 2>/dev/null || true
    fi
    [ -z "$clone_tmp" ] || rm -rf -- "$clone_tmp"
    if [ "$code" -ne 0 ]; then
        echo "[Ascend910] 准备未完成：$stage；退出码 $code；已耗时 $((SECONDS-started)) 秒"
        echo "日志：$log_file"
    fi
    exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_step() {
    stage="$1"; shift
    local step_started=$SECONDS code=0
    echo "[Ascend910] $stage；日志：$log_file"
    # A process group lets cancellation stop the build and its child processes.
    setsid "$@" 9>&- &
    active_pid=$!
    local last_progress=$SECONDS
    while kill -0 "$active_pid" 2>/dev/null; do
        sleep 1
        if (( SECONDS-last_progress >= 15 )) && kill -0 "$active_pid" 2>/dev/null; then
            echo "[Ascend910] $stage进行中，阶段耗时 $((SECONDS-step_started)) 秒，总耗时 $((SECONDS-started)) 秒"
            last_progress=$SECONDS
        fi
    done
    wait "$active_pid" || code=$?
    if [ "$code" -ne 0 ]; then
        kill -TERM -- "-$active_pid" 2>/dev/null || true
    fi
    active_pid=""
    echo "[Ascend910] $stage结束，退出码 $code，阶段耗时 $((SECONDS-step_started)) 秒"
    last_step_code="$code"
    return "$code"
}
echo "[Ascend910] 源码目录：$repo_dir"
echo "[Ascend910] 完整日志：$log_file"
if [ -e "$repo_dir" ] || [ -L "$repo_dir" ]; then
    if [ ! -e "$repo_dir/.git" ] || ! git -C "$repo_dir" rev-parse --verify HEAD >/dev/null 2>&1; then
        echo "目标目录不是有效源码仓库，不覆盖其中内容：$repo_dir" >&2
        exit 1
    fi
    echo "[Ascend910] 源码已存在，跳过克隆；不自动 pull 或重置用户修改"
else
    mkdir -p "$(dirname "$repo_dir")"
    clone_tmp=$(mktemp -d "${repo_dir}.clone.XXXXXX")
    run_step "克隆源码及子模块" git clone --progress --recursive --depth=1 https://github.com/tile-ai/tilelang-ascend.git "$clone_tmp"
    mv -T -- "$clone_tmp" "$repo_dir"
    clone_tmp=""
fi

# Existing checkouts can have uninitialized submodules. Never reset a user's
# selected submodule revision while filling in missing dependencies.
[ -f "$repo_dir/install_ascend.sh" ] || { echo "缺少 install_ascend.sh：$repo_dir"; exit 1; }
submodules_changed=0
submodules=$(git -C "$repo_dir" submodule status --recursive)
if [[ "$submodules" == *$'\nU'* ]] || [[ "$submodules" == U* ]]; then
    echo "子模块存在合并冲突，请先解决冲突。"; exit 1
fi
if [[ "$submodules" == *$'\n-'* ]] || [[ "$submodules" == -* ]]; then
    if [[ "$submodules" == *$'\n+'* ]] || [[ "$submodules" == +* ]]; then
        echo "子模块同时存在未初始化项和自定义版本；请手动补全，避免覆盖版本选择。"; exit 1
    fi
    run_step "初始化子模块" git -C "$repo_dir" submodule update --init --recursive --progress
    submodules_changed=1
fi
# A successful log alone is insufficient: check real artifacts and the current
# Python import path. A failed/interrupted build must be retried.
state_file="$log_dir/build-state"
previous_state=$(cat "$state_file" 2>/dev/null || true)
verify_environment() {
    run_step "检查编译产物与 Python 环境" timeout 90 bash -c '
        set -eo pipefail
        cd "$1"
        [ -f set_env.sh ] || exit 1
        compgen -G "build/*.so" >/dev/null || compgen -G "build/lib/*.so" >/dev/null || exit 1
        source ./set_env.sh
        python3 - "$1" <<"CHECK"
import pathlib, sys
import tilelang
root = pathlib.Path(sys.argv[1]).resolve()
loaded = pathlib.Path(tilelang.__file__).resolve()
if root not in loaded.parents:
    raise SystemExit(f"TileLang 导入路径不属于目标源码：{loaded}")
print(f"Python: {sys.executable}")
print(f"TileLang: {loaded}")
CHECK
    ' preparation-check "$repo_dir"
}
stage="检查编译产物与 Python 环境"
if [ "$submodules_changed" -eq 0 ] && [ "$previous_state" != building ] && [ "$previous_state" != failed ] && verify_environment; then
    echo "[Ascend910] 编译产物与 Python 环境有效，跳过编译"
else
    printf 'building\n' > "$state_file"
    if run_step "编译安装" bash -c 'cd "$1" && bash install_ascend.sh' preparation-build "$repo_dir"; then
        stage="验证编译产物与 Python 环境"
        if ! verify_environment; then
            printf 'failed\n' > "$state_file"
            echo "安装脚本已退出，但环境验证未通过。"; exit "$last_step_code"
        fi
    else
        printf 'failed\n' > "$state_file"
        exit "$last_step_code"
    fi
fi
printf 'verified\n' > "$state_file"
printf '[Ascend910] 源码和编译环境已就绪，总耗时 %s 秒\n' "$((SECONDS-started))"
printf 'TILELANG_DIR=%q\n' "$repo_dir"
echo "后续由环境预检技能验证 CANN、Python 依赖和 NPU；此处成功不代表上板验证通过。"
