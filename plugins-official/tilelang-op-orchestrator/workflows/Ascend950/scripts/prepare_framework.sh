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
if [ "$#" -ne 0 ]; then
    echo "Usage: prepare_framework.sh (source directories are fixed under the plugin)" >&2
    exit 2
fi
source_root="$plugin_dir/repositories/Ascend950"
ops_tilelang_dir="$source_root/ops-tilelang"
tilelang_dir="$source_root/tilelang"
log_dir="$plugin_dir/.preparation/Ascend950"
mkdir -p "$log_dir"
exec 9>"$log_dir/prepare.lock"
if ! flock -n 9; then
    echo "Ascend950 preparation is already running for $source_root; logs: $log_dir" >&2
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
        echo "[Ascend950] 准备未完成：$stage；退出码 $code；已耗时 $((SECONDS-started)) 秒"
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
    echo "[Ascend950] $stage；日志：$log_file"
    # A process group lets cancellation stop Git and its child processes.
    setsid "$@" 9>&- &
    active_pid=$!
    local last_progress=$SECONDS
    while kill -0 "$active_pid" 2>/dev/null; do
        sleep 1
        if (( SECONDS-last_progress >= 15 )) && kill -0 "$active_pid" 2>/dev/null; then
            echo "[Ascend950] $stage进行中，阶段耗时 $((SECONDS-step_started)) 秒，总耗时 $((SECONDS-started)) 秒"
            last_progress=$SECONDS
        fi
    done
    wait "$active_pid" || code=$?
    if [ "$code" -ne 0 ]; then
        kill -TERM -- "-$active_pid" 2>/dev/null || true
    fi
    active_pid=""
    echo "[Ascend950] $stage结束，退出码 $code，阶段耗时 $((SECONDS-step_started)) 秒"
    return "$code"
}
prepare_repo() {
    local repo_name="$1" repo_url="$2" repo_dir="$3"
    stage="检查 $repo_name 源码"
    echo "[Ascend950] $repo_name 源码目录：$repo_dir"
    if [ -e "$repo_dir" ] || [ -L "$repo_dir" ]; then
        if [ ! -e "$repo_dir/.git" ] || ! git -C "$repo_dir" rev-parse --verify HEAD >/dev/null 2>&1; then
            echo "目标目录不是有效源码仓库，不覆盖其中内容：$repo_dir" >&2
            exit 1
        fi
        echo "[Ascend950] $repo_name 源码已存在，跳过克隆；不自动 pull 或重置用户修改"
    else
        mkdir -p "$(dirname "$repo_dir")"
        clone_tmp=$(mktemp -d "${repo_dir}.clone.XXXXXX")
        # Reference sources only; fetch individual submodules when their code is needed.
        run_step "浅克隆 $repo_name 主仓库（不下载子模块）" git clone --progress --depth=1 --no-recurse-submodules "$repo_url" "$clone_tmp"
        mv -T -- "$clone_tmp" "$repo_dir"
        clone_tmp=""
    fi
}

echo "[Ascend950] 完整日志：$log_file"
prepare_repo "ops-tilelang" "https://gitcode.com/cann/ops-tilelang.git" "$ops_tilelang_dir"
prepare_repo "tilelang" "https://github.com/tile-ai/tilelang.git" "$tilelang_dir"

echo "[Ascend950] SOURCE_READY：两个主仓库源码已就绪；子模块按需获取，总耗时 $((SECONDS-started)) 秒"
printf 'OPS_TILELANG_DIR=%q\n' "$ops_tilelang_dir"
printf 'TILELANG_DIR=%q\n' "$tilelang_dir"
