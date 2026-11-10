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
# preflight.sh [--require-geir] <ascend_env_path> [repo] [op_class] [OP] [target_repo]
#
# 校验任务给定/工作流已验证的环境，不替用户挑版本。
#
# 用法: bash scripts/preflight.sh [--require-geir] /path/to/Ascend [<repo> <op_class> <OP> <target_repo>]
# 退出码: 0 = 无 FAIL；1 = 有 FAIL；2 = 用法错误。
# -----------------------------------------------------------------------------------------------------------
set -u

REQUIRE_GEIR=0
if [ "${1:-}" = "--require-geir" ]; then
    REQUIRE_GEIR=1
    shift
fi
ASCEND="${1:-}"
if [ "$#" -gt 5 ] || [ -z "$ASCEND" ] || [ ! -d "$ASCEND" ]; then
    echo "usage: $0 [--require-geir] <ascend_env_path> [repo] [op_class] [OP] [target_repo]"
    exit 2
fi
REPO="${2:-}"
OP_CLASS="${3:-}"
OP="${4:-}"
TARGET_REPO="${5:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

fail=0; warn=0
say_ok()   { echo "[ OK ] $*"; }
say_warn() { echo "[WARN] $*"; warn=$((warn + 1)); }
say_fail() { echo "[FAIL] $*"; fail=$((fail + 1)); }

echo "== ops-repo migration preflight =="
echo "ascend env: $ASCEND"
echo

# --- 1) Ascend 环境可用性 ---------------------------------------------------------
# set_env.sh 可能在根目录、也可能在 ascend-toolkit/ 或具体版本目录下
setenv=""
for c in "$ASCEND/set_env.sh" "$ASCEND/ascend-toolkit/set_env.sh"; do
    [ -f "$c" ] && setenv="$c" && break
done
if [ -z "$setenv" ]; then
    say_fail "no set_env.sh at the specified toolkit path — provide the exact previously verified environment"
else
    say_ok "set_env.sh: $setenv"
    # 归一环境后再校验（清掉可能叠加的多版本路径）
    unset LD_LIBRARY_PATH
    # shellcheck disable=SC1090
    set +u
    if . "$setenv" >/dev/null 2>&1; then
        say_ok "set_env.sh loaded"
    else
        say_fail "set_env.sh failed"
    fi
    set -u
    echo "       ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-}"
    echo "       ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-}"
fi

# --- 2) graph example 的前置条件：内置 op_proto 头 ---------------------------------
# 缺了它，所有 `--run_example {OP} graph` 都会 fatal error: array_ops.h: No such file
if [ -n "${ASCEND_OPP_PATH:-}" ] && [ -f "${ASCEND_OPP_PATH}/built-in/op_proto/inc/array_ops.h" ]; then
    say_ok "opp/built-in/op_proto/inc/array_ops.h present — run_example graph 可编译"
elif [ "$REQUIRE_GEIR" -eq 1 ]; then
    say_fail "缺 \${ASCEND_OPP_PATH}/built-in/op_proto/inc/array_ops.h — report the verified environment's missing GEIR prerequisite"
else
    say_warn "GEIR prerequisite absent; no GEIR route requested (use --require-geir when applicable)"
fi

# --- 3) pre-commit 门禁（收尾 commit 时生效）---------------------------------------
if command -v pre-commit >/dev/null 2>&1; then
    say_ok "pre-commit $(pre-commit --version 2>/dev/null | awk '{print $2}') 可用"
else
    say_warn "pre-commit 不在 PATH —— 收尾 commit 前需自行安装，否则门禁（clang-format / OAT 许可证）不生效"
fi

if [ -n "$REPO" ] && [ ! -d "$REPO" ]; then
    say_fail "repo does not exist: $REPO"
elif [ -n "$REPO" ]; then
    TARGET_REPO="${TARGET_REPO:-$(basename "$REPO")}"
    if python3 - "$SCRIPT_DIR" "$TARGET_REPO" "$OP_CLASS" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from migration_contract import profile
try:
    config = profile(sys.argv[2])
    if sys.argv[3] and sys.argv[3] not in config['classes']:
        raise ValueError('unsupported op_class: ' + sys.argv[3])
except ValueError as exc:
    print(exc, file=sys.stderr)
    sys.exit(1)
PY
    then
        say_ok "target profile/class: $TARGET_REPO/$OP_CLASS"
    else
        say_fail "invalid repo/class contract"
    fi
    if [ -n "$OP_CLASS" ] && [ ! -d "$REPO/$OP_CLASS" ]; then
        say_fail "classification directory does not exist: $OP_CLASS"
    fi
    # --- 4) 目标算子目录是否已被占用 -----------------------------------------------
    if [ -n "$OP_CLASS" ] && [ -n "$OP" ]; then
        if [ -e "$REPO/$OP_CLASS/$OP" ]; then
            say_fail "target already exists: $REPO/$OP_CLASS/$OP — 先确认是否要覆盖同名算子"
        else
            say_ok "target free: $OP_CLASS/$OP"
        fi
    fi

    # --- 5) git 工作区是否干净（收尾要 commit，别把无关改动带进去）------------------
    if git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        if command -v git >/dev/null 2>&1; then
            if ! dirty="$(git -C "$REPO" status --porcelain 2>&1)"; then
                say_fail "git status failed — 无法确认工作区状态："
                printf '%s\n' "$dirty" | head -5 | sed 's/^/       /'
            elif [ -n "$dirty" ]; then
                say_warn "git 工作区不干净（保留无关改动，勿纳入迁移提交）："
                printf '%s\n' "$dirty" | head -5 | sed 's/^/       /'
            else
                say_ok "git 工作区干净"
            fi
        fi
        hook_path="$(git -C "$REPO" rev-parse --path-format=absolute --git-path hooks/pre-commit)"
        if [ -f "$hook_path" ]; then
            say_ok ".git/hooks/pre-commit 已安装"
        else
            say_warn ".git/hooks/pre-commit 未安装 —— 到 <repo> 下执行 pre-commit install"
        fi
    else
        say_fail "target is not a Git checkout"
    fi
fi

echo "----"
echo "FAIL=$fail WARN=$warn"
[ "$fail" -eq 0 ] || exit 1
