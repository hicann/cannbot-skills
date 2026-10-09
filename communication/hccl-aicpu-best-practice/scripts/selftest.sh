#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 本 Skill 的回归测试：实现脚本 + 两份骨架真编译 + 生成器故障注入。
#
#   bash scripts/selftest.sh <HCCL 仓根目录> [--with-aicpu-build] [--in-place] [--strict]
#   （仓路径也可用 export HCCL_REPO=... 给）
#
# 骨架编译需要往 HCCL 仓里真写文件（源文件 + 两处 CMake）。默认**不碰你的工作树**：
# 用 git worktree 在 /tmp 下开一份隔离副本，在副本里生成、编译，跑完 worktree remove。
#
#   --in-place        不开 worktree，直接在你的工作树里做。仅当两个 CMake 文件干净、
#                     且没有同名 probe 文件时才执行；恢复用的是**跑之前存下来的原始字节**，
#                     不调用 git checkout，因此不会吃掉任何未提交改动。
#   --with-aicpu-build 额外在隔离副本里跑一次 bash build.sh --aicpu，
#                     验证 device 侧 libscatter_aicpu_kernel.so 也能编（慢，需要 CANN 环境）。
#
#   --strict          前置条件缺失导致「没验成」的项（缺 compile_commands.json、开不了 worktree、
#                     工作树脏）也按失败返回非 0。CI 里建议加上。
#
# 默认的编译验证只覆盖 host 侧：compile_commands.json 里是 -DHOST_COMPILE -std=c++17。
# 编译产物写到 mktemp 出来的隔离目录，**不会落进你仓里的 build/**。
set -u
set -o pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HCCL=""; IN_PLACE=0; AICPU_BUILD=0; STRICT=0
for arg in "$@"; do
    case "$arg" in
        --in-place)         IN_PLACE=1 ;;
        --with-aicpu-build) AICPU_BUILD=1 ;;
        --strict)           STRICT=1 ;;
        -*) echo "未知参数: $arg" >&2; exit 2 ;;
        *)  HCCL="$arg" ;;
    esac
done
if [ "$IN_PLACE" -eq 1 ] && [ "$AICPU_BUILD" -eq 1 ]; then
    echo "--in-place 与 --with-aicpu-build 不能同时用：device 构建会在你的工作树里产出大量中间物，" >&2
    echo "与「device 构建只在隔离副本内执行」的约定冲突。去掉 --in-place 即可。" >&2
    exit 2
fi
HCCL="${HCCL:-${HCCL_REPO:-}}"
if [ -z "$HCCL" ]; then
    echo "用法: bash scripts/selftest.sh <HCCL 仓根目录> [--with-aicpu-build] [--in-place]" >&2
    exit 2
fi
HCCL="$(cd "$HCCL" 2>/dev/null && pwd)" || { echo "路径不存在: $HCCL" >&2; exit 2; }
[ -d "$HCCL/src/ops" ] || { echo "不像是 HCCL 仓: $HCCL" >&2; exit 2; }

OBJDIR="$(mktemp -d)"          # 编译产物只落在这里，绝不写进用户仓的 build/
PASS=0; FAIL=0; SKIP=0; HARD_SKIP=0
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
ok()   { echo "  [PASS] $1"; PASS=$((PASS + 1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }
# skip = 前置条件缺失导致「没验成」，--strict 下会让退出码非 0
# info = 本来就没打算跑的可选项（如未加 --with-aicpu-build），不影响退出码
skip() { echo "  [SKIP] $1"; SKIP=$((SKIP + 1)); HARD_SKIP=$((HARD_SKIP + 1)); }
info() { echo "  [INFO] $1"; }

# ---------------------------------------------------------------- 只读部分
step "规则登记与离线回归"
python3 -B "$SKILL_DIR/scripts/test_checks.py" && ok "离线回归" || bad "离线回归"
step "1/5 check_template --all：全仓不得有 ERROR"
python3 "$SKILL_DIR/scripts/check_template.py" --repo "$HCCL" --all >/dev/null \
    && ok "全仓无 ERROR" || bad "全仓存在 ERROR"

step "2/5 list_templates --blueprint：蓝本必须都能在仓里解析到"
if blueprints="$(python3 "$SKILL_DIR/scripts/list_templates.py" --repo "$HCCL" --blueprint 2>&1)"; then
    if [[ "$blueprints" == *"已不在仓内"* ]]; then
        bad "有蓝本在当前仓解析不到"
    else
        ok "蓝本全部命中"
    fi
else
    bad "蓝本扫描失败：$blueprints"
fi

step "3/5 缺仓路径时必须报错而不是猜"
if env -u HCCL_REPO python3 "$SKILL_DIR/scripts/list_templates.py" --blueprint >/dev/null 2>&1; then
    bad "没给 --repo 也没报错"
else
    ok "无 --repo/HCCL_REPO 时正确失败"
fi

# ---------------------------------------------------------------- 故障注入
step "4/5 生成器故障注入：CMake 接不上时必须非 0 退出且不落盘"
# 新旧两种仓布局各注入一次（2026-09 目录重构：template 迁入 <op>/algorithm/ 下）
inj_layout=0; inj_fail=0
for LAYOUT in "algorithm/template/aicpu" "template/aicpu"; do
    inj_layout=$((inj_layout + 1))
    FAKE="$(mktemp -d)"
    mkdir -p "$FAKE/src/ops/all_reduce/$LAYOUT" "$FAKE/src/common"
    printf 'set(src_list\n)\n' > "$FAKE/src/ops/all_reduce/$LAYOUT/CMakeLists.txt"
    printf 'enum class AlgoType : uint8_t {\n MESH,\n MESH_ONESHOT,\n NHR,\n UNKNOWN,\n};\n' > "$FAKE/src/common/alg_parse.h"
    # 故意不建 src/scatter_aicpu_kernel.cmake
    python3 "$SKILL_DIR/scripts/new_template.py" --repo "$FAKE" --op all_reduce \
            --class InsTempAllReduceMesh1DFoo --pattern all-reduce-mesh-oneshot >/dev/null 2>&1
    rc=$?
    produced="$(ls "$FAKE/src/ops/all_reduce/$LAYOUT/" | grep -c '^ins_temp_' || true)"
    if [ "$rc" -ne 0 ] && [ "$produced" -eq 0 ]; then
        ok "[$LAYOUT] 第二处 CMake 缺失时退出码 $rc 且未生成源文件"
    else
        bad "[$LAYOUT] 第二处 CMake 缺失：退出码=$rc，生成了 $produced 个源文件（应为 1/非0 和 0）"
        inj_fail=1
    fi
    rm -rf "$FAKE"
done

# ---------------------------------------------------------------- 骨架真编译
step "5/5 两份骨架：生成 → 用仓内真实编译参数编译（host 侧，含 -Werror）"

# 真仓 AICPU template 目录布局探测（新：algorithm/template/aicpu，旧：template/aicpu）
AICPU_REL="src/ops/all_reduce/algorithm/template/aicpu"
[ -d "$HCCL/$AICPU_REL" ] || AICPU_REL="src/ops/all_reduce/template/aicpu"
if [ ! -d "$HCCL/$AICPU_REL" ]; then
    echo "  真仓里找不到 all_reduce 的 aicpu template 目录，跳过本步" >&2
    skip "骨架编译：仓内无 aicpu template 目录"
fi

WORKDIR=""; WT=""; SAVED=""
CMAKE1="$AICPU_REL/CMakeLists.txt"
CMAKE2="src/scatter_aicpu_kernel.cmake"
PROBES="ins_temp_all_reduce_mesh_1D_probe ins_temp_all_reduce_nhr_probe"

cleanup() {
    if [ -n "$WT" ]; then
        git -C "$HCCL" worktree remove --force "$WT" >/dev/null 2>&1 || rm -rf "$WT"
    elif [ -n "$WORKDIR" ]; then
        # 就地模式：删掉自己生成的东西，并把两个 CMake 还原成跑之前存下的原始字节
        for b in $PROBES; do
            rm -f "$WORKDIR/$AICPU_REL/$b.h" "$WORKDIR/$AICPU_REL/$b.cc"
        done
        [ -n "$SAVED" ] && [ -f "$SAVED/1" ] && cp "$SAVED/1" "$WORKDIR/$CMAKE1"
        [ -n "$SAVED" ] && [ -f "$SAVED/2" ] && cp "$SAVED/2" "$WORKDIR/$CMAKE2"
        rm -rf "$SAVED"
    fi
    rm -rf "$OBJDIR"
}
trap cleanup EXIT

if [ "$IN_PLACE" -eq 0 ]; then
    if ! git -C "$HCCL" rev-parse --git-dir >/dev/null 2>&1; then
        skip "$HCCL 不是 git 仓，开不了隔离 worktree（要就地跑请加 --in-place）"
    else
        WT="$(mktemp -d)/hccl-selftest"
        if git -C "$HCCL" worktree add --detach "$WT" HEAD >/dev/null 2>&1; then
            WORKDIR="$WT"
            echo "  隔离 worktree: $WT（你的工作树完全不受影响）"
        else
            WT=""
            skip "git worktree 创建失败（要就地跑请加 --in-place）"
        fi
    fi
else
    # 就地模式的前置条件：两个 CMake 干净、没有同名 probe 文件
    dirty=0
    git -C "$HCCL" rev-parse --git-dir >/dev/null 2>&1 \
        && { git -C "$HCCL" diff --quiet HEAD -- "$CMAKE1" "$CMAKE2" || dirty=1; }
    exists=0
    for b in $PROBES; do
        [ -e "$HCCL/$AICPU_REL/$b.h" ] && exists=1
        [ -e "$HCCL/$AICPU_REL/$b.cc" ] && exists=1
    done
    if [ "$dirty" -ne 0 ]; then
        skip "两处 CMake 有未提交改动，就地模式拒绝执行（去掉 --in-place 用隔离 worktree）"
    elif [ "$exists" -ne 0 ]; then
        skip "仓里已存在同名 probe 文件，拒绝覆盖"
    else
        WORKDIR="$HCCL"
        SAVED="$(mktemp -d)"
        cp "$HCCL/$CMAKE1" "$SAVED/1"; cp "$HCCL/$CMAKE2" "$SAVED/2"
        echo "  就地模式：已备份两个 CMake 的原始字节，结束时按字节还原（不用 git checkout）"
    fi
fi

DATABASE="$HCCL/build/compile_commands.json"
[ -f "$DATABASE" ] || DATABASE="$HCCL/compile_commands.json"

if [ -n "$WORKDIR" ]; then
    for spec in "all-reduce-mesh-oneshot:InsTempAllReduceMesh1DProbe:ins_temp_all_reduce_mesh_1D_probe" \
                "barebone:InsTempAllReduceNhrProbe:ins_temp_all_reduce_nhr_probe"; do
        pattern="${spec%%:*}"; rest="${spec#*:}"; cls="${rest%%:*}"; base="${rest#*:}"
        if ! python3 "$SKILL_DIR/scripts/new_template.py" --repo "$WORKDIR" --op all_reduce \
                --class "$cls" --pattern "$pattern" --desc "selftest probe" >/dev/null; then
            bad "$pattern 生成失败"; continue
        fi
        out="$(python3 "$SKILL_DIR/scripts/compile_probe.py" \
            --repo "$HCCL" --worktree "$WORKDIR" --target "$WORKDIR/$AICPU_REL/$base.cc" \
            --output "$OBJDIR/$base.o" --database "$DATABASE")"; rc=$?
        case "$rc" in
            0) ok "$pattern host 侧编译通过" ;;
            3) skip "$pattern 无法编译验证：${out#SKIP }" ;;   # 拿不到编译数据库 = 跳过，不算通过
            *) echo "$out"; bad "$pattern 编译失败" ;;
        esac
        python3 "$SKILL_DIR/scripts/check_template.py" --strict-new --repo "$WORKDIR" \
                "$WORKDIR/$AICPU_REL/$base.cc" >/dev/null \
            && ok "$pattern 过 check_template" || bad "$pattern 未过 check_template"
    done

    if [ "$AICPU_BUILD" -eq 1 ]; then
        step "附加：device 侧 bash build.sh --aicpu（隔离副本内）"
        AICPU_LOG="$(mktemp "${TMPDIR:-/tmp}/hccl-selftest-aicpu.XXXXXX.log")"
        echo "  device 构建日志: $AICPU_LOG"
        if (cd "$WORKDIR" && bash build.sh --aicpu >"$AICPU_LOG" 2>&1); then
            ok "libscatter_aicpu_kernel.so 编译通过"
        else
            echo "  device 构建日志末尾 20 行："; tail -20 "$AICPU_LOG"
            bad "device 侧编译失败"
        fi
    else
        info "本次只验了 host 侧；device 侧 libscatter_aicpu_kernel.so 要加 --with-aicpu-build"
    fi
fi

printf '\n'
echo "PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
rc="$FAIL"
if [ "$FAIL" -ne 0 ]; then
    echo "selftest 有 $FAIL 项失败"
elif [ "$HARD_SKIP" -ne 0 ]; then
    echo "selftest 通过，但有 $HARD_SKIP 项因前置条件缺失没验成（跳过 ≠ 通过，看上面原因）"
    if [ "$STRICT" -eq 1 ]; then
        echo "--strict：把「没验成」按失败处理"
        rc=1
    else
        echo "CI 里请加 --strict，让「没验成」也返回非 0"
    fi
else
    echo "selftest 全部通过"
fi
exit "$rc"
