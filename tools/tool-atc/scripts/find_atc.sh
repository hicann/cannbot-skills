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
# 定位 atc 可执行文件并自检环境（对齐 skill 验证要求：先 source set_env.sh 确认环境可用）
# 用法：bash find_atc.sh [-v]
#   -v  额外打印 atc --help 头部与 opp 内置算子配置目录探查结果
set -u

VERBOSE=false
[ "${1:-}" = "-v" ] && VERBOSE=true

print_ok()  { echo "✓ $*"; }
print_warn(){ echo "⚠ $*"; }
print_err() { echo "✗ $*" >&2; }

# 功能校验：确认是 CANN 的 atc，且当前环境能正常启动
validate_atc() {
    local atc="$1" out
    [ -n "$atc" ] && [ -x "$atc" ] || return 1
    out=$("$atc" --help 2>&1 || true)
    printf '%s\n' "$out" | grep -q "soc_version"
}

# 从 atc 路径推导 CANN 根目录；PATH 可能是符号链接，先解析真实路径
resolve_cann_home() {
    local atc="$1" real
    real="$(readlink -f "$atc" 2>/dev/null || printf '%s' "$atc")"
    dirname "$(dirname "$real")"
}

# 1. 候选顺序：PATH（用户实际调用）优先，其次 CANN 环境变量，最后常见安装目录
ATC_BIN=""
CANN_HOME=""

if command -v atc >/dev/null 2>&1; then
    PATH_ATC="$(command -v atc)"
    if validate_atc "$PATH_ATC"; then
        ATC_BIN="$PATH_ATC"
        CANN_HOME="$(resolve_cann_home "$PATH_ATC")"
        # wrapper/非常规布局：PATH 推导不出 CANN 根目录时，回退到已设置的 CANN 环境变量
        if [ ! -f "$CANN_HOME/compiler/version.info" ] && [ ! -d "$CANN_HOME/opp" ]; then
            for var in ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME; do
                val="${!var:-}"
                if [ -n "$val" ] && { [ -f "$val/compiler/version.info" ] || [ -d "$val/opp" ]; }; then
                    print_warn "无法从 PATH atc 推导 CANN 根目录，改用 $var=$val"
                    CANN_HOME="$val"
                    break
                fi
            done
        fi
        print_ok "atc: $ATC_BIN"
        # PATH 与环境变量不一致时提示，避免自检对象与 source 的安装不一致
        for var in ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME; do
            val="${!var:-}"
            [ -n "$val" ] || continue
            val_real="$(readlink -f "$val" 2>/dev/null || printf '%s' "$val")"
            home_real="$(readlink -f "$CANN_HOME" 2>/dev/null || printf '%s' "$CANN_HOME")"
            if [ "$val_real" != "$home_real" ]; then
                print_warn "$var=$val 与 PATH 中 atc 所在目录 $CANN_HOME 不一致；后续检查以 PATH 的 atc 为准"
            fi
        done
    else
        print_err "PATH 中的 atc（$PATH_ATC）无法正常运行或不是 CANN ATC（可能是同名程序），请检查 PATH 顺序或先 source CANN 安装目录 set_env.sh"
        echo "  示例：source <CANN安装根>/set_env.sh"
        exit 1
    fi
else
    for c in "${ASCEND_HOME_PATH:-}" "${ASCEND_TOOLKIT_HOME:-}" \
             "$HOME/Ascend/ascend-toolkit/latest" \
             "/usr/local/Ascend/ascend-toolkit/latest" \
             "$HOME/Ascend/cann" "/usr/local/Ascend/cann"; do
        [ -n "$c" ] || continue
        if [ -x "$c/bin/atc" ]; then
            if validate_atc "$c/bin/atc"; then
                ATC_BIN="$c/bin/atc"
                CANN_HOME="$c"
                print_ok "atc: $ATC_BIN"
                break
            else
                print_warn "候选 atc 无法运行：$c/bin/atc（可能未 source set_env.sh）"
            fi
        fi
    done
    if [ -z "$ATC_BIN" ]; then
        print_err "未找到可用的 atc，请先 source CANN 安装目录 set_env.sh（官方要求，勿手动 export 单个变量）"
        echo "  示例：source <CANN安装根>/set_env.sh"
        exit 1
    fi
fi

# 2. 版本（compiler/version.info 的 Version 字段）
VINFO="$CANN_HOME/compiler/version.info"
if [ -f "$VINFO" ]; then
    ver=$(grep '^Version=' "$VINFO" | cut -d'=' -f2)
    print_ok "CANN 版本: ${ver:-未知}"
else
    print_warn "未找到 $VINFO（版本未知）"
fi

# 3. 内置算子配置目录探查（仅信息展示；转换报 EZ3003 时结合此结果排查 OPP 完整性）
if [ -d "$CANN_HOME/opp/built-in" ]; then
    cfg_cnt=$(find "$CANN_HOME/opp/built-in" -maxdepth 6 -type d -name 'config' 2>/dev/null | wc -l)
    print_ok "opp/built-in 存在（config 目录数: $cfg_cnt）"
    if [ "$VERBOSE" = true ]; then
        echo "--- opp/built-in 顶层 ---"
        ls "$CANN_HOME/opp/built-in" 2>/dev/null
    fi
else
    print_warn "opp/built-in 不存在——转换内置算子可能报 EZ3003，请确认 OPP 组件安装完整"
fi

# 4. 基础可用性
if [ "$VERBOSE" = true ]; then
    echo "--- atc --help（头部） ---"
    "$ATC_BIN" --help 2>&1 | head -8 || true
fi
