#!/usr/bin/env bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# 生成/构建 ES API（es_custom）：仅「自定义算子」case 需要——内置算子直接用预生成 es_all
# （设置 GE_ES_API_ROOT/_INCLUDE_DIR/_LIB_DIR/_PYTHONPATH 即可，无需本步骤；
#   缺失时可用兄弟脚本 gen_es_all.sh 自建）。
# 本脚本封装样例根目录的 es_custom 构建流程（cmake -DGE_BUILD_ES_API=ON），并做环境探测与如实降级。
# 宿主中立，禁硬编码默认安装路径（对齐 references/knowledge-base.md）。
# 用法：在样例根目录（含 CMakeLists.txt）执行： bash <skill-root>/scripts/gen_es_api.sh [sample_dir]
set -uo pipefail

SAMPLE_DIR="${1:-$PWD}"
say() { printf '%s\n' "$*"; }
good() { printf '  [OK] %s\n' "$*"; }
miss() { printf '  [WARN] %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
is_cann_home() { [[ -d "$1" && -f "$1/set_env.sh" ]]; }
is_opp_root() {
  [[ -d "$1" ]] &&
    [[ -d "$1/built-in" || -d "$1/vendors" || -d "$1/op_proto" ]]
}

say "== 生成 ES API（es_custom）=="
say "（内置算子无需本步骤：export GE_ES_API_ROOT 指向预生成 es_all 即可）"
say ""

# 1. 环境探测（缺则降级，不硬编码默认安装路径）
missing=0
cann=""
for v in ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME; do
  val="${!v:-}"
  if [[ -n "$val" ]] && is_cann_home "$val"; then cann="$val"; good "$v=$val"; break; fi
done
[[ -n "$cann" ]] || { miss "无可确认的 CANN 根（需目录存在且含 set_env.sh）"; missing=1; }

opp="${ASCEND_OPP_PATH:-}"
if [[ -n "$opp" ]] && is_opp_root "$opp"; then
  good "ASCEND_OPP_PATH=$opp"
elif [[ -n "$cann" ]] && is_opp_root "$cann/opp"; then
  opp="$cann/opp"
  good "OPP 根=$opp（由已验证 CANN 根解析）"
else
  miss "无可确认的 OPP 根"
  missing=1
fi

if command -v cmake >/dev/null 2>&1; then good "cmake 可用"; else miss "cmake 不在 PATH"; missing=1; fi
if [[ -f "$SAMPLE_DIR/CMakeLists.txt" ]]; then good "$SAMPLE_DIR/CMakeLists.txt"; else miss "$SAMPLE_DIR 下无 CMakeLists.txt（不是样例根目录？）"; missing=1; fi

if [[ "$missing" -ne 0 ]]; then
  say ""
  say "环境不完整 → 跳过 es_custom 生成（门禁 G3），如实标注「未运行」。补齐上面 [WARN] 项后重试。"
  say "内置算子路径改用预生成 es_all：export GE_ES_API_ROOT/_INCLUDE_DIR/_LIB_DIR/_PYTHONPATH；缺失可用 gen_es_all.sh 自建。"
  exit 1
fi

# 2. 构建 es_custom
nproc_n="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
say ""
say "-- cmake 配置（-DGE_BUILD_ES_API=ON）--"
cmake -S "$SAMPLE_DIR" -B "$SAMPLE_DIR/build" -DGE_BUILD_ES_API=ON \
  || die "cmake 配置失败（多为 CANN run 包不可用；如实标注「未运行」，勿伪造）"
say "-- 构建 --"
cmake --build "$SAMPLE_DIR/build" -j"$nproc_n" \
  || die "构建失败（见上方日志；先按 references/tips/failure-attribution.md 自查，多是自己漏步）"

say ""
es_output="$SAMPLE_DIR/build/es_output"
custom_lib_count="$(find "$es_output" -type f -name 'libes_custom.so' 2>/dev/null | wc -l | tr -d ' ')"
custom_wheel_count="$(find "$es_output" -type f -name 'es_custom*.whl' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$custom_lib_count" -gt 0 ]]; then
  good "es_custom C/C++ 库已生成（$custom_lib_count 个）"
else
  miss "未找到 libes_custom.so：CMake 成功不等于 custom op ES 库已生成"
fi
if [[ "$custom_wheel_count" -gt 0 ]]; then
  good "es_custom Python wheel 已生成（$custom_wheel_count 个）"
else
  miss "未找到 es_custom Python wheel：Python custom-op pass 仍不可运行"
fi
say "[OK] es_custom 构建完成，产物一般在 $es_output；随后仍须用 check_runtime.py 验证真实 kernel 注册。"
say '加载时按其追加 PYTHONPATH / LD_LIBRARY_PATH（见统一 skill 阶段三“验证顺序与证据”）。'
