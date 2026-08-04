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
# 生成/构建公共内置算子 ES API（es_all）：所有 pass 开发共用的预生成产物。
# 与 gen_es_api.sh 互补：本脚本管 es_all（内置算子），gen_es_api.sh 管 es_custom（自定义算子）。
# 复用优先：产物已完整则直接给出 export 建议并退出；GE_ES_API_* 已由宿主设置时无需运行本脚本。
# 宿主中立，禁硬编码默认安装路径（对齐 references/knowledge-base.md）。
# 用法：bash <skill-root>/scripts/gen_es_all.sh [build_dir]
#   build_dir 默认 $PWD/build；产物固定落在 <build_dir>/es_output（脚本名、参数与产物路径是对外契约）。
set -uo pipefail

BUILD_DIR="${1:-$PWD/build}"
OUTPUT_ROOT="$BUILD_DIR/es_output"
say() { printf '%s\n' "$*"; }
good() { printf '  [OK] %s\n' "$*"; }
miss() { printf '  [WARN] %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
is_cann_home() { [[ -d "$1" && -f "$1/set_env.sh" ]]; }
is_opp_root() {
  [[ -d "$1" ]] &&
    [[ -d "$1/built-in" || -d "$1/vendors" || -d "$1/op_proto" ]]
}

artifacts_ready() {
  [[ -f "$OUTPUT_ROOT/include/es_all/es_all_ops.h" && -f "$OUTPUT_ROOT/lib64/libes_all.so" ]]
}

print_exports() {
  say "按产物 export（Python whl 在 $OUTPUT_ROOT/whl，pip install --target <目录> 后指向它）："
  say "  export GE_ES_API_ROOT=\"$OUTPUT_ROOT\""
  say "  export GE_ES_API_INCLUDE_DIR=\"$OUTPUT_ROOT/include/es_all\""
  say "  export GE_ES_API_LIB_DIR=\"$OUTPUT_ROOT/lib64\""
  say "  export GE_ES_API_PYTHONPATH=\"<es_all whl 的安装目录>\""
}

say "== 生成公共 ES API（es_all）=="
say "（GE_ES_API_ROOT 等已由宿主设置时无需本步骤，直接使用即可）"
say ""

# 0. 复用优先：已有完整产物则不重复构建
if artifacts_ready; then
  good "复用现有产物: $OUTPUT_ROOT"
  print_exports
  exit 0
fi

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
  opp=""
  miss "无可确认的 OPP 根"
  missing=1
fi
if [[ -n "$opp" && ! -d "$opp/built-in/op_proto" ]]; then
  miss "OPP 根下无 built-in/op_proto（es_all 需要内置算子原型库）"
  missing=1
fi

if [[ -n "$cann" && ! -f "$cann/include/ge/cmake/generate_es_package.cmake" ]]; then
  miss "CANN 根下无 include/ge/cmake/generate_es_package.cmake（CANN 版本不支持 ES 包生成？）"
  missing=1
fi

if command -v cmake >/dev/null 2>&1; then good "cmake 可用"; else miss "cmake 不在 PATH"; missing=1; fi

if [[ "$missing" -ne 0 ]]; then
  say ""
  say "环境不完整 → 跳过 es_all 生成，如实标注「未运行」。补齐上面 [WARN] 项后重试。"
  exit 1
fi

# 2. 合成最小构建工程（只写进 build 目录，不污染调用方源码目录）
PROJECT_DIR="$BUILD_DIR/gen_es_all_project"
mkdir -p "$PROJECT_DIR"
cat > "$PROJECT_DIR/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(gen_es_all LANGUAGES CXX)

# CANN_ROOT / OPP_ROOT 由 gen_es_all.sh 从环境变量解析后传入，无硬编码默认值。
set(CANN_ROOT "" CACHE PATH "CANN root resolved from env by gen_es_all.sh")
set(OPP_ROOT "" CACHE PATH "OPP root resolved from env by gen_es_all.sh")
set(GE_INC_FALLBACK "" CACHE PATH "Optional GE repo inc/ base for headers missing from CANN")
if (NOT CANN_ROOT OR NOT OPP_ROOT)
    message(FATAL_ERROR "CANN_ROOT and OPP_ROOT are required (-DCANN_ROOT=... -DOPP_ROOT=...)")
endif ()

set(ES_OUTPUT_PATH ${CMAKE_BINARY_DIR}/es_output CACHE PATH "generated es_all output path")

list(APPEND CMAKE_MODULE_PATH "${CANN_ROOT}/include/ge/cmake")
find_package(GenerateEsPackage REQUIRED)

set(_extra_include_dirs "${CANN_ROOT}/include" "${CANN_ROOT}/include/graph")
execute_process(COMMAND uname -m OUTPUT_VARIABLE _arch OUTPUT_STRIP_TRAILING_WHITESPACE)
if (_arch AND EXISTS "${CANN_ROOT}/${_arch}-linux/include")
    list(APPEND _extra_include_dirs
        "${CANN_ROOT}/${_arch}-linux/include"
        "${CANN_ROOT}/${_arch}-linux/include/ge")
endif ()
if (GE_INC_FALLBACK)
    list(APPEND _extra_include_dirs
        "${GE_INC_FALLBACK}/graph_metadef"
        "${GE_INC_FALLBACK}/graph_metadef/external"
        "${GE_INC_FALLBACK}/external")
endif ()
include_directories(${_extra_include_dirs})

add_library(opgraph_all INTERFACE)
set_target_properties(opgraph_all PROPERTIES
    INTERFACE_LIBRARY_OUTPUT_DIRECTORY "${OPP_ROOT}/built-in/op_proto"
)

add_es_library_and_whl(
    ES_LINKABLE_AND_ALL_TARGET es_all
    OPP_PROTO_TARGET opgraph_all
    OUTPUT_PATH ${ES_OUTPUT_PATH}
)
EOF

cmake_args=("-DCANN_ROOT=$cann" "-DOPP_ROOT=$opp")
if [[ -n "${GE_REPO_PATH:-}" && -d "$GE_REPO_PATH/inc/graph_metadef/external" ]]; then
  cmake_args+=("-DGE_INC_FALLBACK=$GE_REPO_PATH/inc")
  good "启用 GE 仓头文件回退: $GE_REPO_PATH/inc"
fi

# 3. 配置 + 构建
nproc_n="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
say ""
say "-- cmake 配置 --"
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" "${cmake_args[@]}" \
  || die "cmake 配置失败（多为 CANN run 包不可用；如实标注「未运行」，勿伪造）"
say "-- 构建（target: build_es_all）--"
cmake --build "$BUILD_DIR" --target build_es_all -j"$nproc_n" \
  || die "构建失败（见上方日志；先按 references/tips/failure-attribution.md 自查，多是自己漏步）"

# 4. 产物校验
artifacts_ready \
  || die "构建完成但未发现必要产物（$OUTPUT_ROOT 下应有 include/es_all/es_all_ops.h 与 lib64/libes_all.so）"

say ""
say "[OK] es_all 生成完成: $OUTPUT_ROOT"
print_exports
