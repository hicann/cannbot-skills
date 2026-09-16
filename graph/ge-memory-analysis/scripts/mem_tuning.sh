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

# 内存寻优脚本：dump 图作为参数，遍历 topo 模式 × 内存优化策略(mop)组合，重放内存分配，找内存最小的组合
# 用法: bash mem_tuning.sh <dump图.txt> [ut可执行文件路径]（在 GE 仓根目录执行，依赖 GE 开发环境）
# 前提: 已编译 UT（BUILD_METADEF=OFF BUILD_PARSER=OFF bash scripts/build_fwk.sh -d -j16）
#       已编译 imas 工具（默认 imas/ 目录，对比表指标均由 imas 解析各组合日志得出）
#       dump 图需为 静态shape + 引擎分配后 阶段（如 ge_proto_xxx_AfterAssignedLogicStreams.txt），
#       否则算不出 size（全0）或引擎分区失败
# 输出: 组合对比表（含流数/集中清零大小）+ 最优组合 + 各组合完整日志（mem_tuning_logs/，可喂给 imas 生成报告）

set -u

DUMP_FILE=${1:?"用法: bash mem_tuning.sh <dump图.txt> [ut可执行文件]"}
UT_BIN=${2:-"build_ut/tests/ge/ut/ge/ge_manual_test"}
GTEST_FILTER="UtestMemoryAssignerManualTest.memory_assign_with_dump_graph"

if [ ! -f "${DUMP_FILE}" ]; then
  echo "错误: dump图不存在: ${DUMP_FILE}"; exit 1
fi
if [ ! -x "${UT_BIN}" ]; then
  echo "UT可执行文件不存在: ${UT_BIN}，自动编译中（已存在时会直接复用，不会走到这里）..."
  BUILD_METADEF=OFF BUILD_PARSER=OFF bash scripts/build_fwk.sh -d -j16 || { echo "错误: UT编译失败"; exit 1; }
fi
if [ ! -x "${UT_BIN}" ]; then
  echo "错误: UT编译后仍不存在: ${UT_BIN}（检查编译输出）"; exit 1
fi

# imas 工具目录（解析各组合日志汇总行，取内存指标/流数/集中清零大小）
IMAS_DIR=${GE_MT_IMAS_DIR:-"imas"}
if [ ! -x "${IMAS_DIR}/imas" ]; then
  echo "错误: imas工具不存在: ${IMAS_DIR}/imas"
  echo "安装: git clone https://gitee.com/stevenaw/imas.git imas && cd imas && bash make_imas（详见 imas-report-reference.md 第1节）"
  echo "或用 GE_MT_IMAS_DIR 指定已安装目录"; exit 1
fi

# UT 运行要求：清掉外部库路径（依赖同目录 rpath）
unset LD_LIBRARY_PATH ASCEND_OPP_PATH

# 寻优维度（可用环境变量覆盖）：
#   topo 模式: 0=BFS 1=DFS 2=RDFS 3=StableRDFS
#   mop 内存优化策略: default=不设置（默认策略），其余值为 ge.exec.memoryOptimizationPolicy 取值（如 MemoryPriority）
MODES=${GE_MT_MODES:-"0 1 2 3"}
MOPS=${GE_MT_MOPS:-"default MemoryPriority"}

# 每组合日志保存目录（含 info 级日志，可用 imas 生成报告深入分析）
LOG_DIR=${GE_MT_LOG_DIR:-"mem_tuning_logs"}
mkdir -p "${LOG_DIR}"

human_size() {
  local b=${1}
  if [ "${b}" -ge 1073741824 ]; then printf "%.2fG" "$(echo "${b}/1073741824" | bc -l)"
  elif [ "${b}" -ge 1048576 ]; then printf "%.2fM" "$(echo "${b}/1048576" | bc -l)"
  elif [ "${b}" -ge 1024 ]; then printf "%.2fK" "$(echo "${b}/1024" | bc -l)"
  else printf "%sB" "${b}"; fi
}

# 按显示宽度右补空格（中文按2列宽计算；printf %-Ns 按字节填充，中文标题与数值会错位）
pad() {
  local bytes nonascii disp n
  bytes=$(printf '%s' "$1" | wc -c)
  nonascii=$(printf '%s' "$1" | LC_ALL=C sed 's/[ -~]//g' | wc -c)
  disp=$(( bytes - nonascii + nonascii / 3 * 2 ))
  n=$(( $2 - disp ))
  [ "${n}" -lt 0 ] && n=0
  printf '%s%*s' "$1" "${n}" ""
}

# 从 imas 汇总行(imas_out)提取字段并求和（多图/多memtype时取总和），字段如 after_reuse/total/theory_min/atomic/continuous
imas_sum() {
  echo "${imas_out}" | grep -oE "$1:[0-9]+" | grep -oE '[0-9]+' | paste -sd+ | bc
}

# 收集结果: "mode|mop_disp|offset|theory|noreuse|total|topo_name|streams|zero_mem"
RESULTS=""
best_mode=""
best_mop_disp=""
best_offset=-1
for mode in ${MODES}; do
  for mop in ${MOPS}; do
    log="${LOG_DIR}/topo${mode}_mop${mop}.log"
    mop_env=""
    mop_disp="default"
    if [ "${mop}" != "default" ]; then
      mop_env="GE_MT_MEMORY_OPTIMIZATION_POLICY=${mop}"
      mop_disp="${mop}"
    fi
    # 开启 info 日志，确保 [IMAS]AfterAssignMemory（GEEVENT 级）输出
    eval "ASCEND_GLOBAL_LOG_LEVEL=1 \
    GE_MT_DUMP_FILE='$(realpath ${DUMP_FILE})' \
    GE_MT_BUILD_FOR_EVALUATE=1 GE_MT_TOPOSORTING_MODE=${mode} ${mop_env} \
    '${UT_BIN}' --gtest_filter='${GTEST_FILTER}'" >"${log}" 2>&1

    # 用 imas 解析日志取汇总行（与 imas report 同源），指标含流数/集中清零大小
    log_abs=$(realpath "${log}")
    imas_out=$( (cd "${IMAS_DIR}" && ./imas total "${log_abs}") | grep -E 'reuse_rate|reuse rate')
    offset=$(imas_sum after_reuse); offset=${offset:-0}
    theory=$(imas_sum theory_min); theory=${theory:-0}
    noreuse=$(imas_sum no_reuse); noreuse=${noreuse:-0}
    total=$(imas_sum total); total=${total:-0}
    atomic=$(imas_sum atomic); atomic=${atomic:-0}
    continuous=$(imas_sum continuous); continuous=${continuous:-0}
    # 流数（多图时取最大值）
    streams=$(echo "${imas_out}" | grep -oE 'streams:[0-9]+' | grep -oE '[0-9]+' | sort -rn | head -1); streams=${streams:-0}
    # 实际生效的 topo 名（从日志确认，如 BFS/DFS/RDFS/StableRDFS）
    topo_name=$(grep -m1 -o 'topo_mode\[[A-Za-z]*\]' "${log}" | grep -o '\[[A-Za-z]*\]' | tr -d '[]')

    if [ -z "${imas_out}" ]; then
      RESULTS="${RESULTS}${mode}|${mop_disp}|-|-|-|-|-|无输出|-\n"
      echo "  注意: topo=${mode} 内存策略=${mop_disp} 无内存统计输出（日志: ${log}）" >&2
      continue
    fi
    RESULTS="${RESULTS}${mode}|${mop_disp}|${offset}|${theory}|${noreuse}|${total}|${topo_name}|${streams}|$((atomic + continuous))\n"

    if [ "${best_offset}" -lt 0 ] || [ "${offset}" -lt "${best_offset}" ]; then
      best_offset=${offset}; best_mode=${mode}; best_mop_disp=${mop_disp}; best_mop_raw=${mop}
    fi
    # 仅匹配 gtest 失败标记（行首 "[  FAILED  ]"），避免误匹配日志中的 "Failed to xxx" WARNING
    [ -n "$(grep -E '^\[  FAILED  \]' "${log}")" ] && echo "  警告: topo=${mode} 内存策略=${mop_disp} 用例失败，结果可能无效（日志: ${log}）" >&2
  done
done

# ---- 结果展示 ----
echo ""
echo "==== 内存寻优结果: $(basename ${DUMP_FILE}) ===="
echo "维度: topo=[$(echo ${MODES} | tr ' ' ',')] x 内存策略=[$(echo ${MOPS} | tr ' ' ',')]"
echo ""
# 表头用 ASCII 列名：中文字符在不同终端/字体下宽度渲染不一致（1或2列），ASCII 可保证表头与数据在任何等宽环境对齐；中文含义见表下图例
echo "$(pad "#" 2) $(pad "Total" 20) $(pad "AfterReuse" 20) $(pad "TheoryMin" 20) $(pad "Atomic+Cont" 20) $(pad "Streams" 6) $(pad "topo" 18) Policy"
echo "------------------------------------------------------------------------------------------------------------------------------------"
idx=0
echo -e "${RESULTS}" | sort -t'|' -k3,3n -k2,2r | while IFS='|' read -r m p off th nr tt tn st zc; do
  [ -z "${m}" ] && continue
  idx=$((idx+1))
  if [ "${off}" = "-" ]; then
    echo "$(pad "${idx}" 2) $(pad "无输出" 20) $(pad "${m}(${tn})" 18) ${p}"
  else
    echo "$(pad "${idx}" 2) $(pad "${tt} ($(human_size ${tt}))" 20) $(pad "${off} ($(human_size ${off}))" 20) $(pad "${th} ($(human_size ${th}))" 20) $(pad "${zc} ($(human_size ${zc}))" 20) $(pad "${st}" 6) $(pad "${m}(${tn})" 18) ${p}"
  fi
done
echo "------------------------------------------------------------------------------------------------------------------------------------"
echo "图例: Total=总内存大小 AfterReuse=复用后大小 TheoryMin=理论最小值 Atomic+Cont=集中清零(atomic+continuous) Streams=流数 Policy=内存策略"

if [ -n "${best_mode:-}" ]; then
  best_log="${LOG_DIR}/topo${best_mode}_mop${best_mop_raw}.log"
  echo "最优组合: topo=${best_mode}($(grep -m1 -o 'topo_mode\[[A-Za-z]*\]' "${best_log}" | grep -o '\[[A-Za-z]*\]' | tr -d '[]')), 内存策略=${best_mop_disp}, 复用后大小=${best_offset} ($(human_size ${best_offset}))"
  config_str="ge.topoSortingMode=\"${best_mode}\""
  [ "${best_mop_disp}" != "default" ] && config_str="${config_str} + ge.exec.memoryOptimizationPolicy=\"${best_mop_disp}\""
  echo "推荐配置: ${config_str}"
  echo ""
  echo "深入分析: cd ${IMAS_DIR} && ./imas report ${best_log}"
  echo "全部日志: ${LOG_DIR}/"
else
  echo "未获得有效结果（检查 dump 图是否为静态shape+引擎分配后阶段，日志见 ${LOG_DIR}/）"; exit 1
fi
