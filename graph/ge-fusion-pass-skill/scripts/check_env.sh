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
# 环境体检（doctor）：一次性探测本 skill 三段流水线依赖的环境能力，
# 并说明「缺 X → 会跳过哪一步验证」。宿主中立：只探测、不假设 harness 注入，
# 禁硬编码默认安装路径（对齐 references/knowledge-base.md）。纯信息性，默认 exit 0。
# 用法：bash <skill-root>/scripts/check_env.sh   （从任意目录调用均可）
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ok=0; warn=0
say()  { printf '%s\n' "$*"; }
good() { printf '  [OK] %s\n' "$*"; ok=$((ok+1)); }
miss() { printf '  [WARN] %s\n' "$*"; warn=$((warn+1)); }
have() { command -v "$1" >/dev/null 2>&1; }
is_cann_home() { [[ -d "$1" && -f "$1/set_env.sh" ]]; }
is_opp_root() {
  [[ -d "$1" ]] &&
    [[ -d "$1/built-in" || -d "$1/vendors" || -d "$1/op_proto" ]]
}

say "== ge-fusion-pass-skill 环境体检 =="
say "（只探测不改环境；缺失项如实标注，对应验证步骤会被跳过——门禁 G3）"
say ""

# 1. CANN / OPP 能力 ───────────────────────────────────────
say "1. CANN / OPP（C++ 构建 / ES 检查 / 触发 GE 编译）"
cann_home=""
for v in ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME; do
  val="${!v:-}"
  if [[ -n "$val" ]]; then
    if is_cann_home "$val"; then
      good "$v=$val"
      cann_home="${cann_home:-$val}"
    else
      miss "$v=$val（不是可确认的 CANN 根：缺目录或 set_env.sh）"
    fi
  fi
done
[[ -n "$cann_home" ]] || miss "CANN 根不可用 → 跳过依赖 CANN toolkit 的构建/编译步骤"

opp_root=""
if [[ -n "${ASCEND_OPP_PATH:-}" ]]; then
  if is_opp_root "$ASCEND_OPP_PATH"; then
    opp_root="$ASCEND_OPP_PATH"
    good "ASCEND_OPP_PATH=$opp_root"
  else
    miss "ASCEND_OPP_PATH=$ASCEND_OPP_PATH（不是可确认的 OPP 根）"
  fi
elif [[ -n "$cann_home" ]] && is_opp_root "$cann_home/opp"; then
  opp_root="$cann_home/opp"
  good "OPP 根=$opp_root（由已验证 CANN 根解析）"
else
  miss "OPP 根不可用 → 跳过 op proto / vendor / ES 相关检查"
fi
say ""

# 2. 编译器工具 ────────────────────────────────────────────
say "2. ATC / pyatc（离线编译验证）"
for t in atc pyatc; do
  if have "$t"; then good "$t 可用（$(command -v "$t")）"; else miss "$t 不在 PATH → 跳过对应离线编译验证"; fi
done
say ""

# 3. soc version ───────────────────────────────────────────
say "3. soc_version（离线编译 --soc_version 需完整值，如 Ascend910_9362）"
if soc_out="$(bash "$ROOT/scripts/detect_soc_version.sh" 2>/dev/null)"; then
  line="$(printf '%s\n' "$soc_out" | grep -E '^ATC argument: --soc_version=' || true)"
  if [[ -n "$line" ]]; then good "$line"; else miss "DSMI 未返回完整 soc_version → 离线编译前需手动确认"; fi
else
  miss "无法探测 soc_version（无 NPU 或缺 CANN DSMI helper）→ 离线编译前需手动确认"
fi
say ""

# 4. GE 仓（文档 + 样例，已全量开源，不再依赖 ge-document）──
say "4. GE 仓（接口清单 / 签名核对 / 注册阶段与开关 / 样例）"
say "   解析：scripts/sync_ge_repo.sh —— GE_REPO_PATH 有效则只读复用，否则刷新 master 缓存"
if repo="$(bash "$ROOT/scripts/sync_ge_repo.sh" --offline 2>/dev/null)"; then
  good "GE 仓根=$repo"
  if [[ -f "$repo/examples/fusion_pass/README.md" ]]; then
    good "examples/fusion_pass（实操开发指南 + 样例）"
    for sample_group in pattern_base_pass graph_base_pass; do
      sample_dir="$repo/examples/fusion_pass/$sample_group"
      if [[ -d "$sample_dir" ]]; then
        sample_count="$(find "$sample_dir" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
        good "$sample_group 当前 checkout 子样例数=$sample_count（以 README/实际目录为准）"
      else
        miss "缺 examples/fusion_pass/$sample_group → 对应样例路线不可用"
      fi
    done
  else
    miss "缺 examples/fusion_pass → C++/Python 开发指南与样例不可用"
  fi
  if [[ -d "$repo/docs/zh/api/graph_engine_api" ]]; then
    good "docs/zh/api/graph_engine_api（API 文档）"
  else
    miss "缺 docs/zh/api/graph_engine_api → 签名核对需回退 CANN 头文件"
  fi
  options_doc="$repo/docs/zh/api/graph_engine_api/cpp/ge/options_params/options_parameters_description.md"
  if [[ -f "$options_doc" ]]; then
    good "cpp/ge/options_params/options_parameters_description.md（开关策略 / 注册阶段依据）"
  else
    miss "缺 cpp/ge/options_params/options_parameters_description.md → 从 API README 定位后再核对；注册阶段/开关按降级规则标注假设"
  fi
else
  miss "GE 仓不可达（无有效 GE_REPO_PATH，且无本地缓存；默认不联网）"
  miss "  → 需最新 master：向用户说明对象/URL/用途/落盘/替代方案并取得授权后，跑 bash \"$ROOT/scripts/sync_ge_repo.sh\" --allow-network"
  miss "  → 或设 GE_REPO_PATH 指向已有 checkout（脚本只读复用，不动你的本地改动）"
  miss "  → 在此之前：接口清单/签名/注册阶段/开关 均按降级规则标注假设，不编结论"
fi
say ""

# 5. ES API（内置算子 es_all）─────────────────────────────
say "5. ES API（内置算子用预生成 es_all；自定义算子才需 es_custom，见随包 gen_es_api.sh）"
esroot="${GE_ES_API_ROOT:-}"
if [[ -n "$esroot" && -d "$esroot" ]]; then
  good "GE_ES_API_ROOT=$esroot"
elif [[ -n "$esroot" ]]; then
  miss "GE_ES_API_ROOT=$esroot（目录不存在）"
else
  miss "GE_ES_API_ROOT 未设置"
fi

es_include="${GE_ES_API_INCLUDE_DIR:-${esroot:+$esroot/include}}"
if [[ -n "$es_include" && -f "$es_include/es_all/es_all_ops.h" ]]; then
  good "es_all C++ 头文件=$es_include/es_all/es_all_ops.h"
else
  miss "es_all C++ 头文件不可用 → C++ ES API 签名/构建检查跳过"
fi

es_lib="${GE_ES_API_LIB_DIR:-${esroot:+$esroot/lib64}}"
if [[ -n "$es_lib" && -f "$es_lib/libes_all.so" ]]; then
  good "es_all 库=$es_lib/libes_all.so"
else
  miss "es_all 库不可用 → ES 链接/运行检查跳过"
fi

es_python="${GE_ES_API_PYTHONPATH:-}"
if [[ -n "$es_python" && -d "$es_python" ]]; then
  if have python3 && PYTHONPATH="$es_python${PYTHONPATH:+:$PYTHONPATH}" \
      LD_LIBRARY_PATH="$es_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      python3 -c 'import ge.es' >/dev/null 2>&1; then
    good "Python ES API 可导入（GE_ES_API_PYTHONPATH=$es_python）"
  else
    miss "GE_ES_API_PYTHONPATH 存在但 import ge.es 失败"
  fi
elif [[ -n "$es_python" ]]; then
  miss "GE_ES_API_PYTHONPATH=$es_python（目录不存在）"
else
  miss "GE_ES_API_PYTHONPATH 未设置 → Python ES API 检查跳过"
fi
say ""

# 6. 在线路径依赖 ─────────────────────────────────────────
say "6. 在线路径依赖（Python pass 加载 / 在线脚本触发编译）"
if have python3; then
  for mod in torch_npu npu_bridge; do
    if python3 -c "import $mod" >/dev/null 2>&1; then good "python: import $mod"; else miss "python: 无法 import $mod → 对应在线触发路径跳过"; fi
  done
  if python3 -c 'import torch_npu; import torchair; import ge, ge.passes' >/dev/null 2>&1; then
    good "python: torch_npu → torchair → ge.passes 导入顺序可用"
  else
    miss "python: torch_npu → torchair → ge.passes 导入链不可用 → Python pass 路径跳过"
  fi
  if python3 -c 'import tensorflow as tf; assert hasattr(tf, "compat")' >/dev/null 2>&1; then
    good "python: TensorFlow 可导入（TF1 graph-mode 仍需核对版本）"
  else
    miss "python: TensorFlow 不可导入 → TF/PB 生成与在线路径跳过"
  fi
  if python3 "$ROOT/scripts/check_runtime.py" --strict >/dev/null 2>&1; then
    good "Python pass bridge runtime 可用"
  else
    miss "Python pass bridge runtime 不完整 → 详见 $ROOT/scripts/check_runtime.py 输出"
  fi
else
  miss "python3 不在 PATH → Python pass 加载与在线脚本相关步骤跳过"
fi
say ""

# 汇总 ─────────────────────────────────────────────────────
say "== 汇总：$ok 项就绪 / $warn 项缺失 =="
if [[ "$warn" -eq 0 ]]; then
  say "环境完整，三段流水线的验证步骤均可实跑。"
else
  say "存在缺失项：上面标 [WARN] 的能力对应步骤会被如实标注「未运行」并跳过（门禁 G3），其余照跑。"
  say "补齐可按各行提示设置环境变量或本地 clone/生成。"
fi
exit 0
