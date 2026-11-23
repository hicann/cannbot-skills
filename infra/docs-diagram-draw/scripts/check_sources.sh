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
# check_sources.sh — 规范源版本指纹校验（不保留源文件本体，只锁 commit SHA）。
#
# 指纹文件：references/sources.manifest（每行：源文件路径 | 核实时的源仓 commit SHA | 日期 | 主题）
# 源仓：gitcode.com/cann/docs（master 分支）
#
# 校验方式：对每个源路径调 GitCode API 查该文件在 master 的最新 commit，
# 与 manifest 中记录的 SHA 比对——一致=规范源未更新；不一致=漂移。
#
# 漂移处理流程（人工审阅，勿跳过）：
#   1. 下载新版人工 diff（raw 地址见 [DRIFT] 输出），评估是否影响速查表/样式库/矩阵
#   2. 逐项重新提炼受影响内容（含人工判断，不可自动重生成）
#   3. 随本 skill 发版提交，并更新 sources.manifest 的 SHA 与核实日期
#
# 退出码：0=指纹一致  1=远端已更新（漂移）  2=网络/API 失败
set -u

MANIFEST="$(cd "$(dirname "$0")/../references" && pwd)/sources.manifest"
[ -f "$MANIFEST" ] || { echo "[FAIL]  未找到指纹文件 $MANIFEST"; exit 2; }
API_BASE="https://api.gitcode.com/api/v5/repos/cann/docs/commits"
RAW_BASE="https://raw.gitcode.com/cann/docs/raw/master"

rc=0
while IFS=' ' read -r path pinned_sha _date _subject; do
  case "$path" in ''|\#*) continue ;; esac
  resp="$(curl -fsSL --max-time 30 "$API_BASE?path=$path&ref=master&per_page=1" 2>/dev/null)" || {
    echo "[FAIL]  无法查询 $path 的最新 commit（网络/API 不可达）"
    rc=2
    continue
  }
  latest_sha="$(printf '%s' "$resp" | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["sha"])' 2>/dev/null)" || {
    echo "[FAIL]  $path 的 API 响应解析失败"
    rc=2
    continue
  }
  if [ "$latest_sha" = "$pinned_sha" ]; then
    echo "[OK]    $path 指纹一致（$(printf '%s' "$latest_sha" | cut -c1-12)）"
  else
    echo "[DRIFT] $path 规范源已更新（$pinned_sha → $latest_sha）"
    echo "        下载新版比对: $RAW_BASE/$path"
    echo "        处理：人工 diff 评估 → 重新提炼速查表/样式库/矩阵 → 更新 sources.manifest → 随 skill 发版"
    rc=1
  fi
done < "$MANIFEST"
exit "$rc"
