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
# 解析 GE 仓根路径，默认**纯只读、绝不联网**；仅 `--allow-network` 才会 fetch/clone。
#
# 契约（供 skill / agent / 其他脚本调用）：
#   - stdout 只输出一行：解析到的 GE 仓根**绝对路径**
#       GE_REPO_PATH="$(bash scripts/sync_ge_repo.sh)" || 按门禁 G3 降级
#   - 所有日志 / 告警 / access log 走 stderr，不污染 stdout
#   - 解析不到 → exit 1；调用方如实标注「GE 仓不可达」，**不猜路径**
#
# 模式矩阵（核心：解析与联网解耦）：
#   模式             | GE_REPO_PATH 有效 | 既有缓存 | 无缓存
#   -----------------+-------------------+----------+--------
#   默认(无参)/--offline | 只读复用          | 用缓存   | exit 1 降级
#   --allow-network  | 只读复用(联网无意义)| fetch 刷新| clone
#   - $GE_REPO_PATH 分支**永远只读**，与模式无关（可能是 fork / 含本地改动）
#   - 默认与 --offline 行为一致：都绝不联网；--offline 保留为显式断言语义
#
# 解析顺序：
#   1. $GE_REPO_PATH —— 用户自有 checkout，**永远只读复用，绝不 fetch/reset**
#   2. skill 自有缓存 $GE_REPO_CACHE_DIR —— 默认直接用（可能陈旧）；仅
#      `--allow-network` 才刷新到 origin/master 最新或浅克隆
#
# Access log（任务5.2第4条）：每次调用向 stderr 输出一行 `GE_REPO_ACCESS: ...`
# 结构化记录（URL/目录/动作/结果/revision/未联网）；设置了 $GE_REPO_ACCESS_LOG
# 时另追加写该文件，供 skill / agent 引用为证据。stderr 记录绝不污染 stdout。
#
# 禁硬编码：不猜 ~/Workspace/ge 之类的位置；根路径只能来自 env 或skill 自有缓存。
#
# 用法：
#   bash scripts/sync_ge_repo.sh                  # 默认：纯只读解析，绝不联网
#   bash scripts/sync_ge_repo.sh --offline        # 同上，显式断言不联网；无缓存则失败
#   bash scripts/sync_ge_repo.sh --allow-network  # 显式联网：刷新/克隆缓存到 master 最新
set -uo pipefail

GE_REPO_URL="${GE_REPO_URL:-https://gitcode.com/cann/ge.git}"
GE_REPO_CACHE_DIR="${GE_REPO_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/ge-fusion-pass-skill/ge}"

network=0
case "${1:-}" in
  --allow-network) network=1 ;;
  --offline) ;;
  "") ;;
  *) printf 'ERROR: 未知参数 %s（支持 --offline / --allow-network）\n' "$1" >&2; exit 2 ;;
esac

log() { printf '%s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# access log：始终打 stderr；$GE_REPO_ACCESS_LOG 指向文件时追加写一行。
access_log() {
  local line="GE_REPO_ACCESS: mode=$1 network=$2 $3"
  printf '%s\n' "$line" >&2
  if [[ -n "${GE_REPO_ACCESS_LOG:-}" ]]; then
    printf '%s\n' "$line" >>"$GE_REPO_ACCESS_LOG" 2>/dev/null || true
  fi
}

# 取缓存仓当前 revision（无 git / 非 git 仓 → 空字符串）
cache_revision() {
  git -C "$GE_REPO_CACHE_DIR" rev-parse --short HEAD 2>/dev/null || true
}

# GE 仓根的判据：融合 pass 样例 + 中文文档树同时存在
is_ge_root() { [[ -d "$1/examples/fusion_pass" && -d "$1/docs/zh" ]]; }
emit_root()  { (cd "$1" && pwd -P); }

# ── 1. 用户自有 checkout：永远只读，不刷新（与模式无关） ────
if [[ -n "${GE_REPO_PATH:-}" ]]; then
  if is_ge_root "$GE_REPO_PATH"; then
    log "使用 GE_REPO_PATH（用户自有 checkout，只读，不 fetch/reset）：$GE_REPO_PATH"
    log "  → 需要最新 master：向用户说明后用 --allow-network 重跑，或直接设 GE_REPO_PATH 指向更新后的 checkout"
    access_log "$([[ "$network" -eq 1 ]] && echo allow-network || echo default)" \
               "no" "resolved=$GE_REPO_PATH"
    emit_root "$GE_REPO_PATH"
    exit 0
  fi
  log "警告：GE_REPO_PATH=$GE_REPO_PATH 不像 GE 仓根（缺 examples/fusion_pass 或 docs/zh），改用skill 缓存。"
fi

# ── 2. 既有缓存：默认直接用（不刷新）；联网模式才刷新 ─────────
if [[ "$network" -eq 0 ]]; then
  # 默认 / --offline：只认既有缓存，绝不联网
  if is_ge_root "$GE_REPO_CACHE_DIR"; then
    log "使用既有缓存（默认不联网，内容可能陈旧）：$GE_REPO_CACHE_DIR"
    log "  → 需要最新 master：向用户说明后用 --allow-network 重跑（skill 须先取得用户授权）"
    access_log "default" "no" "resolved=$GE_REPO_CACHE_DIR"
    emit_root "$GE_REPO_CACHE_DIR"
    exit 0
  fi
  access_log "default" "no" "resolved=UNRESOLVED"
  die "默认不联网且无可用缓存 → 按门禁 G3 标注「GE 仓不可达」，不要猜路径。
  → 需要联网：skill 先向用户说明对象/URL/用途/落盘/替代方案并取得授权，再用 --allow-network 重跑。"
fi

# ── 3. --allow-network：刷新 / 克隆skill 自有缓存 ──────────────
command -v git >/dev/null 2>&1 \
  || die "git 不在 PATH，无法建立/刷新 GE 缓存 → 请设置 GE_REPO_PATH 指向已有 checkout。"

refresh_cache() {
  if [[ -d "$GE_REPO_CACHE_DIR/.git" ]]; then
    log "刷新 GE master 缓存（--allow-network 已授权）：$GE_REPO_CACHE_DIR"
    git -C "$GE_REPO_CACHE_DIR" fetch --depth 1 origin master >&2 \
      && git -C "$GE_REPO_CACHE_DIR" reset -q --hard FETCH_HEAD >&2
    return
  fi
  # 缓存目录存在但不是 git 仓：非空则拒绝，绝不擅自删除用户数据
  if [[ -d "$GE_REPO_CACHE_DIR" ]] &&
     [[ -n "$(find "$GE_REPO_CACHE_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    log "缓存目录已存在且非 git 仓、非空：$GE_REPO_CACHE_DIR"
    log "  → 请手动清理，或用 GE_REPO_CACHE_DIR 指向别处；本脚本不删除任何目录。"
    return 1
  fi
  log "克隆 GE master 到缓存（--allow-network 已授权）：$GE_REPO_CACHE_DIR"
  mkdir -p "$(dirname "$GE_REPO_CACHE_DIR")" || return 1
  git clone --depth 1 -b master "$GE_REPO_URL" "$GE_REPO_CACHE_DIR" >&2
}

if refresh_cache && is_ge_root "$GE_REPO_CACHE_DIR"; then
  access_log "allow-network" "yes" \
    "url=$GE_REPO_URL dir=$GE_REPO_CACHE_DIR action=ok revision=$(cache_revision)"
  emit_root "$GE_REPO_CACHE_DIR"
  exit 0
fi

# 刷新失败但有陈旧缓存 → 降级用旧的，并明确告警
if is_ge_root "$GE_REPO_CACHE_DIR"; then
  log "警告：刷新失败（网络不可达？），回退到既有缓存，内容可能陈旧：$GE_REPO_CACHE_DIR"
  access_log "allow-network" "yes" \
    "url=$GE_REPO_URL dir=$GE_REPO_CACHE_DIR action=fail result=fallback revision=$(cache_revision)"
  emit_root "$GE_REPO_CACHE_DIR"
  exit 0
fi

access_log "allow-network" "yes" \
  "url=$GE_REPO_URL dir=$GE_REPO_CACHE_DIR action=fail result=unresolved"
die "GE 仓不可达：既无有效 GE_REPO_PATH，也无法克隆 $GE_REPO_URL
  → 按门禁 G3 如实标注「GE 仓证据缺失」，不要猜默认安装目录或个人工作区路径。"
