#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
set -euo pipefail

BRAND="cannbot"
PLUGIN="shmem-ops-generator"
VERSION="0.1.0"
INCLUDED_SKILLS="shmem-ops-dev shmem-ops-design shmem-ops-testcase-gen shmem-ops-code-gen shmem-ops-compile-debug shmem-ops-correctness-eval shmem-ops-code-review shmem-ops-torch-bind shmem-ops-performance-eval shmem-ops-performance-optim"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$SCRIPT_DIR"
LOCAL_SKILL_ROOT="$PLUGIN_ROOT/skills"

show_help() {
  cat <<'EOF'
SHMEM Ops 插件安装器

用法：
  init.sh [project|global] [opencode|claude|trae|cursor|copilot] [install_path]

参数：
  project       项目级安装（默认）
  global        全局安装
  tool          目标 Agent 工具，默认 opencode
  install_path  项目级安装的目标项目目录；建议传绝对路径，省略时使用当前目录
EOF
}

LEVEL="project"
TOOL="opencode"
INSTALL_PATH="$PWD"

for arg in "$@"; do
  case "$arg" in
    --help|-h) show_help; exit 0 ;;
    project|global) LEVEL="$arg" ;;
    opencode|claude|trae|cursor|copilot) TOOL="$arg" ;;
    *) INSTALL_PATH="$arg" ;;
  esac
done

if [ ! -d "$LOCAL_SKILL_ROOT" ] || [ ! -f "$PLUGIN_ROOT/AGENTS.md" ]; then
  echo "插件目录不完整：缺少 skills/ 或 AGENTS.md" >&2
  exit 1
fi

if [ "$LEVEL" = "global" ]; then
  if [ "$TOOL" = "opencode" ]; then
    CONFIG_ROOT="$HOME/.config/opencode"
  elif [ "$TOOL" = "claude" ]; then
    CONFIG_ROOT="$HOME/.claude"
  elif [ "$TOOL" = "trae" ]; then
    if [ -d "$HOME/.traecli" ]; then CONFIG_ROOT="$HOME/.traecli"; else CONFIG_ROOT="$HOME/.trae"; fi
  elif [ "$TOOL" = "cursor" ]; then
    CONFIG_ROOT="$HOME/.cursor"
  elif [ "$TOOL" = "copilot" ]; then
    CONFIG_ROOT="$HOME/.copilot"
  fi
else
  mkdir -p "$INSTALL_PATH"
  PROJECT_ROOT="$(cd "$INSTALL_PATH" && pwd)"
  if [ "$TOOL" = "opencode" ]; then
    CONFIG_ROOT="$PROJECT_ROOT/.opencode"
  elif [ "$TOOL" = "claude" ]; then
    CONFIG_ROOT="$PROJECT_ROOT/.claude"
  elif [ "$TOOL" = "trae" ]; then
    CONFIG_ROOT="$PROJECT_ROOT/.trae"
  elif [ "$TOOL" = "cursor" ]; then
    CONFIG_ROOT="$PROJECT_ROOT/.cursor"
  elif [ "$TOOL" = "copilot" ]; then
    CONFIG_ROOT="$PROJECT_ROOT/.github"
  fi
fi

if [ "$TOOL" = "claude" ]; then CONTEXT_FILE="CLAUDE.md"; else CONTEXT_FILE="AGENTS.md"; fi

COLLISIONS=0
safe_link() {
  local src="$1" dst="$2" owned_root="$3" current
  if [ -L "$dst" ]; then
    current="$(readlink -f "$dst" 2>/dev/null || true)"
    case "$current" in
      "$owned_root"/*) ;;
      *) echo "警告：$dst 已链接到其他位置，将替换为 SHMEM Ops 版本" >&2; COLLISIONS=$((COLLISIONS + 1)) ;;
    esac
    rm -f "$dst"
  elif [ -e "$dst" ]; then
    echo "错误：$dst 是真实文件或目录，为避免覆盖已跳过；请先手动处理" >&2
    COLLISIONS=$((COLLISIONS + 1))
    return 1
  fi
  ln -sfn "$src" "$dst"
}

mkdir -p "$CONFIG_ROOT/skills"
installed_skills=""
installed_count=0
for skill in $INCLUDED_SKILLS; do
  source_dir="$LOCAL_SKILL_ROOT/$skill"
  if [ ! -f "$source_dir/SKILL.md" ]; then
    echo "错误：插件缺少 skill：$skill" >&2
    exit 1
  fi
  if safe_link "$(realpath "$source_dir")" "$CONFIG_ROOT/skills/$skill" "$(realpath "$LOCAL_SKILL_ROOT")"; then
    installed_skills="$installed_skills\"$skill\","
    installed_count=$((installed_count + 1))
  fi
done

if safe_link "$(realpath "$PLUGIN_ROOT/AGENTS.md")" "$CONFIG_ROOT/$CONTEXT_FILE" "$(realpath "$PLUGIN_ROOT")"; then
  context_status="installed"
else
  context_status="skipped"
fi

installed_skills="${installed_skills%,}"
cat > "$CONFIG_ROOT/cannbot-manifest.json" <<EOF
{
  "brand": "$BRAND",
  "version": "$VERSION",
  "team": "$PLUGIN",
  "level": "$LEVEL",
  "tool": "$TOOL",
  "installed_skills": [$installed_skills],
  "installed_agents": [],
  "brand_dir": "$CONFIG_ROOT",
  "install_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "SHMEM Ops $VERSION 安装完成"
echo "  配置目录：$CONFIG_ROOT"
echo "  Skills：$installed_count/10"
echo "  主配置：$CONTEXT_FILE ($context_status)"
if [ "$COLLISIONS" -gt 0 ]; then echo "  冲突：$COLLISIONS（请检查上方警告）"; fi
