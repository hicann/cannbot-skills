#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# triton-op-generator plugin installer.
# Reference layout: cannbot-skills/plugins-official/ops-direct-invoke/init.sh
# ----------------------------------------------------------------------------------------------------------

set -e

# --- Color & output helpers ---
if [ -t 1 ]; then
  GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
  CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'
else
  GREEN=''; YELLOW=''; RED=''; CYAN=''; BOLD=''; DIM=''; NC=''
fi

ok()   { echo -e "  ${DIM}${GREEN}✓${NC}${DIM} $*${NC}"; }
warn() { echo -e "  ${YELLOW}⚠${NC}${DIM} $*${NC}"; }
err()  { echo -e "  ${RED}✗${NC}${DIM} $*${NC}"; }
info() { echo -e "  ${DIM}${CYAN}→${NC}${DIM} $*${NC}"; }
step() { echo -e "${DIM}$*${NC}"; }

BRAND="triton-op-generator"
VERSION="1.0.0"

# --- Plugin-specific filters ---
# Skill whitelist (space-separated list) - all skills bundled with this plugin
INCLUDED_SKILLS="triton-task-extractor triton-op-designer triton-op-coding triton-op-verifier triton-latency-optimizer npu-arch"

# Source agent file (used as CLAUDE.md / AGENTS.md source)
SOURCE_AGENT_FILE="AGENTS.md"

show_banner() {
  echo ""
  echo -e "${CYAN}"
  cat << 'BANNER'
   ____    _    _   _ _   _         ____        _
  / ___|  / \  | \ | | \ | |       | __ )  ___ | |_
 | |     / _ \ |  \| |  \| |       |  _ \ / _ \| __|
 | |___ / ___ \| |\  | |\  |       | |_) | (_) | |_
  \____/_/   \_\_| \_|_| \_|       |____/ \___/ \__|
BANNER
  echo -e "${NC}"
  echo -e "  ${BOLD}Triton-Ascend Operator Code Generation Team${NC}"
  echo ""
}

show_help() {
    cat << EOF
Triton-Op-Generator - Plugin Installer

Usage: install.sh [level] [tool]

Arguments:
  level   - Installation level: "project" (default) or "global"
  tool    - Target tool: "opencode" (default), "claude", "trae", "cursor", or "copilot"

Options:
  --help  - Show this help message

Examples:
  install.sh                       # Project-level, OpenCode
  install.sh project opencode      # Project-level, OpenCode
  install.sh global  opencode      # Global-level, OpenCode
  install.sh project claude        # Project-level, Claude Code
  install.sh global  claude        # Global-level, Claude Code
  install.sh project trae          # Project-level, Trae
  install.sh project cursor        # Project-level, Cursor
  install.sh project copilot       # Project-level, Copilot
  install.sh global  copilot       # Global-level, Copilot

Installation paths:
  OpenCode: .opencode/skills/ + AGENTS.md  (auto-discovered)
  Claude:   .claude/skills/ + CLAUDE.md    (per-item symlinks auto-created)
  Trae:     .trae/skills/ + CLAUDE.md      (project-level only)
  Cursor:   .cursor/skills/ + AGENTS.md    (auto-discovered)
  Copilot:  .github/skills/ + AGENTS.md    (project-level)
            ~/.copilot/skills/ + AGENTS.md (global)

After installation, launch directly:
  OpenCode: opencode
  Claude:   claude
  Trae:     通过 CLI 或 IDE 启动
  Cursor:   通过 Cursor IDE 启动
  Copilot:  通过 GitHub Copilot CLI / IDE 启动

Note: This plugin uses CLAUDE.md/AGENTS.md for direct in-session execution.
      All phases will be visible in real-time during execution.
EOF
}

LEVEL="project"
TOOL="opencode"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$SCRIPT_DIR"

# Try to find the shared ops directory.
# Priority: 1) sibling of plugins-official (source tree)
#           2) plugin cache skills directory (triton-op-generator-skills)
#           3) marketplace ops directory
if [ -d "$PLUGIN_ROOT/../../ops" ]; then
    LOCAL_SKILL_ROOT="$(cd "$PLUGIN_ROOT/../../ops" && pwd)"
elif [ -d "$PLUGIN_ROOT/../../../ops" ]; then
    LOCAL_SKILL_ROOT="$(cd "$PLUGIN_ROOT/../../../ops" && pwd)"
elif [ -d "$PLUGIN_ROOT/../triton-op-generator-skills/1.0.0" ]; then
    # Plugin cache mode: skills are in sibling triton-op-generator-skills directory
    LOCAL_SKILL_ROOT="$(cd "$PLUGIN_ROOT/../triton-op-generator-skills/1.0.0" && pwd)"
elif [ -d "$PLUGIN_ROOT/../../marketplaces/cannbot/ops" ]; then
    # Fallback to marketplace ops
    LOCAL_SKILL_ROOT="$(cd "$PLUGIN_ROOT/../../marketplaces/cannbot/ops" && pwd)"
else
    # Fallback: search upward for ops/ or triton-op-generator-skills/ directory
    SEARCH_DIR="$PLUGIN_ROOT"
    LOCAL_SKILL_ROOT=""
    for _ in $(seq 1 5); do
        SEARCH_DIR="$(dirname "$SEARCH_DIR")"
        if [ -d "$SEARCH_DIR/ops" ]; then
            LOCAL_SKILL_ROOT="$(cd "$SEARCH_DIR/ops" && pwd)"
            break
        fi
        if [ -d "$SEARCH_DIR/triton-op-generator-skills" ]; then
            # Find the version directory
            VER_DIR=$(ls -d "$SEARCH_DIR/triton-op-generator-skills"/*/ 2>/dev/null | head -1)
            if [ -n "$VER_DIR" ]; then
                LOCAL_SKILL_ROOT="$(cd "$VER_DIR" && pwd)"
                break
            fi
        fi
    done
fi

if [ -z "$LOCAL_SKILL_ROOT" ] || [ ! -d "$LOCAL_SKILL_ROOT" ]; then
    err "Cannot find shared ops/ directory. Please run install.sh from the source tree."
    exit 1
fi

for arg in "$@"; do
    case "$arg" in
        --help)                 show_help; exit 0 ;;
        global|project)         LEVEL="$arg" ;;
        opencode|claude|trae|cursor|copilot)   TOOL="$arg" ;;
        *)  echo "Error: Unknown argument '$arg'. Valid: global, project, opencode, claude, trae, cursor, copilot, --help."
            exit 1 ;;
    esac
done

# Determine config root directory and target md filename
if [ "$LEVEL" = "global" ]; then
    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$HOME/.config/opencode"
    elif [ "$TOOL" = "trae" ]; then
        echo "Error: Global installation is not supported for Trae. Use project-level instead."
        exit 1
    elif [ "$TOOL" = "copilot" ]; then
        CONFIG_ROOT="$HOME/.copilot"
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_ROOT="$HOME/.cursor"
    else
        CONFIG_ROOT="$HOME/.claude"
    fi
else
    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$PLUGIN_ROOT/.opencode"
    elif [ "$TOOL" = "trae" ]; then
        CONFIG_ROOT="$PLUGIN_ROOT/.trae"
    elif [ "$TOOL" = "copilot" ]; then
        CONFIG_ROOT="$PLUGIN_ROOT/.github"
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_ROOT="$PLUGIN_ROOT/.cursor"
    else
        CONFIG_ROOT="$PLUGIN_ROOT/.claude"
    fi
fi

# Determine target md filename based on tool
if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "copilot" ]; then
    TARGET_MD_NAME="AGENTS.md"
else
    TARGET_MD_NAME="CLAUDE.md"
fi

BRAND_DIR="$CONFIG_ROOT"

show_banner
echo "  Tool:      $TOOL"
echo "  Level:     $LEVEL"
echo "  Path:      $CONFIG_ROOT"
echo "  MD File:   $TARGET_MD_NAME"
echo ""

# --- Step 0: Confirmation before installation ---
step "[0/4] Checking items to be installed..."

# Collect skills to install (from local skills/)
SKILLS_TO_INSTALL=""
SKILL_COUNT=0
for skill_entry in "$LOCAL_SKILL_ROOT"/*; do
    [ -e "$skill_entry" ] || continue
    name=$(basename "$skill_entry")
    echo "$INCLUDED_SKILLS" | grep -qw "$name" || continue
    SKILLS_TO_INSTALL="$SKILLS_TO_INSTALL $name"
    SKILL_COUNT=$((SKILL_COUNT + 1))
done

# Check if source agent file exists
SOURCE_AGENT_PATH="$PLUGIN_ROOT/$SOURCE_AGENT_FILE"
AGENT_FILE_EXISTS=false
if [ -f "$SOURCE_AGENT_PATH" ]; then
    AGENT_FILE_EXISTS=true
fi

# Display installation plan
echo ""
echo -e "${BOLD}以下内容将被安装/替换：${NC}"
echo ""

if [ "$SKILL_COUNT" -gt 0 ]; then
    echo -e "${CYAN}Skills (${SKILL_COUNT} 项)：${NC}"
    for name in $SKILLS_TO_INSTALL; do
        target="$BRAND_DIR/skills/$name"
        if [ -e "$target" ] || [ -L "$target" ]; then
            echo -e "  ${YELLOW}$name${NC}"
        else
            echo -e "  ${GREEN}$name${NC}"
        fi
    done
    echo ""
fi

if [ "$AGENT_FILE_EXISTS" = true ]; then
    echo -e "${CYAN}${TARGET_MD_NAME} (1 项)：${NC}"
    target="$BRAND_DIR/$TARGET_MD_NAME"
    if [ -e "$target" ] || [ -L "$target" ]; then
        echo -e "  ${YELLOW}${TARGET_MD_NAME}${NC}"
    else
        echo -e "  ${GREEN}${TARGET_MD_NAME}${NC}"
    fi
    echo ""
fi

echo -e "${BOLD}${YELLOW}注意：仅替换上述白名单内的内容，不影响其他已存在的 skills${NC}"
echo ""
ok "开始安装..."
echo ""

# --- Step 1: Create directory + per-item symlinks ---
step "[1/4] Setting up plugin directory..."
mkdir -p "$BRAND_DIR/skills"

# Skills: per-item symlinks for all tools (claude/opencode/trae)
# Pre-clean existing skill symlinks (only whitelist items)
for skill_entry in "$LOCAL_SKILL_ROOT"/*; do
    [ -e "$skill_entry" ] || continue
    name=$(basename "$skill_entry")
    echo "$INCLUDED_SKILLS" | grep -qw "$name" || continue
    target="$BRAND_DIR/skills/$name"
    if [ -e "$target" ] || [ -L "$target" ]; then
        rm -rf "$target"
    fi
done

skill_link_count=0
for skill_entry in "$LOCAL_SKILL_ROOT"/*; do
    [ -e "$skill_entry" ] || continue
    name=$(basename "$skill_entry")
    echo "$INCLUDED_SKILLS" | grep -qw "$name" || continue
    ln -sfn "$(realpath "$skill_entry")" "$BRAND_DIR/skills/$name"
    skill_link_count=$((skill_link_count + 1))
done
ok "Skills: $skill_link_count linked"

# Clean broken symlinks (left over from earlier runs / renamed entries)
for link in "$BRAND_DIR/skills"/*; do
    [ -L "$link" ] && [ ! -e "$link" ] && rm "$link"
done
echo ""

# --- Step 2: Install config file (AGENTS.md / CLAUDE.md) ---
step "[2/3] Installing configuration..."

# Determine target path for config file
# Project-level: install in current directory (PWD) so Claude Code can discover it
# Global-level: install in CONFIG_ROOT
if [ "$LEVEL" = "project" ]; then
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "copilot" ]; then
        config_target="$PWD/AGENTS.md"
    else
        config_target="$PWD/CLAUDE.md"
    fi
else
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "copilot" ]; then
        config_target="$CONFIG_ROOT/AGENTS.md"
    else
        config_target="$CONFIG_ROOT/CLAUDE.md"
    fi
fi

config_src="$PLUGIN_ROOT/AGENTS.md"

# Skip only when source file is already at target location (same filename and same directory)
# This only happens for OpenCode project-level when PLUGIN_ROOT = PWD (AGENTS.md → AGENTS.md)
# For Claude, source is AGENTS.md but target is CLAUDE.md, so always need symlink
if { [ "$TOOL" = "opencode" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "copilot" ]; } && [ "$LEVEL" = "project" ] && [ "$PLUGIN_ROOT" = "$PWD" ]; then
    ok "$(basename "$config_target") already in current directory"
else
    if [ "$LEVEL" = "global" ]; then
        # Global mode: generate a copy with absolute paths so that
        # relative references work from any CWD.
        # Must remove existing symlink first, otherwise `>` would truncate
        # the symlink target (the original AGENTS.md) before sed reads it.
        [ -e "$config_target" ] || [ -L "$config_target" ] && rm -f "$config_target"
        PLUGIN_ROOT_ABS="$(realpath "$PLUGIN_ROOT")"
        ESCAPED_ROOT="$(echo "$PLUGIN_ROOT_ABS" | sed 's/#/\\#/g')"
        sed \
          -e "s#\`workflows/#\`${ESCAPED_ROOT}/workflows/#g" \
          "$config_src" > "$config_target"
        ok "$(basename "$config_target") (absolute paths for global mode)"
    else
        ln -sf "$config_src" "$config_target"
        ok "$(basename "$config_target")"
    fi
fi
echo ""

# --- Step 3: Health check + manifest ---
step "[3/3] Running health check..."
health_ok=true
health_errors=""

for sub in skills; do
  target="$BRAND_DIR/$sub"
  if [ -d "$target" ]; then
    count=$(ls -d "$target"/* 2>/dev/null | wc -l)
    [ "$count" -eq 0 ] && { health_errors="${health_errors}\n  ${YELLOW}⚠${NC} $sub/ is empty"; }
  else
    health_errors="${health_errors}\n  ${RED}✗${NC} $sub/ missing"
    health_ok=false
  fi
done

# Check config file (AGENTS.md / CLAUDE.md)
if [ "$LEVEL" = "project" ]; then
    # Project-level: config file is in current directory (PWD)
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "copilot" ]; then
        [ -f "$PWD/AGENTS.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} AGENTS.md missing in current directory"; health_ok=false; }
    else
        [ -f "$PWD/CLAUDE.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} CLAUDE.md missing in current directory"; health_ok=false; }
    fi
else
    # Global-level: config file in CONFIG_ROOT
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "copilot" ]; then
        [ -f "$CONFIG_ROOT/AGENTS.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} AGENTS.md missing"; health_ok=false; }
    else
        [ -f "$CONFIG_ROOT/CLAUDE.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} CLAUDE.md missing"; health_ok=false; }
    fi
fi

# Generate brand manifest
MANIFEST="$CONFIG_ROOT/triton-op-generator-manifest.json"

SKILLS_JSON="[]"
if [ -d "$BRAND_DIR/skills" ]; then
  SKILLS_JSON=$(ls -d "$BRAND_DIR/skills"/* 2>/dev/null | while read d; do
    basename "$d"
  done | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
fi

cat > "$MANIFEST" << MANIFEST_EOF
{
  "brand": "$BRAND",
  "version": "$VERSION",
  "team": "$BRAND",
  "level": "$LEVEL",
  "tool": "$TOOL",
  "installed_skills": $SKILLS_JSON,
  "installed_md_file": "$TARGET_MD_NAME",
  "brand_dir": "$CONFIG_ROOT",
  "install_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
MANIFEST_EOF

[ -f "$MANIFEST" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} Manifest generation failed"; health_ok=false; }

if [ "$health_ok" = true ] && [ -z "$health_errors" ]; then
  ok "All checks passed"
else
  echo -e "$health_errors"
  [ "$health_ok" = true ] && warn "Some warnings, see above" || err "Some checks failed, see above"
fi
echo ""

# --- Step 4: Summary & Quick Start ---
step "[4/4] Done."
echo ""
echo -e "  ${GREEN}${BOLD}✓ triton-op-generator installed successfully!${NC}"
echo ""
echo -e "  ${BOLD}Quick Start:${NC}"
if [ "$TOOL" = "opencode" ]; then
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}opencode${NC}"
  echo -e "  ${CYAN}2.${NC} 直接输入需求: ${GREEN}${BOLD}生成一个 Triton-Ascend 框架的 softmax 算子实现，ASCEND_RT_VISIBLE_DEVICES=1${NC}"
elif [ "$TOOL" = "trae" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 CLI/IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} 直接输入需求: ${GREEN}${BOLD}生成一个 Triton-Ascend 框架的 softmax 算子实现，ASCEND_RT_VISIBLE_DEVICES=1${NC}"
elif [ "$TOOL" = "copilot" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 GitHub Copilot CLI / IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} 直接输入需求: ${GREEN}${BOLD}生成一个 Triton-Ascend 框架的 softmax 算子实现，ASCEND_RT_VISIBLE_DEVICES=1${NC}"
elif [ "$TOOL" = "cursor" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 Cursor IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} 直接输入需求: ${GREEN}${BOLD}生成一个 Triton-Ascend 框架的 softmax 算子实现，ASCEND_RT_VISIBLE_DEVICES=1${NC}"
else
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}claude${NC}"
  echo -e "  ${CYAN}2.${NC} 直接输入需求: ${GREEN}${BOLD}生成一个 Triton-Ascend 框架的 softmax 算子实现，ASCEND_RT_VISIBLE_DEVICES=1${NC}"
fi
echo ""
echo -e "  ${DIM}Note: 所有执行阶段将在当前会话中实时显示${NC}"
echo ""
