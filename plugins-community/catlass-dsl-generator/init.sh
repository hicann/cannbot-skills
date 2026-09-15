#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with
# the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -e

# CANNBot · catlass-dsl-generator 多工具安装脚本
#
# 将本插件（CATLASS DSL 算子开发工作流）安装到各 AI 编码工具的发现路径：
#   - skills  : <config>/skills/catlass-dsl-{design,develop,bench,optimize,knowledge}
#   - 知识库  : <config>/knowledge（插件根只读内置知识，OKF bundle）
#   - 配置    : AGENTS.md / CLAUDE.md（工具每个新会话自动加载，无需对话上下文即可恢复工作流）
#   - 清单    : <config>/cannbot-manifest.json（幂等安装与健康检查依据）
#
# 用法:
#   bash init.sh [project|global] [tool] [install_path]
#   bash init.sh --uninstall [project|global] [tool] [install_path]
#
# tool ∈ { opencode, claude, trae, cursor, codex, copilot, codearts }（默认 opencode）
#   - opencode: .opencode/{skills,knowledge}/     配置 AGENTS.md
#   - claude  : .claude/{skills,knowledge}/       配置 CLAUDE.md
#   - trae    : .trae/.marscode/.traecli 按变体   配置 AGENTS.md（仅项目级）
#   - cursor  : .cursor/{skills,knowledge}/      配置 AGENTS.md
#   - codex   : .agents/{skills,knowledge}/      配置 AGENTS.md
#   - copilot : .github/{skills,knowledge}/      配置 AGENTS.md
#   - codearts: .codeartsdoer/{skills,knowledge}/ 配置 AGENTS.md
#
# Trae 变体自动检测（与官方插件一致）:
#   ~/.trae-cn  → .trae     (Trae IDE)
#   ~/.marscode → .marscode  (Trae 插件版)
#   ~/.traecli  → .traecli  (Trae CLI)

# --- Color & output helpers ---
if [ -t 1 ]; then
  GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
  CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'
else
  GREEN=''; YELLOW=''; RED=''; CYAN=''; BOLD=''; DIM=''; NC=''
fi

ok()   { echo -e "  ${DIM}${GREEN}ok${NC}${DIM} $*${NC}"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC}${DIM} $*${NC}"; }
err()  { echo -e "  ${RED}xx${NC}${DIM} $*${NC}"; }
info() { echo -e "  ${DIM}${CYAN}>>${NC}${DIM} $*${NC}"; }
step() { echo -e "${DIM}$*${NC}"; }

BRAND="cannbot"
VERSION="0.3.0"
# Plain ASCII marker: must NOT contain '!' or other history-expansion
# triggers -- interactive shells may expand such chars when sourcing this
# script (observed on AI-integrated terminals).
AGENTS_MARKER="generated-by-cannbot-catlass-dsl-generator-init"

# 本插件全部 skills（安装/卸载白名单，仅触碰这些内容）
INCLUDED_SKILLS="catlass-dsl-design catlass-dsl-develop catlass-dsl-bench catlass-dsl-optimize catlass-dsl-knowledge"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$SCRIPT_DIR"

detect_trae_variant() {
    if [ -d "$HOME/.trae-cn" ]; then
        TRAE_VARIANT="ide"
    elif [ -d "$HOME/.marscode" ]; then
        TRAE_VARIANT="plugin"
    elif [ -d "$HOME/.traecli" ]; then
        TRAE_VARIANT="cli"
    else
        TRAE_VARIANT="unknown"
    fi
}

show_help() {
    cat << 'EOF'
CANNBot - CATLASS DSL Generator Installer (v0.3.0)

Usage:
  init.sh [level] [tool] [install_path]      Install (default: project opencode)
  init.sh --uninstall [level] [tool] [install_path]
                                             Remove only whitelisted content installed by this script

Arguments:
  level        - "project" (default) or "global" (not supported for trae)
  tool         - opencode (default), claude, trae, cursor, codex, copilot, codearts
  install_path - Project root for project-level install (default: current directory)

Installation paths:
  OpenCode:     .opencode/{skills,knowledge}/    + AGENTS.md
  Claude Code:  .claude/{skills,knowledge}/      + CLAUDE.md
  Trae IDE:     .trae/{skills,knowledge}/       + AGENTS.md (project-level only)
  Trae Plugin:  .marscode/{skills,knowledge}/    + AGENTS.md (project-level only)
  Trae CLI:     .traecli/{skills,knowledge}/    + AGENTS.md (project-level only)
  Cursor:       .cursor/{skills,knowledge}/      + AGENTS.md
  Codex:        .agents/{skills,knowledge}/      + AGENTS.md
  Copilot:      .github/{skills,knowledge}/      + AGENTS.md
  CodeArts:     .codeartsdoer/{skills,knowledge}/ + AGENTS.md
  Global mode installs under the tool's home config dir (e.g. ~/.claude, ~/.cursor).

Notes:
  - All installed entries are symlinks; the plugin repo stays the single source.
  - Uninstall removes ONLY whitelisted skills/knowledge/config/manifest; pass the
    same level/tool/install_path used at install time.
EOF
}

# --- Argument parsing ---
LEVEL="project"
TOOL="opencode"
ACTION="install"
INSTALL_PATH=""

for arg in "$@"; do
    case "$arg" in
        --help|-h)        show_help; exit 0 ;;
        --uninstall)      ACTION="uninstall" ;;
        project|global)   LEVEL="$arg" ;;
        opencode|claude|trae|cursor|codex|copilot|codearts) TOOL="$arg" ;;
        *)                INSTALL_PATH="$arg" ;;
    esac
done

if [ "$LEVEL" = "global" ] && [ "$TOOL" = "trae" ]; then
    err "Global installation is not supported for Trae. Use: bash init.sh project trae"
    exit 1
fi

# --- Resolve project root ---
if [ -n "$INSTALL_PATH" ]; then
    PROJECT_ROOT="$(cd "$INSTALL_PATH" && pwd)"
else
    PROJECT_ROOT="$PWD"
fi

# --- Resolve per-tool paths ---
CONFIG_FILE_NAME="AGENTS.md"
if [ "$TOOL" = "claude" ]; then
    CONFIG_FILE_NAME="CLAUDE.md"
fi

if [ "$LEVEL" = "global" ]; then
    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$HOME/.config/opencode"
    elif [ "$TOOL" = "claude" ]; then
        CONFIG_ROOT="$HOME/.claude"
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_ROOT="$HOME/.cursor"
    elif [ "$TOOL" = "copilot" ]; then
        CONFIG_ROOT="$HOME/.copilot"
    elif [ "$TOOL" = "codearts" ]; then
        CONFIG_ROOT="$HOME/.codeartsdoer"
    else
        # codex
        CONFIG_ROOT="$HOME/.agents"
    fi
    SKILL_DISCOVERY_ROOT="$CONFIG_ROOT/skills"
    if [ "$TOOL" = "claude" ]; then
        CONFIG_FILE_TARGET="$CONFIG_ROOT/CLAUDE.md"
    else
        CONFIG_FILE_TARGET="$CONFIG_ROOT/AGENTS.md"
    fi
else
    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$PROJECT_ROOT/.opencode"
    elif [ "$TOOL" = "claude" ]; then
        CONFIG_ROOT="$PROJECT_ROOT/.claude"
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_ROOT="$PROJECT_ROOT/.cursor"
    elif [ "$TOOL" = "copilot" ]; then
        CONFIG_ROOT="$PROJECT_ROOT/.github"
    elif [ "$TOOL" = "codearts" ]; then
        CONFIG_ROOT="$PROJECT_ROOT/.codeartsdoer"
    elif [ "$TOOL" = "trae" ]; then
        detect_trae_variant
        case "$TRAE_VARIANT" in
            plugin) CONFIG_ROOT="$PROJECT_ROOT/.marscode" ;;
            cli)    CONFIG_ROOT="$PROJECT_ROOT/.traecli" ;;
            *)      CONFIG_ROOT="$PROJECT_ROOT/.trae" ;;
        esac
    else
        # codex
        CONFIG_ROOT="$PROJECT_ROOT/.agents"
    fi
    SKILL_DISCOVERY_ROOT="$CONFIG_ROOT/skills"
    CONFIG_FILE_TARGET="$PROJECT_ROOT/$CONFIG_FILE_NAME"
fi
MANIFEST="$CONFIG_ROOT/cannbot-manifest.json"

# ------------------------------------------------------------------
# Uninstall: remove ONLY whitelisted entries created by this script
# ------------------------------------------------------------------
uninstall() {
    step "[*] Uninstalling catlass-dsl-generator ($LEVEL/$TOOL) from $PROJECT_ROOT ..."
    echo ""

    for name in $INCLUDED_SKILLS; do
        target="$SKILL_DISCOVERY_ROOT/$name"
        if [ -L "$target" ] || [ -d "$target" ]; then
            rm -rf "$target"
            ok "removed skill $name"
        else
            info "skill $name not present, skipping"
        fi
    done

    # knowledge is a generic dir name -- only remove it if it resolves to
    # THIS plugin's knowledge dir; a link owned by another plugin is kept.
    kn="$CONFIG_ROOT/knowledge"
    if [ -L "$kn" ]; then
        if [ "$(readlink -f "$kn")" = "$(readlink -f "$PLUGIN_ROOT/knowledge")" ]; then
            rm -f "$kn"
            ok "removed knowledge"
        else
            info "knowledge not owned by this plugin, kept untouched"
        fi
    elif [ -e "$kn" ]; then
        info "knowledge is not a symlink, kept untouched"
    else
        info "knowledge not present, skipping"
    fi

    # cannbot-manifest.json is brand-generic -- verify the plugin field
    # before removing, so another cannbot plugin's manifest survives.
    mf="$CONFIG_ROOT/cannbot-manifest.json"
    if [ -f "$mf" ]; then
        if grep -q '"plugin": *"catlass-dsl-generator"' "$mf" 2>/dev/null; then
            rm -f "$mf"
            ok "removed cannbot-manifest.json"
        else
            info "cannbot-manifest.json not owned by this plugin, kept untouched"
        fi
    else
        info "cannbot-manifest.json not present, skipping"
    fi

    if [ -f "$CONFIG_FILE_TARGET" ] && head -20 "$CONFIG_FILE_TARGET" | grep -qF "$AGENTS_MARKER"; then
        rm -f "$CONFIG_FILE_TARGET"
        ok "removed $CONFIG_FILE_NAME (generated by this script)"
    else
        info "$CONFIG_FILE_NAME not generated by this script (or absent), kept untouched"
    fi

    # Clean broken whitelisted symlinks and empty dirs created by us
    if [ -d "$SKILL_DISCOVERY_ROOT" ]; then
        for name in $INCLUDED_SKILLS; do
            link="$SKILL_DISCOVERY_ROOT/$name"
            [ -L "$link" ] && [ ! -e "$link" ] && rm -f "$link"
        done
        [ -z "$(ls -A "$SKILL_DISCOVERY_ROOT")" ] && rmdir "$SKILL_DISCOVERY_ROOT" 2>/dev/null && ok "removed empty skills/"
    fi
    # Remove the tool config dir itself only if we left it completely empty
    if [ -d "$CONFIG_ROOT" ] && [ -z "$(ls -A "$CONFIG_ROOT")" ] && [ "$CONFIG_ROOT" != "$HOME" ]; then
        rmdir "$CONFIG_ROOT" 2>/dev/null && ok "removed empty $CONFIG_ROOT/"
    fi

    echo ""
    ok "Uninstall complete. Only whitelisted entries were removed."
}

# ------------------------------------------------------------------
# Install
# ------------------------------------------------------------------
install() {
    step "[1/4] Preparing $SKILL_DISCOVERY_ROOT ..."
    mkdir -p "$SKILL_DISCOVERY_ROOT"
    if [ "$TOOL" = "trae" ]; then
        case "$TRAE_VARIANT" in
            ide)     info "Detected: TRAE IDE (.trae-cn)" ;;
            plugin)  info "Detected: TRAE Plugin (.marscode)" ;;
            cli)     info "Detected: TRAE CLI (.traecli)" ;;
            unknown)
                warn "TRAE variant not detected; defaulting to IDE path (.trae)"
                warn "If you use TRAE Plugin, ensure ~/.marscode exists before re-running"
                warn "If you use TRAE CLI, ensure ~/.traecli exists before re-running" ;;
        esac
    fi
    echo ""

    # Display plan
    echo -e "${BOLD}以下内容将被安装/替换（均为软链，不复制文件）：${NC}"
    echo ""
    for name in $INCLUDED_SKILLS; do
        target="$SKILL_DISCOVERY_ROOT/$name"
        if [ -e "$target" ] || [ -L "$target" ]; then
            echo -e "  ${YELLOW}$name${NC}"
        else
            echo -e "  ${GREEN}$name${NC}"
        fi
    done
    [ -e "$CONFIG_ROOT/knowledge" ] || [ -L "$CONFIG_ROOT/knowledge" ] \
        && echo -e "  ${YELLOW}knowledge/${NC}" || echo -e "  ${GREEN}knowledge/${NC}"
    if [ -f "$CONFIG_FILE_TARGET" ]; then
        if head -20 "$CONFIG_FILE_TARGET" | grep -qF "$AGENTS_MARKER"; then
            echo -e "  ${YELLOW}$CONFIG_FILE_NAME${NC} (由本脚本生成，将更新)"
        else
            echo -e "  ${YELLOW}$CONFIG_FILE_NAME${NC} (已存在自定义内容，将备份后替换)"
        fi
    else
        echo -e "  ${GREEN}$CONFIG_FILE_NAME${NC} (将创建)"
    fi
    echo ""
    echo -e "${BOLD}${YELLOW}注意：仅替换上述白名单内的内容，不影响其他已存在的 skills${NC}"
    echo ""

    # --- Step 1: skill symlinks ---
    step "[2/4] Linking skills ..."
    link_count=0
    for name in $INCLUDED_SKILLS; do
        src="$PLUGIN_ROOT/skills/$name"
        if [ ! -d "$src" ]; then
            err "skill source missing: $src"
            exit 1
        fi
        ln -sfn "$(realpath "$src")" "$SKILL_DISCOVERY_ROOT/$name"
        link_count=$((link_count + 1))
    done
    ok "Skills: $link_count symlinks -> $SKILL_DISCOVERY_ROOT"

    # --- knowledge bundle (read-only built-in, sibling of skills/) ---
    if [ -d "$PLUGIN_ROOT/knowledge" ]; then
        ln -sfn "$(realpath "$PLUGIN_ROOT/knowledge")" "$CONFIG_ROOT/knowledge"
        ok "Knowledge: $CONFIG_ROOT/knowledge (read-only built-in OKF bundle)"
    else
        warn "knowledge/ not found in plugin root, skipping"
    fi

    # --- Step 2: config file (auto-loaded by every new session) ---
    step "[3/4] Installing $CONFIG_FILE_NAME ..."
    config_src="$PLUGIN_ROOT/AGENTS.md"
    if [ ! -f "$config_src" ]; then
        err "config source missing: $config_src"
        exit 1
    fi
    if [ -f "$CONFIG_FILE_TARGET" ] && ! head -20 "$CONFIG_FILE_TARGET" | grep -qF "$AGENTS_MARKER"; then
        backup="${CONFIG_FILE_TARGET}.bak.$(date +%Y%m%d_%H%M%S)"
        cp -a "$CONFIG_FILE_TARGET" "$backup"
        warn "$CONFIG_FILE_NAME already exists, backed up to $(basename "$backup")"
    fi

    # The plugin-root AGENTS.md is the single source of the workflow config;
    # dynamic paths are injected via placeholder + sed (same approach as the
    # official plugins' global mode).
    tmpfile=$(mktemp)
    sed -e "s#__CANNBOT_SKILLS_ROOT__#$SKILL_DISCOVERY_ROOT#g" \
        -e "s#__CANNBOT_KNOWLEDGE_ROOT__#$CONFIG_ROOT/knowledge#g" \
        "$config_src" > "$tmpfile"
    mv "$tmpfile" "$CONFIG_FILE_TARGET"
    ok "$CONFIG_FILE_NAME -> $CONFIG_FILE_TARGET"

    # --- Step 3: manifest + health check ---
    step "[4/4] Health check ..."
    health_ok=true
    health_errors=""

    for name in $INCLUDED_SKILLS; do
        target="$SKILL_DISCOVERY_ROOT/$name"
        if [ -L "$target" ] && [ -e "$target/SKILL.md" ]; then
            :
        else
            health_errors="${health_errors}\n  ${RED}xx${NC} skill $name broken"
            health_ok=false
        fi
    done

    if [ ! -d "$CONFIG_ROOT/knowledge" ]; then
        health_errors="${health_errors}\n  ${YELLOW}[WARN]${NC} knowledge/ missing"
    fi

    [ -f "$CONFIG_FILE_TARGET" ] || { health_errors="${health_errors}\n  ${RED}xx${NC} $CONFIG_FILE_NAME missing"; health_ok=false; }

    SKILLS_JSON=$(printf '%s\n' $INCLUDED_SKILLS | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
    cat > "$MANIFEST" << MANIFEST_EOF
{
  "brand": "CANNBot",
  "plugin": "catlass-dsl-generator",
  "version": "$VERSION",
  "team": "plugins-community",
  "level": "$LEVEL",
  "tool": "$TOOL",
  "trae_variant": "${TRAE_VARIANT:-}",
  "installed_skills": $SKILLS_JSON,
  "knowledge": "$CONFIG_ROOT/knowledge",
  "skills_dir": "$SKILL_DISCOVERY_ROOT",
  "config_file": "$CONFIG_FILE_TARGET",
  "install_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
MANIFEST_EOF
    [ -f "$MANIFEST" ] || { health_errors="${health_errors}\n  ${RED}xx${NC} Manifest generation failed"; health_ok=false; }

    if [ "$health_ok" = true ] && [ -z "$health_errors" ]; then
        ok "All checks passed"
    else
        echo -e "$health_errors"
        [ "$health_ok" = true ] && warn "Some warnings, see above" || err "Some checks failed, see above"
    fi

    # --- Summary & Quick Start ---
    echo ""
    echo -e "  ${GREEN}${BOLD}catlass-dsl-generator installed successfully${NC}"
    echo ""
    echo -e "  ${BOLD}Quick Start:${NC}"
    case "$TOOL" in
        opencode) echo -e "  ${CYAN}1.${NC} 启动 CLI: opencode（或新开会话，$CONFIG_FILE_NAME 自动生效）" ;;
        claude)   echo -e "  ${CYAN}1.${NC} 启动 CLI: claude（或新开会话，$CONFIG_FILE_NAME 自动生效）" ;;
        trae)     echo -e "  ${CYAN}1.${NC} 在 Trae 中打开本项目（或新开会话，$CONFIG_FILE_NAME 自动生效）" ;;
        cursor)   echo -e "  ${CYAN}1.${NC} 通过 Cursor IDE 打开本项目（$CONFIG_FILE_NAME 自动生效）" ;;
        codex)    echo -e "  ${CYAN}1.${NC} 启动 CLI: codex（或新开会话，$CONFIG_FILE_NAME 自动生效）" ;;
        copilot)  echo -e "  ${CYAN}1.${NC} 通过 GitHub Copilot CLI / IDE 启动" ;;
        codearts) echo -e "  ${CYAN}1.${NC} 通过 CodeArts CLI / IDE 启动" ;;
    esac
    echo -e "  ${CYAN}2.${NC} 描述算子需求，例如："
    echo -e "     ${GREEN}${BOLD}使用 catlass-dsl-design 设计一个 QuantMatMul+GELU 融合算子：INT8 输入，per-token/per-channel 量化系数，tanh 近似 GELU${NC}"
    echo -e "  ${CYAN}3.${NC} 批准设计后：${GREEN}${BOLD}我批准 DESIGN.md，请使用 catlass-dsl-develop 按设计完成开发${NC}"
    echo ""
}

if [ "$ACTION" = "uninstall" ]; then
    uninstall
else
    install
fi
