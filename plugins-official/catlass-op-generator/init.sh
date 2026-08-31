#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
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

# Safe install config file with backup and conflict handling.
# $1 = generated temp file path
# $2 = target file path
# $3 = display name
# $4 = install level (global/project)
safe_install_file() {
    local tmpfile="$1"
    local target="$2"
    local name="$3"
    local level="$4"

    # Idempotency: skip if identical
    if [ -e "$target" ] && diff -q "$tmpfile" "$target" > /dev/null 2>&1; then
        info "$name already up to date"
        rm -f "$tmpfile"
        return 0
    fi

    # Backup existing file before overwriting
    if [ -e "$target" ] || [ -L "$target" ]; then
        local backup
        backup="${target}.bak.$(date +%Y%m%d_%H%M%S)"
        cp -a "$target" "$backup"
        warn "$name already exists, backed up to $(basename "$backup")"

        # Interactive prompt for global mode
        if [ "$level" = "global" ] && [ -t 0 ] && [ -t 1 ]; then
            echo ""
            echo -e "  ${BOLD}${YELLOW}⚠  $name 存在自定义内容，请选择操作：${NC}"
            echo -e "    ${BOLD}[O]${NC} 覆盖      - 用插件内容替换（原内容已备份）"
            echo -e "    ${BOLD}[M]${NC} 合并      - 插件内容置顶，保留原自定义内容"
            echo -e "    ${BOLD}[S]${NC} 跳过      - 保持现有文件不变"
            printf "  ${BOLD}${CYAN}→${NC} ${BOLD}请输入选择 [O/M/S]:${NC} "
            read -r choice < /dev/tty
            case "$choice" in
                [Mm]*)
                    cat "$tmpfile" > "${target}.new"
                    echo "" >> "${target}.new"
                    echo "<!-- === User custom content below === -->" >> "${target}.new"
                    echo "" >> "${target}.new"
                    cat "$target" >> "${target}.new"
                    mv "${target}.new" "$target"
                    ok "$name (merged with backup)"
                    rm -f "$tmpfile"
                    return 0
                    ;;
                [Ss]*)
                    info "$name skipped (backup preserved)"
                    rm -f "$tmpfile"
                    return 0
                    ;;
                *) ;; # default: overwrite
            esac
        fi
    fi

    # Overwrite (default for project mode or non-interactive)
    mv "$tmpfile" "$target"
    if [ "$level" = "global" ]; then
        ok "$name (absolute paths for global mode)"
    else
        ok "$name (absolute paths for project mode)"
    fi
}


BRAND="cannbot"
VERSION="1.0.0"
ASC_DEVKIT_REPO="https://gitcode.com/cann/asc-devkit.git"
ASC_DEVKIT_REF="master"

# --- Plugin-specific filters ---
EXCLUDED_SKILL=""
# Skill whitelist (space-separated list) - references shared ops/.
# Catlass plugin pulls catlass-op-{design,develop,perf-tune} plus the common
# Ascend C kernel skills used by architect/developer/reviewer subagents.
INCLUDED_SKILLS="catlass-op-design catlass-op-develop catlass-op-perf-tune ascendc-tiling-design npu-arch ascendc-api-best-practices ops-precision-standard ascendc-docs-search ascendc-env-check ascendc-precision-debug ops-profiling ascendc-runtime-debug ascendc-code-review torch-ascendc-op-extension"
# Agent whitelist (shell pattern) - uses local agents/
INCLUDED_AGENT_PATTERN="catlass-op-*"

# Detect TRAE variant by scanning global config directories.
# Sets global: TRAE_VARIANT=(ide|plugin|cli|unknown)
detect_trae_variant() {
    if [ -d "$HOME/.trae" ]; then
        TRAE_VARIANT="ide"
    elif [ -d "$HOME/.marscode" ]; then
        TRAE_VARIANT="plugin"
    elif [ -d "$HOME/.traecli" ]; then
        TRAE_VARIANT="cli"
    else
        TRAE_VARIANT="unknown"
    fi
}

show_banner() {
  echo ""
  echo -e "${CYAN}"
  cat << 'BANNER'
   ____    _    _   _ _   _ ____        _
  / ___|  / \  | \ | | \ | | __ )  ___ | |_
 | |     / _ \ |  \| |  \| |  _ \ / _ \| __|
 | |___ / ___ \| |\  | |\  | |_) | (_) | |_
  \____/_/   \_\_| \_|_| \_|____/ \___/ \__|
BANNER
  echo -e "${NC}"
  echo -e "  ${BOLD}Catlass Ascend C Kernel Dev Team${NC}"
  echo ""
}

show_help() {
    cat << EOF
CANNBot - Catlass Ascend C Kernel Development Environment Installer

Usage: init.sh [level] [tool] [install_path]

Arguments:
  level        - Installation level: "project" (default) or "global"
  tool         - Target tool: "opencode" (default), "claude", "trae", "cursor", "codex", "copilot", or "codearts"
  install_path - Project-level installation directory (default: current working directory)

Options:
  --help  - Show this help message

Examples:
  init.sh                      # Project-level, OpenCode
  init.sh project opencode     # Project-level, OpenCode
  init.sh global claude        # Global-level, Claude Code
  init.sh project claude       # Project-level, Claude Code
  init.sh project trae         # Project-level, Trae
  init.sh project cursor       # Project-level, Cursor
  init.sh project codex        # Project-level, Codex
  init.sh global codex         # Global-level, Codex
  init.sh project copilot      # Project-level, Copilot
  init.sh global copilot       # Global-level, Copilot
  init.sh project codearts     # Project-level, CodeArts
  init.sh project opencode /path/to/proj  # Project-level, OpenCode, custom path

Installation paths (CANNBot brand):
  OpenCode:     .opencode/{skills,agents}/     (auto-discovered)
  Claude:       .claude/{skills,agents}/       (per-skill symlinks auto-created)
  Trae IDE:     .trae/{skills,agents}/         (symlinks, project-level only)
  Trae Plugin:  .marscode/{skills,agents}/     (symlinks, project-level only)
  Trae CLI:     .traecli/{skills,agents}/      (symlinks, project-level only)
  Cursor:       .cursor/{skills,agents}/     + AGENTS.md in project root
  Codex:        .agents/skills/ + .codex/agents/ + AGENTS.md in project root
                ~/.agents/skills/ + ~/.codex/{agents,AGENTS.md} (global)
  Copilot:      .github/{skills,agents}/       (symlinks, project-level)
                ~/.copilot/{skills,agents}/    (symlinks, global)
  CodeArts:     .codeartsdoer/{skills,agents}/  (symlinks, project-level)
                ~/.codeartsdoer/{skills,agents}/ (symlinks, global)

After installation, launch directly:
  OpenCode: opencode
  Claude:   claude
  Trae:     通过 CLI 或 IDE 启动
  Cursor:   通过 Cursor IDE 启动
  Codex:    codex
  Copilot:  通过 GitHub Copilot CLI / IDE 启动
  CodeArts: 通过 CodeArts CLI / IDE 启动
EOF
}

LEVEL="project"
TOOL="opencode"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$SCRIPT_DIR"
# Agents: use local agents/ directory (migrated with plugin)
LOCAL_AGENT_ROOT="$PLUGIN_ROOT/agents"
# Codex custom agents use standalone TOML definitions.
CODEX_AGENT_ROOT="$LOCAL_AGENT_ROOT/codex"
# Skills: reference shared ops directory
if [ -d "$PLUGIN_ROOT/../../ops" ]; then
    SHARED_SKILL_ROOT="$(cd "$PLUGIN_ROOT/../../ops" && pwd)"
else
    SHARED_SKILL_ROOT=""
fi

for arg in "$@"; do
    case "$arg" in
        --help)            show_help; exit 0 ;;
        global|project)    LEVEL="$arg" ;;
        opencode|claude|trae|cursor|codex|copilot|codearts)   TOOL="$arg" ;;
    esac
done

# If last argument is not a known keyword, treat it as install_path (used by ST sandbox)
if [ $# -gt 0 ]; then
    last_arg="${!#}"
    case "$last_arg" in
        --help|global|project|opencode|claude|trae|cursor|codex|copilot|codearts) ;;
        *) INSTALL_PATH="$last_arg" ;;
    esac
fi

# Determine config root directory
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
    elif [ "$TOOL" = "codex" ]; then
        CONFIG_ROOT="$HOME/.codex"
    elif [ "$TOOL" = "codearts" ]; then
        CONFIG_ROOT="$HOME/.codeartsdoer"
    else
        CONFIG_ROOT="$HOME/.claude"
    fi
else
    if [ -n "$INSTALL_PATH" ]; then
        CONFIG_ROOT_BASE="$(cd "$INSTALL_PATH" && pwd)"
    else
        CONFIG_ROOT_BASE="$PWD"
    fi
    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.opencode"
    elif [ "$TOOL" = "trae" ]; then
        detect_trae_variant
        case "$TRAE_VARIANT" in
            plugin) CONFIG_ROOT="$CONFIG_ROOT_BASE/.marscode" ;;
            cli)    CONFIG_ROOT="$CONFIG_ROOT_BASE/.traecli" ;;
            *)      CONFIG_ROOT="$CONFIG_ROOT_BASE/.trae" ;;
        esac
    elif [ "$TOOL" = "copilot" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.github"
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.cursor"
    elif [ "$TOOL" = "codex" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.codex"
    elif [ "$TOOL" = "codearts" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.codeartsdoer"
    else
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.claude"
    fi
fi

CANNBOT_DIR="$CONFIG_ROOT"
SKILL_DISCOVERY_ROOT="$CONFIG_ROOT/skills"
AGENT_DISCOVERY_ROOT="$CONFIG_ROOT/agents"
if [ "$TOOL" = "codex" ]; then
    if [ "$LEVEL" = "global" ]; then
        SKILL_DISCOVERY_ROOT="$HOME/.agents/skills"
    else
        SKILL_DISCOVERY_ROOT="$CONFIG_ROOT_BASE/.agents/skills"
    fi
fi

# Clean up legacy cannbot subdirectory from previous installations
if [ -e "$CONFIG_ROOT/$BRAND" ] || [ -L "$CONFIG_ROOT/$BRAND" ]; then
    rm -rf "$CONFIG_ROOT/$BRAND"
fi
# OpenCode: also clean legacy teams link
if [ "$TOOL" = "opencode" ] && [ -L "$CONFIG_ROOT/teams" ]; then
    rm -f "$CONFIG_ROOT/teams"
fi

show_banner
echo "  Tool:      $TOOL"
echo "  Level:     $LEVEL"
echo "  Path:      $CONFIG_ROOT"
if [ "$TOOL" = "codex" ]; then
    echo "  Skills:    $SKILL_DISCOVERY_ROOT"
    echo "  Subagents: $AGENT_DISCOVERY_ROOT"
fi
echo ""

if [ "$TOOL" = "trae" ]; then
    case "$TRAE_VARIANT" in
        ide)
            info "Detected: TRAE IDE (.trae)"
            ;;
        plugin)
            info "Detected: TRAE Plugin (.marscode)"
            ;;
        cli)
            info "Detected: TRAE CLI (.traecli)"
            ;;
        unknown)
            warn "TRAE variant not detected; defaulting to IDE path"
            warn "If you use TRAE Plugin, ensure ~/.marscode exists before re-running"
            warn "If you use TRAE CLI, ensure ~/.traecli exists before re-running"
            ;;
    esac
    echo ""
fi

# --- Step 0: Confirmation before installation ---
step "[0/5] Checking items to be installed..."

# Collect skills to install (from shared ops)
SKILLS_TO_INSTALL=""
SKILL_COUNT=0
for skill_dir in "$SHARED_SKILL_ROOT"/*/; do
    [ -d "$skill_dir" ] || continue
    name=$(basename "$skill_dir")
    echo "$INCLUDED_SKILLS" | grep -qw "$name" || continue
    [ -n "$EXCLUDED_SKILL" ] && [ "$name" = "$EXCLUDED_SKILL" ] && continue
    SKILLS_TO_INSTALL="$SKILLS_TO_INSTALL $name"
    SKILL_COUNT=$((SKILL_COUNT + 1))
done

# Collect agents to install (from local agents/)
AGENTS_TO_INSTALL=""
AGENT_COUNT=0
AGENT_SOURCE_ROOT="$LOCAL_AGENT_ROOT"
if [ "$TOOL" = "codex" ]; then
    AGENT_SOURCE_ROOT="$CODEX_AGENT_ROOT"
fi
for agent_entry in "$AGENT_SOURCE_ROOT"/*; do
    [ -e "$agent_entry" ] || continue
    name=$(basename "$agent_entry")
    base="${name%.*}"
    [[ "$base" != $INCLUDED_AGENT_PATTERN ]] && continue
    AGENTS_TO_INSTALL="$AGENTS_TO_INSTALL $name"
    AGENT_COUNT=$((AGENT_COUNT + 1))
done

# Display installation plan
echo ""
echo -e "${BOLD}以下内容将被安装/替换：${NC}"
echo ""

if [ "$SKILL_COUNT" -gt 0 ]; then
    echo -e "${CYAN}Skills (${SKILL_COUNT} 项)：${NC}"
    for name in $SKILLS_TO_INSTALL; do
        target="$SKILL_DISCOVERY_ROOT/$name"
        if [ -e "$target" ] || [ -L "$target" ]; then
            echo -e "  ${YELLOW}$name${NC}"
        else
            echo -e "  ${GREEN}$name${NC}"
        fi
    done
    echo ""
fi

if [ "$AGENT_COUNT" -gt 0 ]; then
    echo -e "${CYAN}Agents (${AGENT_COUNT} 项)：${NC}"
    if [ "$TOOL" = "codex" ]; then
        if [ -L "$AGENT_DISCOVERY_ROOT" ] || [ ! -e "$AGENT_DISCOVERY_ROOT" ] || \
           { [ -d "$AGENT_DISCOVERY_ROOT" ] && [ -z "$(ls -A "$AGENT_DISCOVERY_ROOT")" ]; }; then
            echo -e "  ${GREEN}agents/${NC} → 将创建目录软连接到 ${CODEX_AGENT_ROOT}"
        else
            echo -e "  ${YELLOW}agents/${NC} → 目录已有内容，将保留原内容并安装兼容 TOML 文件"
        fi
        echo -e "    ${DIM}目标路径: $AGENT_DISCOVERY_ROOT${NC}"
        for name in $AGENTS_TO_INSTALL; do
            echo -e "    ${DIM}- $name${NC}"
        done
    else
        for name in $AGENTS_TO_INSTALL; do
            target="$AGENT_DISCOVERY_ROOT/$name"
            if [ -e "$target" ] || [ -L "$target" ]; then
                echo -e "  ${YELLOW}$name${NC}"
            else
                echo -e "  ${GREEN}$name${NC}"
            fi
        done
    fi
    echo ""
fi

echo -e "${CYAN}配置文件：${NC}"
if [ "$LEVEL" = "project" ]; then
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "codex" ] || [ "$TOOL" = "copilot" ] || [ "$TOOL" = "codearts" ]; then
        config_target="$CONFIG_ROOT_BASE/AGENTS.md"
    else
        config_target="$CONFIG_ROOT_BASE/CLAUDE.md"
    fi
else
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "codex" ] || [ "$TOOL" = "copilot" ] || [ "$TOOL" = "codearts" ]; then
        config_target="$CONFIG_ROOT/AGENTS.md"
    else
        config_target="$CONFIG_ROOT/CLAUDE.md"
    fi
fi
config_src="$PLUGIN_ROOT/AGENTS.md"
if { [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "codex" ] || [ "$TOOL" = "copilot" ] || [ "$TOOL" = "codearts" ]; } && [ "$LEVEL" = "project" ] && [ "$PLUGIN_ROOT" = "$CONFIG_ROOT_BASE" ]; then
    echo -e "  ${GREEN}$(basename "$config_target")${NC} (已存在，无需操作)"
elif [ -e "$config_target" ] || [ -L "$config_target" ]; then
    echo -e "  ${YELLOW}$(basename "$config_target")${NC} (将被替换)"
else
    echo -e "  ${GREEN}$(basename "$config_target")${NC} (将创建)"
fi

echo ""
echo -e "${BOLD}${YELLOW}注意：仅替换上述白名单内的内容，不影响其他已存在的 skills/agents${NC}"
echo ""
ok "开始安装..."
echo ""

# --- Step 1: Create directory symlinks ---
step "[1/5] Setting up CANNBot directory..."
mkdir -p "$CANNBOT_DIR"

step1_summary=""
step1_warns=""
if [ "$TOOL" = "opencode" ]; then
    # OpenCode: per-item symlinks for skills (from shared ops, whitelist filtered)
    mkdir -p "$CANNBOT_DIR/skills"
    # Pre-clean existing skill symlinks (only whitelist items)
    for skill_dir in "$SHARED_SKILL_ROOT"/*/; do
        [ -d "$skill_dir" ] || continue
        name=$(basename "$skill_dir")
        # Only clean skills that are in whitelist
        echo "$INCLUDED_SKILLS" | grep -qw "$name" || continue
        target="$CANNBOT_DIR/skills/$name"
        [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
    done
    skill_count=0
    for skill_dir in "$SHARED_SKILL_ROOT"/*/; do
        [ -d "$skill_dir" ] || continue
        name=$(basename "$skill_dir")
        # Check if skill is in whitelist (space-separated list)
        echo "$INCLUDED_SKILLS" | grep -qw "$name" || continue
        [ -n "$EXCLUDED_SKILL" ] && [ "$name" = "$EXCLUDED_SKILL" ] && continue
        ln -sfn "$(realpath "$skill_dir")" "$CANNBOT_DIR/skills/$name"
        skill_count=$((skill_count + 1))
    done
    step1_summary="skills(${skill_count}) "

    # OpenCode: per-item symlinks for agents (from local agents/, whitelist filtered)
    mkdir -p "$CANNBOT_DIR/agents"
    # Pre-clean existing agent symlinks (only whitelist items)
    for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
        [ -e "$agent_entry" ] || continue
        name=$(basename "$agent_entry")
        base_name="${name%.md}"
        # Only clean agents that match whitelist pattern
        [[ "$base_name" != $INCLUDED_AGENT_PATTERN ]] && continue
        target="$CANNBOT_DIR/agents/$name"
        [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
    done
    agent_count=0
    for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
        [ -e "$agent_entry" ] || continue
        name=$(basename "$agent_entry")
        base_name="${name%.md}"
        [[ "$base_name" != $INCLUDED_AGENT_PATTERN ]] && continue
        ln -sfn "$(realpath "$agent_entry")" "$CANNBOT_DIR/agents/$name"
        agent_count=$((agent_count + 1))
    done
    step1_summary="${step1_summary}agents(${agent_count})"
    ok "Linked: $step1_summary"
else
    # Trae/Claude/Cursor/Codex/Copilot: per-item symlinks handled in Step 3
    mkdir -p "$SKILL_DISCOVERY_ROOT"
    if [ "$TOOL" = "codex" ]; then
        # Keep agents/ absent so Step 3 can install it as a directory symlink.
        mkdir -p "$(dirname "$AGENT_DISCOVERY_ROOT")"
    else
        mkdir -p "$AGENT_DISCOVERY_ROOT"
    fi
    ok "Prepared: $SKILL_DISCOVERY_ROOT, $AGENT_DISCOVERY_ROOT"
fi
[ -n "$step1_warns" ] && echo -e "$step1_warns"
echo ""

# --- Step 2: Install config file (AGENTS.md / CLAUDE.md) ---
step "[2/5] Installing configuration..."

# Determine target path for config file
if [ "$LEVEL" = "project" ]; then
    # Project-level: config file should be in install base directory
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "codex" ] || [ "$TOOL" = "copilot" ] || [ "$TOOL" = "codearts" ]; then
        config_target="$CONFIG_ROOT_BASE/AGENTS.md"
    else
        config_target="$CONFIG_ROOT_BASE/CLAUDE.md"
    fi
else
    # Global-level: config file in CONFIG_ROOT
    mkdir -p "$CONFIG_ROOT"
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "codex" ] || [ "$TOOL" = "copilot" ] || [ "$TOOL" = "codearts" ]; then
        config_target="$CONFIG_ROOT/AGENTS.md"
    else
        config_target="$CONFIG_ROOT/CLAUDE.md"
    fi
fi

config_src="$PLUGIN_ROOT/AGENTS.md"

# Primary config symlink / copy
if { [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "codex" ] || [ "$TOOL" = "copilot" ] || [ "$TOOL" = "codearts" ]; } && [ "$LEVEL" = "project" ] && [ "$PLUGIN_ROOT" = "$CONFIG_ROOT_BASE" ]; then
    ok "$(basename "$config_target") already in current directory"
else
    if [ "$LEVEL" = "global" ]; then
        # Global mode: generate a copy with absolute paths so that
        # relative references (workflows/, asc-devkit/) work from any CWD.
        # Must remove existing symlink first, otherwise `>` would truncate
        # the symlink target (the original AGENTS.md) before sed reads it.
        PLUGIN_ROOT_ABS="$(realpath "$PLUGIN_ROOT")"
        ESCAPED_ROOT="$(echo "$PLUGIN_ROOT_ABS" | sed 's/#/\\#/g')"
        tmpfile=$(mktemp)
        sed \
          -e "s#bash workflows/scripts/#bash ${ESCAPED_ROOT}/workflows/scripts/#g" \
          -e "s#](workflows/#](${ESCAPED_ROOT}/workflows/#g" \
          -e "s#\`workflows/#\`${ESCAPED_ROOT}/workflows/#g" \
          -e "s#asc-devkit/docs/#${ESCAPED_ROOT}/asc-devkit/docs/#g" \
          -e "s#asc-devkit/examples/#${ESCAPED_ROOT}/asc-devkit/examples/#g" \
          "$config_src" > "$tmpfile"
        safe_install_file "$tmpfile" "$config_target" "AGENTS.md" "$LEVEL"
    else
        ln -sf "$config_src" "$config_target"
        ok "$(basename "$config_target")"
    fi
fi

# Also create config symlink in CONFIG_ROOT (for OpenCode/Trae discovery in .opencode/ / .trae/)
if { [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "codex" ] || [ "$TOOL" = "copilot" ] || [ "$TOOL" = "codearts" ]; } && [ "$LEVEL" = "project" ]; then
    if [ "$CONFIG_ROOT/AGENTS.md" != "$config_target" ]; then
        mkdir -p "$CONFIG_ROOT"
        ln -sf "$config_src" "$CONFIG_ROOT/AGENTS.md"
        ok "AGENTS.md → $(basename "$CONFIG_ROOT")/"
    fi
fi

# Link workflows directory
if [ -d "$PLUGIN_ROOT/workflows" ]; then
    mkdir -p "$CONFIG_ROOT"
    ln -sfn "$(realpath "$PLUGIN_ROOT/workflows")" "$CONFIG_ROOT/workflows"
    ok "workflows"
else
    warn "workflows/ not found, skipping"
fi
echo ""

# --- Step 3: Configure tool discovery ---
step "[3/5] Configuring tool discovery..."

if [ "$TOOL" = "opencode" ]; then
    # OpenCode: skills/ agents already at auto-scan paths, no extra discovery needed
    ok "Auto-scan: skills/, agents/"
else
    # Trae/Claude/Cursor/Codex/Copilot: create per-skill discovery symlinks (with filter, from shared ops)
    DISCOVERY="$SKILL_DISCOVERY_ROOT"

    # Pre-clean existing skills (only whitelist items)
    for skill_dir in "$SHARED_SKILL_ROOT"/*/; do
        [ -d "$skill_dir" ] || continue
        name=$(basename "$skill_dir")
        # Only clean skills that are in whitelist
        echo "$INCLUDED_SKILLS" | grep -qw "$name" || continue
        target="$DISCOVERY/$name"
        [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
    done

    link_count=0
    for skill_dir in "$SHARED_SKILL_ROOT"/*/; do
        [ -d "$skill_dir" ] || continue
        name=$(basename "$skill_dir")
        # Check if skill is in whitelist (space-separated list)
        echo "$INCLUDED_SKILLS" | grep -qw "$name" || continue
        [ -n "$EXCLUDED_SKILL" ] && [ "$name" = "$EXCLUDED_SKILL" ] && continue
        target="$DISCOVERY/$name"
        ln -sfn "$(realpath "$skill_dir")" "$target"
        link_count=$((link_count + 1))
    done

    # Clean broken symlinks
    for link in "$DISCOVERY"/*/; do
        link="${link%/}"
        [ -L "$link" ] && [ ! -e "$link" ] && rm "$link"
    done

    ok "Skills: $link_count discovery symlinks"

    # Trae/Claude/Cursor: also create agent discovery symlinks (from local agents/)
    AGENT_DISCOVERY="$AGENT_DISCOVERY_ROOT"

    agent_link_count=0
    if [ "$TOOL" = "codex" ]; then
        # Codex may ignore symlinked custom-agent TOML files (openai/codex#15345).
        # Install regular TOML files (with __CANNBOT_AGENT_SOURCE__ resolved) so
        # multiple plugins can coexist in the same .codex/agents/ directory.
        mkdir -p "$AGENT_DISCOVERY"
        for agent_entry in "$CODEX_AGENT_ROOT"/*.toml; do
            [ -f "$agent_entry" ] || continue
            name=$(basename "$agent_entry")
            base="${name%.toml}"
            [[ "$base" != $INCLUDED_AGENT_PATTERN ]] && continue
            canonical_agent="$LOCAL_AGENT_ROOT/$base.md"
            escaped_agent="$(echo "$canonical_agent" | sed 's/[&|\\]/\\&/g')"
            tmpfile=$(mktemp)
            sed "s|__CANNBOT_AGENT_SOURCE__|$escaped_agent|g" "$agent_entry" > "$tmpfile"
            safe_install_file "$tmpfile" "$AGENT_DISCOVERY/$name" "$name" "$LEVEL"
            agent_link_count=$((agent_link_count + 1))
        done
        ok "Agents: $agent_link_count compatible TOML files"
    else
        # Pre-clean existing agents (only whitelist items)
        for agent_entry in "$AGENT_SOURCE_ROOT"/*; do
            [ -e "$agent_entry" ] || continue
            name=$(basename "$agent_entry")
            base="${name%.*}"
            # Only clean agents that match whitelist pattern
            [[ "$base" != $INCLUDED_AGENT_PATTERN ]] && continue
            target="$AGENT_DISCOVERY/$name"
            [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
        done

        for agent_entry in "$AGENT_SOURCE_ROOT"/*; do
            [ -e "$agent_entry" ] || continue
            name=$(basename "$agent_entry")
            base="${name%.*}"
            [[ "$base" != $INCLUDED_AGENT_PATTERN ]] && continue
            target="$AGENT_DISCOVERY/$name"
            ln -sfn "$(realpath "$agent_entry")" "$target"
            agent_link_count=$((agent_link_count + 1))
        done

        # Clean broken symlinks
        for link in "$AGENT_DISCOVERY"/*; do
            [ -L "$link" ] && [ ! -e "$link" ] && rm "$link"
        done

        ok "Agents: $agent_link_count discovery symlinks"
    fi
fi
echo ""

# --- Step 4: Setup asc-devkit ---
step "[4/5] Setting up asc-devkit..."
ASC_DEVKIT_DIR="$SCRIPT_DIR/asc-devkit"

if [ -d "$ASC_DEVKIT_DIR" ]; then
    cd "$ASC_DEVKIT_DIR"
    git checkout . 2>/dev/null || true
    git fetch --quiet origin 2>/dev/null || warn "git fetch failed"
    git checkout --quiet "$ASC_DEVKIT_REF" 2>/dev/null || true
    git merge --quiet "origin/$ASC_DEVKIT_REF" 2>/dev/null || warn "git merge failed, using existing version"
    cd "$SCRIPT_DIR"
    ok "asc-devkit updated ($ASC_DEVKIT_REF)"
else
    git clone --quiet "$ASC_DEVKIT_REPO" "$ASC_DEVKIT_DIR" 2>/dev/null || warn "git clone failed, skipping asc-devkit"
    if [ -d "$ASC_DEVKIT_DIR" ]; then
        cd "$ASC_DEVKIT_DIR"
        git checkout --quiet "$ASC_DEVKIT_REF" 2>/dev/null || true
        cd "$SCRIPT_DIR"
        ok "asc-devkit cloned ($ASC_DEVKIT_REF)"
    fi
fi

if [ -d "$ASC_DEVKIT_DIR" ]; then
    python3 "$SHARED_SKILL_ROOT/ascendc-docs-search/scripts/clean_markdown.py" --dir "$ASC_DEVKIT_DIR" --no-backup --quiet > /dev/null 2>&1 || warn "markdown cleanup failed"
fi

# For global mode: also symlink asc-devkit into CONFIG_ROOT so it can be discovered
# from any working directory (not just the plugin directory)
if [ "$LEVEL" = "global" ] && [ -d "$ASC_DEVKIT_DIR" ]; then
    ln -sfn "$(realpath "$ASC_DEVKIT_DIR")" "$CONFIG_ROOT/asc-devkit"
    ok "asc-devkit → $CONFIG_ROOT/"
fi
echo ""

# --- Step 5: Health check ---
step "[5/5] Running health check..."
health_ok=true
health_errors=""

# Check discovery directories
for target in "$SKILL_DISCOVERY_ROOT" "$AGENT_DISCOVERY_ROOT"; do
  sub=$(basename "$target")
  if [ -d "$target" ]; then
    count=$(ls -d "$target"/* 2>/dev/null | wc -l)
    [ "$count" -eq 0 ] && { health_errors="${health_errors}\n  ${YELLOW}⚠${NC} $sub/ is empty"; }
  else
    health_errors="${health_errors}\n  ${RED}✗${NC} $sub/ missing"
    health_ok=false
  fi
done

# Check asc-devkit
if [ ! -d "$ASC_DEVKIT_DIR" ]; then
  health_errors="${health_errors}\n  ${YELLOW}⚠${NC} asc-devkit not available"
fi
# Check global asc-devkit symlink
if [ "$LEVEL" = "global" ] && [ ! -d "$CONFIG_ROOT/asc-devkit" ]; then
  health_errors="${health_errors}\n  ${YELLOW}⚠${NC} asc-devkit symlink missing in $CONFIG_ROOT"
fi

# Check asc-devkit API docs are searchable
if [ -d "$ASC_DEVKIT_DIR" ]; then
  FIND_API_DOC="$SHARED_SKILL_ROOT/ascendc-docs-search/scripts/find_api_doc.sh"
  if [ -f "$FIND_API_DOC" ]; then
    if ASC_DEVKIT_DIR="$ASC_DEVKIT_DIR" bash "$FIND_API_DOC" Add > /dev/null 2>&1; then
      ok "asc-devkit API docs searchable"
    else
      health_errors="${health_errors}\n  ${YELLOW}⚠${NC} asc-devkit API docs not searchable (find returned no results for 'Add')"
    fi
  fi
fi

# Check config file
if [ "$LEVEL" = "project" ]; then
    # Project-level: config file is in install base directory
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "codex" ] || [ "$TOOL" = "copilot" ] || [ "$TOOL" = "codearts" ]; then
        [ -f "$CONFIG_ROOT_BASE/AGENTS.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} AGENTS.md missing in install base directory"; health_ok=false; }
    else
        [ -f "$CONFIG_ROOT_BASE/CLAUDE.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} CLAUDE.md missing in install base directory"; health_ok=false; }
    fi
else
    # Global-level: config file in CONFIG_ROOT
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "codex" ] || [ "$TOOL" = "copilot" ] || [ "$TOOL" = "codearts" ]; then
        [ -f "$CONFIG_ROOT/AGENTS.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} AGENTS.md missing"; health_ok=false; }
    else
        [ -f "$CONFIG_ROOT/CLAUDE.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} CLAUDE.md missing"; health_ok=false; }
    fi
fi

# Generate brand manifest
MANIFEST="$CONFIG_ROOT/cannbot-manifest.json"

SKILLS_JSON="[]"
if [ -d "$SKILL_DISCOVERY_ROOT" ]; then
  SKILLS_JSON=$(ls -d "$SKILL_DISCOVERY_ROOT"/*/ 2>/dev/null | while read d; do
    d="${d%/}"; echo "${d##*/}"
  done | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
fi

AGENTS_JSON="[]"
if [ -d "$AGENT_DISCOVERY_ROOT" ]; then
  AGENTS_JSON=$(ls -d "$AGENT_DISCOVERY_ROOT"/* 2>/dev/null | while read d; do
    echo "${d##*/}"
  done | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
fi

cat > "$MANIFEST" << MANIFEST_EOF
{
  "brand": "CANNBot",
  "version": "$VERSION",
  "team": "$(basename "$SCRIPT_DIR")",
  "level": "$LEVEL",
  "tool": "$TOOL",
  "installed_skills": $SKILLS_JSON,
  "installed_agents": $AGENTS_JSON,
  "brand_dir": "$CONFIG_ROOT",
  "skills_dir": "$SKILL_DISCOVERY_ROOT",
  "agents_dir": "$AGENT_DISCOVERY_ROOT",
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

# --- Summary & Quick Start ---
echo ""
echo -e "  ${GREEN}${BOLD}✓ CANNBot installed successfully!${NC}"
echo ""
echo -e "  ${BOLD}Quick Start:${NC}"
CATLASS_CLONE='在工作区根准备 catlass 源码（与 operators/ 平级）：git clone https://gitcode.com/cann/catlass.git'
CATLASS_PROMPT='帮我开发一个 catlass_matmul_add 算子，A/B 为 fp16，C 为 fp32，目标 SoC Atlas A2，shape 主要是 M=N=K=512'
if [ "$TOOL" = "opencode" ]; then
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}opencode${NC}"
  echo -e "  ${CYAN}2.${NC} $CATLASS_CLONE"
  echo -e "  ${CYAN}3.${NC} 告诉 CANNBot: ${GREEN}${BOLD}${CATLASS_PROMPT}${NC}"
elif [ "$TOOL" = "codex" ]; then
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}codex${NC}"
  echo -e "  ${CYAN}2.${NC} $CATLASS_CLONE"
  echo -e "  ${CYAN}3.${NC} 告诉 CANNBot: ${GREEN}${BOLD}${CATLASS_PROMPT}${NC}"
elif [ "$TOOL" = "trae" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 CLI/IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} $CATLASS_CLONE"
  echo -e "  ${CYAN}3.${NC} 告诉 CANNBot: ${GREEN}${BOLD}${CATLASS_PROMPT}${NC}"
elif [ "$TOOL" = "cursor" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 Cursor IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} $CATLASS_CLONE"
  echo -e "  ${CYAN}3.${NC} 告诉 CANNBot: ${GREEN}${BOLD}${CATLASS_PROMPT}${NC}"
elif [ "$TOOL" = "copilot" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 GitHub Copilot CLI / IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} $CATLASS_CLONE"
  echo -e "  ${CYAN}3.${NC} 告诉 CANNBot: ${GREEN}${BOLD}${CATLASS_PROMPT}${NC}"
elif [ "$TOOL" = "codearts" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 CodeArts CLI / IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} $CATLASS_CLONE"
  echo -e "  ${CYAN}3.${NC} 告诉 CANNBot: ${GREEN}${BOLD}${CATLASS_PROMPT}${NC}"
else
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}claude${NC}"
  echo -e "  ${CYAN}2.${NC} $CATLASS_CLONE"
  echo -e "  ${CYAN}3.${NC} 告诉 CANNBot: ${GREEN}${BOLD}${CATLASS_PROMPT}${NC}"
fi
echo ""
