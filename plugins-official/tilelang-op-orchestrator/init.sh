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


# Detect TRAE variant and return appropriate config root.
# Detect TRAE variant by scanning global config directories.
# Sets global: TRAE_VARIANT=(ide|plugin|cli|unknown)
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

BRAND="cannbot"
VERSION="1.1.0"

# --- Plugin-specific filters ---
EXCLUDED_SKILL=""
# Skill whitelist (space-separated list) - references shared ops
INCLUDED_SKILLS="tilelang-api-best-practices tilelang-env-check tilelang-op-design tilelang-op-develop tilelang-op-test-design tilelang-perf-optimization tilelang-programming-model-guide tilelang-review tilelang-submodule-pull"
# Agent whitelist (shell pattern) - uses local agents/
INCLUDED_AGENT_PATTERN="tilelang-op-*"

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
  echo -e "  ${BOLD}TileLang Operator Dev Team${NC}"
  echo ""
}

show_help() {
    cat << EOF
CANNBot - TileLang Operator Development Environment Installer

Usage: init.sh [level] [tool] [install_path]

Arguments:
  level        - Installation level: "project" (default) or "global"
  tool         - Target tool: "opencode" (default), "claude", "trae", "cursor", or "copilot"
  install_path - Project-level installation directory (default: current working directory)

Options:
  --help  - Show this help message

Examples:
  init.sh                              # Project-level, OpenCode
  init.sh project opencode             # Project-level, OpenCode
  init.sh global claude                # Global-level, Claude Code
  init.sh project claude               # Project-level, Claude Code
  init.sh project trae                 # Project-level, Trae
  init.sh project cursor               # Project-level, Cursor
  init.sh project opencode /path/to/proj  # Project-level, OpenCode, custom path
  init.sh project trae /path/to/proj      # Project-level, Trae, custom path
  init.sh project cursor /path/to/proj    # Project-level, Cursor, custom path

Installation paths (CANNBot brand):
  OpenCode: .opencode/{skills,agents}/     + AGENTS.md in project root
  Claude:   .claude/{skills,agents}/ + CLAUDE.md in project root
  Trae IDE:     .trae/{skills,agents}/       + AGENTS.md in project root
  Trae Plugin:  .marscode/{skills,agents}/   + AGENTS.md in project root
  Trae CLI:     .traecli/{skills,agents}/    + AGENTS.md in project root
  Cursor:       .cursor/{skills,agents}/     + AGENTS.md in project root
  Copilot:      .github/{skills,agents}/      + AGENTS.md in project root (project)
                ~/.copilot/{skills,agents}/   + AGENTS.md (global)

After installation, launch directly:
  OpenCode: opencode
  Claude:   claude
  Trae:     通过 CLI 或 IDE 启动
  Cursor:   通过 Cursor IDE 启动
  Copilot:  通过 GitHub Copilot CLI / IDE 启动
EOF
}

LEVEL="project"
TOOL="opencode"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$SCRIPT_DIR"
# Agents: use local agents/ directory (migrated with plugin)
LOCAL_AGENT_ROOT="$PLUGIN_ROOT/agents"
# Skills: reference shared ops directory
SHARED_SKILL_ROOT="$(cd "$PLUGIN_ROOT/../../ops" && pwd)"

for arg in "$@"; do
    case "$arg" in
        --help)            show_help; exit 0 ;;
        global|project)    LEVEL="$arg" ;;
        opencode|claude|trae|cursor|copilot)   TOOL="$arg" ;;
    esac
done

# If last argument is not a known keyword, treat it as install_path
if [ $# -gt 0 ]; then
    last_arg="${!#}"
    case "$last_arg" in
        --help|global|project|opencode|claude|trae|cursor|copilot) ;;
        *) INSTALL_PATH="$last_arg" ;;
    esac
fi

# Determine config root directory
if [ "$LEVEL" = "global" ]; then
    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$HOME/.config/opencode"
    elif [ "$TOOL" = "trae" ] && [ "$LEVEL" = "project" ]; then
        detect_trae_variant
        case "$TRAE_VARIANT" in
            plugin) CONFIG_ROOT="$HOME/.marscode" ;;
            cli)    CONFIG_ROOT="$HOME/.traecli" ;;
            *)      CONFIG_ROOT="$HOME/.trae-cn" ;;
        esac
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_ROOT="$HOME/.cursor"
    elif [ "$TOOL" = "copilot" ]; then
        CONFIG_ROOT="$HOME/.copilot"
    else
        CONFIG_ROOT="$HOME/.claude"
    fi
else
    # Project-level: default to current directory, allow override via install_path arg
    if [ -n "$INSTALL_PATH" ]; then
        INSTALL_BASE="$(cd "$INSTALL_PATH" && pwd)"
        CONFIG_ROOT_BASE="$INSTALL_BASE"
    else
        INSTALL_BASE="$PWD"
        CONFIG_ROOT_BASE="$INSTALL_BASE"
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
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.cursor"
    elif [ "$TOOL" = "copilot" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.github"
    else
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.claude"
    fi
fi

CANNBOT_DIR="$CONFIG_ROOT"

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
echo ""

if [ "$TOOL" = "trae" ] && [ "$LEVEL" = "project" ]; then
    case "$TRAE_VARIANT" in
        ide)
            info "Detected: TRAE IDE (.trae-cn / .trae)"
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
for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
    [ -e "$agent_entry" ] || continue
    name=$(basename "$agent_entry")
    base="${name%.md}"
    [[ "$base" != $INCLUDED_AGENT_PATTERN ]] && continue
    AGENTS_TO_INSTALL="$AGENTS_TO_INSTALL $name"
    AGENT_COUNT=$((AGENT_COUNT + 1))
done

# Display installation plan
echo ""
echo -e "${BOLD}以下内容将被安装/替换：${NC}"
echo ""
echo -e "${CYAN}Skills (${SKILL_COUNT} 项，来自共享 ops 目录)：${NC}"
for name in $SKILLS_TO_INSTALL; do
    target="$CANNBOT_DIR/skills/$name"
    src="$SHARED_SKILL_ROOT/$name"
    if [ -e "$target" ] || [ -L "$target" ]; then
        echo -e "  ${YELLOW}$name${NC} → 将被替换为软连接到 ${src}"
    else
        echo -e "  ${GREEN}$name${NC} → 将创建软连接到 ${src}"
    fi
    echo -e "    ${DIM}目标路径: $target${NC}"
done

echo ""
echo -e "${CYAN}Agents (${AGENT_COUNT} 项，来自本地 agents/)：${NC}"
for name in $AGENTS_TO_INSTALL; do
    target="$CANNBOT_DIR/agents/$name"
    src="$LOCAL_AGENT_ROOT/$name"
    if [ -e "$target" ] || [ -L "$target" ]; then
        echo -e "  ${YELLOW}$name${NC} → 将被替换为软连接到 ${src}"
    else
        echo -e "  ${GREEN}$name${NC} → 将创建软连接到 ${src}"
    fi
    echo -e "    ${DIM}目标路径: $target${NC}"
done

echo ""
echo -e "${CYAN}配置文件：${NC}"
config_src="$PLUGIN_ROOT/AGENTS.md"
if [ "$TOOL" = "opencode" ]; then
    if [ "$LEVEL" = "project" ]; then
        config_target="$INSTALL_BASE/AGENTS.md"
    else
        config_target="$CONFIG_ROOT/AGENTS.md"
    fi
    if [ "$LEVEL" = "project" ] && [ "$PLUGIN_ROOT" = "$INSTALL_BASE" ]; then
        echo -e "  ${GREEN}AGENTS.md${NC} → 已存在于项目目录，无需创建软链接"
    elif [ -e "$config_target" ] || [ -L "$config_target" ]; then
        echo -e "  ${YELLOW}AGENTS.md${NC} → 将被替换为软连接到 ${config_src}"
    else
        echo -e "  ${GREEN}AGENTS.md${NC} → 将创建软连接到 ${config_src}"
    fi
    echo -e "    ${DIM}目标路径: $config_target${NC}"
elif [ "$TOOL" = "claude" ]; then
    if [ "$LEVEL" = "project" ]; then
        config_target="$INSTALL_BASE/CLAUDE.md"
    else
        config_target="$CONFIG_ROOT/CLAUDE.md"
    fi
    if [ -e "$config_target" ] || [ -L "$config_target" ]; then
        echo -e "  ${YELLOW}CLAUDE.md${NC} (将被替换)"
    else
        echo -e "  ${GREEN}CLAUDE.md${NC} (将创建)"
    fi
else
    if [ "$LEVEL" = "project" ]; then
        config_target="$INSTALL_BASE/AGENTS.md"
    else
        config_target="$CONFIG_ROOT/AGENTS.md"
    fi
    if [ -e "$config_target" ] || [ -L "$config_target" ]; then
        echo -e "  ${YELLOW}AGENTS.md${NC} (将被替换)"
    else
        echo -e "  ${GREEN}AGENTS.md${NC} (将创建)"
    fi
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
    # Claude/Trae/Cursor/Copilot: create directories (per-item symlinks handled in Step 3)
    mkdir -p "$CONFIG_ROOT/skills" "$CONFIG_ROOT/agents"
    ok "Prepared: skills/, agents/, rules/"
fi
[ -n "$step1_warns" ] && echo -e "$step1_warns"
echo ""

# --- Step 2: Install config file (AGENTS.md / CLAUDE.md) ---
step "[2/5] Installing configuration..."

config_src="$PLUGIN_ROOT/AGENTS.md"

if [ "$TOOL" = "opencode" ]; then
    # OpenCode: AGENTS.md in project root (or CONFIG_ROOT for global)
    if [ "$LEVEL" = "project" ]; then
        config_target="$INSTALL_BASE/AGENTS.md"
    else
        config_target="$CONFIG_ROOT/AGENTS.md"
    fi
    if [ "$LEVEL" = "project" ] && [ "$PLUGIN_ROOT" = "$INSTALL_BASE" ]; then
        ok "AGENTS.md already in project directory"
    else
        if [ "$LEVEL" = "global" ] || { [ "$LEVEL" = "project" ] && [ "$INSTALL_BASE" != "$SCRIPT_DIR" ]; }; then
            PLUGIN_ROOT_ABS="$(realpath "$PLUGIN_ROOT")"
            ESCAPED_ROOT="$(echo "$PLUGIN_ROOT_ABS" | sed 's/#/\\#/g')"
            ESCAPED_OPS="$(echo "$SHARED_SKILL_ROOT" | sed 's/#/\\#/g')"
            tmpfile=$(mktemp)
            sed \
              -e "s#tilelang-ascend/#${ESCAPED_ROOT}/tilelang-ascend/#g" \
              -e "s#\.\./\.\./ops/#${ESCAPED_OPS}/#g" \
              "$config_src" > "$tmpfile"
            safe_install_file "$tmpfile" "$config_target" "AGENTS.md" "$LEVEL"
        else
            ln -sf "$config_src" "$config_target"
            ok "AGENTS.md"
        fi
    fi
elif [ "$TOOL" = "claude" ]; then
    # Claude: CLAUDE.md in project root (or CONFIG_ROOT for global)
    if [ "$LEVEL" = "project" ]; then
        config_target="$INSTALL_BASE/CLAUDE.md"
    else
        config_target="$CONFIG_ROOT/CLAUDE.md"
    fi
    if [ "$config_src" = "$config_target" ]; then
        info "$(basename "$config_target") already at target location"
    elif [ "$LEVEL" = "global" ] || { [ "$LEVEL" = "project" ] && [ "$INSTALL_BASE" != "$SCRIPT_DIR" ]; }; then
        PLUGIN_ROOT_ABS="$(realpath "$PLUGIN_ROOT")"
        ESCAPED_ROOT="$(echo "$PLUGIN_ROOT_ABS" | sed 's/#/\\#/g')"
        ESCAPED_OPS="$(echo "$SHARED_SKILL_ROOT" | sed 's/#/\\#/g')"
        tmpfile=$(mktemp)
        sed \
          -e "s#tilelang-ascend/#${ESCAPED_ROOT}/tilelang-ascend/#g" \
          -e "s#\.\./\.\./ops/#${ESCAPED_OPS}/#g" \
          "$config_src" > "$tmpfile"
        safe_install_file "$tmpfile" "$config_target" "CLAUDE.md" "$LEVEL"
    else
        if [ -e "$config_target" ] && [ ! -L "$config_target" ]; then
            backup="${config_target}.bak.$(date +%Y%m%d_%H%M%S)"
            cp -a "$config_target" "$backup"
            warn "CLAUDE.md already exists, backed up to $(basename "$backup")"
        fi
        ln -sf "$config_src" "$config_target"
        ok "CLAUDE.md"
    fi
else
    # Cursor: AGENTS.md in project root (same as OpenCode)
    if [ "$LEVEL" = "project" ]; then
        config_target="$INSTALL_BASE/AGENTS.md"
    else
        config_target="$CONFIG_ROOT/AGENTS.md"
    fi
    if [ "$config_src" = "$config_target" ]; then
        info "$(basename "$config_target") already at target location"
    elif [ "$LEVEL" = "global" ] || { [ "$LEVEL" = "project" ] && [ "$INSTALL_BASE" != "$SCRIPT_DIR" ]; }; then
        PLUGIN_ROOT_ABS="$(realpath "$PLUGIN_ROOT")"
        ESCAPED_ROOT="$(echo "$PLUGIN_ROOT_ABS" | sed 's/#/\\#/g')"
        ESCAPED_OPS="$(echo "$SHARED_SKILL_ROOT" | sed 's/#/\\#/g')"
        tmpfile=$(mktemp)
        sed \
          -e "s#tilelang-ascend/#${ESCAPED_ROOT}/tilelang-ascend/#g" \
          -e "s#\.\./\.\./ops/#${ESCAPED_OPS}/#g" \
          "$config_src" > "$tmpfile"
        safe_install_file "$tmpfile" "$config_target" "AGENTS.md" "$LEVEL"
    else
        if [ -e "$config_target" ] && [ ! -L "$config_target" ]; then
            backup="${config_target}.bak.$(date +%Y%m%d_%H%M%S)"
            cp -a "$config_target" "$backup"
            warn "AGENTS.md already exists, backed up to $(basename "$backup")"
        fi
        ln -sf "$config_src" "$config_target"
        ok "AGENTS.md"
    fi
fi
echo ""

# --- Step 3: Configure tool discovery ---
step "[3/5] Configuring tool discovery..."

if [ "$TOOL" = "opencode" ]; then
    # OpenCode: skills/ agents already at auto-scan paths, no extra discovery needed
    ok "Auto-scan: skills/, agents/"
else
    # Claude/Trae/Cursor/Copilot: create per-skill discovery symlinks (with filter, from shared ops)
    DISCOVERY="$CONFIG_ROOT/skills"

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

    # Claude/Trae/Cursor: also create agent discovery symlinks (from local agents/)
    AGENT_DISCOVERY="$CONFIG_ROOT/agents"

    # Pre-clean existing agents (only whitelist items)
    for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
        [ -e "$agent_entry" ] || continue
        name=$(basename "$agent_entry")
        base="${name%.md}"
        # Only clean agents that match whitelist pattern
        [[ "$base" != $INCLUDED_AGENT_PATTERN ]] && continue
        target="$AGENT_DISCOVERY/$name"
        [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
    done

    agent_link_count=0
    for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
        [ -e "$agent_entry" ] || continue
        name=$(basename "$agent_entry")
        base="${name%.md}"
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
echo ""

# --- Step 4: Clone TileLang-Ascend source repository ---
step "[4/5] Preparing TileLang-Ascend source repository..."

TILELANG_DIR="$PLUGIN_ROOT/tilelang-ascend"
if [ -d "$TILELANG_DIR" ] && [ -d "$TILELANG_DIR/.git" ]; then
    ok "TileLang-Ascend already exists: $TILELANG_DIR"
else
    mkdir -p "$(dirname "$TILELANG_DIR")"
    if command -v git &> /dev/null; then
        info "Cloning TileLang-Ascend source repository..."
        if git clone --recursive https://github.com/tile-ai/tilelang-ascend.git "$TILELANG_DIR" 2>/dev/null; then
            ok "Cloned tilelang-ascend to $TILELANG_DIR"
        else
            warn "Clone failed — clone manually: git clone --recursive https://github.com/tile-ai/tilelang-ascend.git $TILELANG_DIR"
        fi
    else
        warn "git not found — install git and clone manually: git clone --recursive https://github.com/tile-ai/tilelang-ascend.git $TILELANG_DIR"
    fi
fi

# For global mode: also symlink tilelang-ascend into CONFIG_ROOT so it can be discovered
# from any working directory (not just the plugin directory)
if [ "$LEVEL" = "global" ] && [ -d "$TILELANG_DIR" ]; then
    ln -sfn "$(realpath "$TILELANG_DIR")" "$CONFIG_ROOT/tilelang-ascend"
    ok "tilelang-ascend → $CONFIG_ROOT/"
fi

# For project-level with custom target: also symlink tilelang-ascend into INSTALL_BASE
# so relative references from agents work correctly
if [ "$LEVEL" = "project" ] && [ -d "$TILELANG_DIR" ]; then
    if [ "$INSTALL_BASE" != "$SCRIPT_DIR" ]; then
        ln -sfn "$(realpath "$TILELANG_DIR")" "$INSTALL_BASE/tilelang-ascend"
        ok "tilelang-ascend → $INSTALL_BASE/"
    fi
fi
echo ""

# --- Step 5: Health check ---
step "[5/5] Running health check..."
health_ok=true
health_errors=""

# Check directory symlinks
for sub in skills agents; do
  target="$CANNBOT_DIR/$sub"
  if [ -d "$target" ]; then
    count=$(ls -d "$target"/* 2>/dev/null | wc -l)
    [ "$count" -eq 0 ] && { health_errors="${health_errors}\n  ${YELLOW}⚠${NC} $sub/ is empty"; }
  else
    health_errors="${health_errors}\n  ${RED}✗${NC} $sub/ missing"
    health_ok=false
  fi
done

# Check TileLang-Ascend source repository
if [ -d "$TILELANG_DIR/docs" ]; then
    ok "TileLang-Ascend source present"
else
    health_errors="${health_errors}\n  $(echo -e "${YELLOW}⚠${NC}") tilelang-ascend repo not found — API Explorer / Design skills need docs/"
fi
# Check global tilelang-ascend symlink
if [ "$LEVEL" = "global" ] && [ ! -d "$CONFIG_ROOT/tilelang-ascend" ]; then
  health_errors="${health_errors}\n  ${YELLOW}⚠${NC} tilelang-ascend symlink missing in $CONFIG_ROOT"
fi
# When installed to a custom directory, also check the symlink there
if [ "$LEVEL" = "project" ] && [ "$INSTALL_BASE" != "$SCRIPT_DIR" ] && [ ! -d "$INSTALL_BASE/tilelang-ascend" ]; then
  health_errors="${health_errors}\n  ${YELLOW}⚠${NC} tilelang-ascend symlink missing in $INSTALL_BASE"
fi

# Check config file
if [ "$TOOL" = "opencode" ]; then
    if [ "$LEVEL" = "project" ]; then
        [ -f "$INSTALL_BASE/AGENTS.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} AGENTS.md missing in project directory"; health_ok=false; }
    else
        [ -f "$CONFIG_ROOT/AGENTS.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} AGENTS.md missing"; health_ok=false; }
    fi
elif [ "$TOOL" = "claude" ]; then
    if [ "$LEVEL" = "project" ]; then
        [ -f "$INSTALL_BASE/CLAUDE.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} CLAUDE.md missing in project directory"; health_ok=false; }
    else
        [ -f "$CONFIG_ROOT/CLAUDE.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} CLAUDE.md missing"; health_ok=false; }
    fi
else
    if [ "$LEVEL" = "project" ]; then
        [ -f "$INSTALL_BASE/AGENTS.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} AGENTS.md missing in project directory"; health_ok=false; }
    else
        [ -f "$CONFIG_ROOT/AGENTS.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} AGENTS.md missing"; health_ok=false; }
    fi
fi

# Generate brand manifest
MANIFEST="$CONFIG_ROOT/cannbot-manifest.json"

SKILLS_JSON="[]"
if [ -d "$CANNBOT_DIR/skills" ]; then
  SKILLS_JSON=$(ls -d "$CANNBOT_DIR/skills"/*/ 2>/dev/null | while read d; do
    basename "$d"
  done | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
fi

AGENTS_JSON="[]"
if [ -d "$CANNBOT_DIR/agents" ]; then
  AGENTS_JSON=$(ls -d "$CANNBOT_DIR/agents"/* 2>/dev/null | while read d; do
    basename "$d"
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
if [ "$TOOL" = "opencode" ]; then
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}opencode${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}帮我写一个 softmax 算子${NC}"
elif [ "$TOOL" = "trae" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 CLI/IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}帮我写一个 softmax 算子${NC}"
elif [ "$TOOL" = "cursor" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 Cursor IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}帮我写一个 softmax 算子${NC}"
elif [ "$TOOL" = "copilot" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 GitHub Copilot CLI / IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}帮我写一个 softmax 算子${NC}"
else
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}claude${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}帮我写一个 softmax 算子${NC}"
fi
echo ""
