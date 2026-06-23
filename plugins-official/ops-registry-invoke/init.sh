#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#
# Team Dependency Installer
# Automatically scans team AGENTS.md, resolves all dependencies (skills + agents + agent skills),
# and installs only what this team needs.
#
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
VERSION="1.2.0"

# --- Plugin-specific filters ---
# Skill whitelist (space-separated list) - references shared ops + local workflow
INCLUDED_SKILLS="ascendc-blaze-best-practice ascendc-api-best-practices ascendc-code-review ascendc-crash-debug ascendc-direct-invoke-template ascendc-docs-gen ascendc-docs-search ascendc-env-check npu-arch ascendc-performance-best-practices ascendc-precision-debug ascendc-regbase-best-practice ascendc-registry-invoke-template ascendc-runtime-debug ascendc-st-design ascendc-tiling-design ascendc-ut-develop ops-precision-standard ops-profiling ops-registry-invoke-workflow ops-spec-gen gitcode-toolkit gitcode-pr-handler gitcode-issue-gen gitcode-issue-handler"
# Agent whitelist (shell pattern) - uses local agents/
INCLUDED_AGENT_PATTERN="ascendc-ops-*"

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAM_NAME="$(basename "$SCRIPT_DIR")"
PLUGIN_ROOT="$SCRIPT_DIR"
SHARED_SKILL_ROOT="$(cd "$SCRIPT_DIR/../../ops" && pwd)"
INFRA_SKILL_ROOT="$(cd "$PLUGIN_ROOT/../../infra" && pwd)"
LOCAL_AGENT_ROOT="$SCRIPT_DIR/agents"

# --- Infra skill helpers ---
collect_infra_skills() {
    INFRA_SKILL_NAMES=()
    for skill_dir in "$INFRA_SKILL_ROOT"/*/; do
        [ -d "$skill_dir" ] || continue
        local name
        name=$(basename "$skill_dir")
        echo "$INCLUDED_SKILLS" | grep -qw "$name" || continue
        INFRA_SKILL_NAMES+=("$name")
    done
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
  echo -e "  ${BOLD}Team Dependency Installer${NC}"
  echo ""
}

show_help() {
    cat << EOF
Team Dependency Installer - Auto-scan and install team dependencies

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
  init.sh project copilot              # Project-level, Copilot
  init.sh global copilot               # Global-level, Copilot
  init.sh project opencode /path/to/proj  # Project-level, OpenCode, custom path
  init.sh project trae /path/to/proj      # Project-level, Trae, custom path

Installation paths:
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

# --- Parse arguments ---
LEVEL="project"
TOOL="opencode"
INSTALL_PATH=""

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

# --- Determine config root ---
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

# --- Clean up legacy ---
if [ -e "$CONFIG_ROOT/$BRAND" ] || [ -L "$CONFIG_ROOT/$BRAND" ]; then
    rm -rf "$CONFIG_ROOT/$BRAND"
fi
if [ "$TOOL" = "opencode" ] && [ -L "$CONFIG_ROOT/teams" ]; then
    rm -f "$CONFIG_ROOT/teams"
fi

# ============================================================
# Dependency Resolution
# ============================================================

# Parse YAML list from AGENTS.md (handles `skills:` and `agents:` sections)
parse_yaml_list() {
    local file="$1"
    local key="$2"
    awk -v key="$key" '
        $0 ~ "^"key":" { flag=1; next }
        /^[^ ]/ { flag=0 }
        flag && /^ +- / { sub(/^ +- /, ""); print }
    ' "$file" 2>/dev/null | grep -v '^$' || true
}

# Resolve agent source path (from local agents/ directory)
resolve_agent_src() {
    local agent="$1"
    if [ -d "$LOCAL_AGENT_ROOT/$agent" ]; then
        echo "$LOCAL_AGENT_ROOT/$agent"
    elif [ -f "$LOCAL_AGENT_ROOT/$agent.md" ]; then
        echo "$LOCAL_AGENT_ROOT/$agent.md"
    else
        echo ""
    fi
}

# Resolve skill source path (shared ops/ first, then infra/, then team-local)
# For team-local skills, searches subdirectories for a SKILL.md with matching `name:`.
resolve_skill_src() {
    local skill="$1"
    if [ -d "$SHARED_SKILL_ROOT/$skill" ]; then
        echo "$SHARED_SKILL_ROOT/$skill"
        return
    fi
    if [ -d "$INFRA_SKILL_ROOT/$skill" ]; then
        echo "$INFRA_SKILL_ROOT/$skill"
        return
    fi
    # Search team-local subdirectories for SKILL.md with matching name
    for dir in "$SCRIPT_DIR"/*/; do
        [ -d "$dir" ] || continue
        if [ -f "$dir/SKILL.md" ]; then
            local name
            name=$(awk '/^name:/{print $2; exit}' "$dir/SKILL.md" 2>/dev/null)
            if [ "$name" = "$skill" ]; then
                echo "$dir"
                return
            fi
        fi
    done
    echo ""
}

# Collect all dependencies
resolve_dependencies() {
    local agents_file="$SCRIPT_DIR/AGENTS.md"
    if [ ! -f "$agents_file" ]; then
        err "AGENTS.md not found in $SCRIPT_DIR"
        exit 1
    fi

    # Direct dependencies from team AGENTS.md
    local direct_skills=$(parse_yaml_list "$agents_file" "skills")
    local direct_agents=$(parse_yaml_list "$agents_file" "agents")

    # Collect all skills (direct + transitive from agents)
    local all_skills="$direct_skills"
    local agent_skills_map=""

    for agent in $direct_agents; do
        local agent_file=""
        # Try AGENT.md first, then {agent_name}.md (from local agents/)
        if [ -f "$LOCAL_AGENT_ROOT/$agent/AGENT.md" ]; then
            agent_file="$LOCAL_AGENT_ROOT/$agent/AGENT.md"
        elif [ -f "$LOCAL_AGENT_ROOT/$agent/$agent.md" ]; then
            agent_file="$LOCAL_AGENT_ROOT/$agent/$agent.md"
        elif [ -f "$LOCAL_AGENT_ROOT/$agent.md" ]; then
            agent_file="$LOCAL_AGENT_ROOT/$agent.md"
        fi

        if [ -n "$agent_file" ]; then
            local skills=$(parse_yaml_list "$agent_file" "skills")
            if [ -n "$skills" ]; then
                all_skills="$all_skills
$skills"
                agent_skills_map="$agent_skills_map
  $agent → $(echo $skills | tr '\n' ', ' | sed 's/,$//')"
            fi
        else
            warn "Agent config not found: $agent (expected AGENT.md or $agent.md)"
        fi
    done

    # Deduplicate
    DIRECT_SKILLS=$(echo "$direct_skills" | sort -u | grep -v '^$' || true)
    DIRECT_AGENTS=$(echo "$direct_agents" | sort -u | grep -v '^$' || true)
    ALL_SKILLS=$(echo "$all_skills" | sort -u | grep -v '^$' || true)
    AGENT_SKILLS_MAP="$agent_skills_map"

    # Counts
    DIRECT_SKILL_COUNT=$(echo "$DIRECT_SKILLS" | grep -c '.' || echo 0)
    DIRECT_AGENT_COUNT=$(echo "$DIRECT_AGENTS" | grep -c '.' || echo 0)
    ALL_SKILL_COUNT=$(echo "$ALL_SKILLS" | grep -c '.' || echo 0)
}

# ============================================================
# Main
# ============================================================

show_banner
echo "  Team:      $TEAM_NAME"
echo "  Tool:      $TOOL"
echo "  Level:     $LEVEL"
echo "  Path:      $CONFIG_ROOT"
echo ""

if [ "$TOOL" = "trae" ]; then
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

# Resolve dependencies
step "[1/7] Resolving team dependencies..."
resolve_dependencies

echo -e "  ${BOLD}Direct skills ($DIRECT_SKILL_COUNT):${NC}"
for s in $DIRECT_SKILLS; do echo -e "    ${GREEN}•${NC} $s"; done

echo -e "  ${BOLD}Direct agents ($DIRECT_AGENT_COUNT):${NC}"
for a in $DIRECT_AGENTS; do echo -e "    ${GREEN}•${NC} $a"; done

if [ -n "$AGENT_SKILLS_MAP" ]; then
    echo -e "  ${BOLD}Agent skills (transitive):${NC}"
    echo -e "$AGENT_SKILLS_MAP" | while IFS= read -r line; do
        [ -n "$line" ] && echo -e "    ${CYAN}→${NC}${DIM}$line${NC}"
    done
fi

echo -e "  ${BOLD}Total skills to install: $ALL_SKILL_COUNT${NC}"
echo ""

# Collect infra skills to install (from infra/)
collect_infra_skills
INFRA_SKILLS_TO_INSTALL="${INFRA_SKILL_NAMES[*]}"
INFRA_SKILL_COUNT=${#INFRA_SKILL_NAMES[@]}

if [ "$INFRA_SKILL_COUNT" -gt 0 ]; then
    echo -e "${CYAN}Infra Skills (${INFRA_SKILL_COUNT} 项)：${NC}"
    for name in $INFRA_SKILLS_TO_INSTALL; do
        target="$CANNBOT_DIR/skills/$name"
        if [ -e "$target" ] || [ -L "$target" ]; then
            echo -e "  ${YELLOW}$name${NC}"
        else
            echo -e "  ${GREEN}$name${NC}"
        fi
    done
    echo ""
fi

# --- Step 2: Verify dependencies ---
step "[2/7] Verifying dependencies..."
missing_skills=""
for skill in $ALL_SKILLS; do
    src=$(resolve_skill_src "$skill")
    if [ -z "$src" ]; then
        missing_skills="$missing_skills $skill"
    fi
done

if [ -n "$missing_skills" ]; then
    warn "Missing skills:$missing_skills"
    warn "Install these skills manually or verify names in AGENTS.md"
else
    ok "All dependencies already present"
fi
echo ""

# --- Step 3: Create directory symlinks ---
step "[3/7] Setting up CANNBot directory..."
mkdir -p "$CANNBOT_DIR"

if [ "$TOOL" = "opencode" ]; then
    # OpenCode: directory-level symlink for skills (auto-scan)
    # Create a temp dir with only the needed skills, then symlink it
    SKILLS_LINK_DIR="$CANNBOT_DIR/skills"
    rm -rf "$SKILLS_LINK_DIR"
    mkdir -p "$SKILLS_LINK_DIR"

    skill_count=0
    for skill in $ALL_SKILLS; do
        src=$(resolve_skill_src "$skill")
        if [ -n "$src" ]; then
            ln -sfn "$(realpath "$src")" "$SKILLS_LINK_DIR/$skill"
            skill_count=$((skill_count + 1))
        else
            warn "Skill not found: $skill"
        fi
    done
    ok "Skills: $skill_count linked"

    # OpenCode: per-item symlinks for agents
    AGENTS_LINK_DIR="$CANNBOT_DIR/agents"
    rm -rf "$AGENTS_LINK_DIR"
    mkdir -p "$AGENTS_LINK_DIR"

    agent_count=0
    for agent in $DIRECT_AGENTS; do
        src=$(resolve_agent_src "$agent")
        if [ -n "$src" ]; then
            link_name=$(basename "$src")
            ln -sfn "$(realpath "$src")" "$AGENTS_LINK_DIR/$link_name"
            agent_count=$((agent_count + 1))
        else
            warn "Agent not found: $agent"
        fi
    done
    ok "Agents: $agent_count linked"
else
    # Trae/Claude/Cursor/Copilot: create directories (per-item symlinks handled in Step 5)
    mkdir -p "$CONFIG_ROOT/skills" "$CONFIG_ROOT/agents"
    ok "Prepared: skills/, agents/, rules/"
fi
echo ""

# --- Step 4: Install config file ---
step "[4/7] Installing configuration..."
mkdir -p "$CONFIG_ROOT"

# Link team-local workflow skill
link_workflow_skill() {
    if [ -d "$SCRIPT_DIR/workflow" ]; then
        ln -sfn "$(realpath "$SCRIPT_DIR/workflow")" "$CONFIG_ROOT/skills/ops-registry-invoke-workflow"
        ok "workflow/ → skills/ops-registry-invoke-workflow"
    fi
}

config_src="$SCRIPT_DIR/AGENTS.md"

if [ "$TOOL" = "opencode" ]; then
    # OpenCode: AGENTS.md in project root (or CONFIG_ROOT for global)
    if [ "$LEVEL" = "project" ]; then
        config_target="$INSTALL_BASE/AGENTS.md"
    else
        config_target="$CONFIG_ROOT/AGENTS.md"
    fi
    if [ "$config_src" = "$config_target" ]; then
        info "$(basename "$config_target") already at target location"
    elif [ "$LEVEL" = "global" ] || { [ "$LEVEL" = "project" ] && [ "$INSTALL_BASE" != "$SCRIPT_DIR" ]; }; then
        PLUGIN_ROOT_ABS="$(realpath "$SCRIPT_DIR")"
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
        ok "AGENTS.md"
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
        PLUGIN_ROOT_ABS="$(realpath "$SCRIPT_DIR")"
        ESCAPED_ROOT="$(echo "$PLUGIN_ROOT_ABS" | sed 's/#/\\#/g')"
        tmpfile=$(mktemp)
        sed \
          -e "s#bash workflows/scripts/#bash ${ESCAPED_ROOT}/workflows/scripts/#g" \
          -e "s#](workflows/#](${ESCAPED_ROOT}/workflows/#g" \
          -e "s#\`workflows/#\`${ESCAPED_ROOT}/workflows/#g" \
          -e "s#asc-devkit/docs/#${ESCAPED_ROOT}/asc-devkit/docs/#g" \
          -e "s#asc-devkit/examples/#${ESCAPED_ROOT}/asc-devkit/examples/#g" \
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
    # Trae/Cursor: AGENTS.md in project root (same as OpenCode)
    if [ "$LEVEL" = "project" ]; then
        config_target="$INSTALL_BASE/AGENTS.md"
    else
        config_target="$CONFIG_ROOT/AGENTS.md"
    fi
    if [ "$config_src" = "$config_target" ]; then
        info "$(basename "$config_target") already at target location"
    elif [ "$LEVEL" = "global" ] || { [ "$LEVEL" = "project" ] && [ "$INSTALL_BASE" != "$SCRIPT_DIR" ]; }; then
        PLUGIN_ROOT_ABS="$(realpath "$SCRIPT_DIR")"
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
        if [ -e "$config_target" ] && [ ! -L "$config_target" ]; then
            backup="${config_target}.bak.$(date +%Y%m%d_%H%M%S)"
            cp -a "$config_target" "$backup"
            warn "AGENTS.md already exists, backed up to $(basename "$backup")"
        fi
        ln -sf "$config_src" "$config_target"
        ok "AGENTS.md"
    fi
fi

link_workflow_skill

echo ""

# --- Step 5: Configure tool discovery ---
step "[5/7] Configuring tool discovery..."

if [ "$TOOL" = "opencode" ]; then
    ok "Auto-scan: skills/, agents/"
else
    # Trae/Claude/Cursor/Copilot: create per-skill discovery symlinks
    DISCOVERY="$CONFIG_ROOT/skills"
    link_count=0
    for skill in $ALL_SKILLS; do
        src=$(resolve_skill_src "$skill")
        [ -n "$src" ] || continue
        ln -sfn "$(realpath "$src")" "$DISCOVERY/$skill"
        link_count=$((link_count + 1))
    done
    ok "Skills: $link_count discovery symlinks"

    # Claude/Cursor: agent discovery symlinks
    AGENT_DISCOVERY="$CONFIG_ROOT/agents"
    # Pre-clean existing agent symlinks (both with and without .md)
    for agent in $DIRECT_AGENTS; do
        rm -f "$AGENT_DISCOVERY/$agent" "$AGENT_DISCOVERY/$agent.md"
    done
    agent_link_count=0
    for agent in $DIRECT_AGENTS; do
        src=$(resolve_agent_src "$agent")
        [ -n "$src" ] || continue
        link_name=$(basename "$src")
        ln -sfn "$(realpath "$src")" "$AGENT_DISCOVERY/$link_name"
        agent_link_count=$((agent_link_count + 1))
    done
    ok "Agents: $agent_link_count discovery symlinks"
fi
echo ""

# --- Step 6: Setup asc-devkit ---
step "[6/7] Setting up asc-devkit..."
ASC_DEVKIT_DIR="$SCRIPT_DIR/asc-devkit"

if [ -d "$ASC_DEVKIT_DIR" ]; then
    cd "$ASC_DEVKIT_DIR"
    git checkout . 2>/dev/null || true
    git pull --quiet 2>/dev/null || warn "git pull failed, using existing version"
    cd "$SCRIPT_DIR"
    ok "asc-devkit updated"
else
    git clone --quiet https://gitcode.com/cann/asc-devkit.git "$ASC_DEVKIT_DIR" 2>/dev/null && cd "$ASC_DEVKIT_DIR" && git checkout --quiet 31f3ab38 || warn "git clone failed, skipping asc-devkit"
    [ -d "$ASC_DEVKIT_DIR" ] && ok "asc-devkit cloned"
fi

if [ -d "$ASC_DEVKIT_DIR" ]; then
    # Try shared skills location for clean_markdown.py, with fallback
    CLEAN_SCRIPT=""
    for base in "$SHARED_SKILL_ROOT" "$SCRIPT_DIR/../../skills"; do
        if [ -f "$(cd "$base" 2>/dev/null && pwd)/ascendc-docs-search/scripts/clean_markdown.py" ]; then
            CLEAN_SCRIPT="$(cd "$base" 2>/dev/null && pwd)/ascendc-docs-search/scripts/clean_markdown.py"
            break
        fi
    done
    if [ -n "$CLEAN_SCRIPT" ]; then
        python3 "$CLEAN_SCRIPT" --dir "$ASC_DEVKIT_DIR" --no-backup --quiet > /dev/null 2>&1 || warn "markdown cleanup failed"
    else
        warn "clean_markdown.py not found in any known location, skipping"
    fi
fi

# For project-level with custom target: also symlink asc-devkit into INSTALL_BASE
# so relative references from agents/workflow work correctly
if [ "$LEVEL" = "project" ] && [ -d "$ASC_DEVKIT_DIR" ]; then
    if [ "$INSTALL_BASE" != "$SCRIPT_DIR" ]; then
        ln -sfn "$(realpath "$ASC_DEVKIT_DIR")" "$INSTALL_BASE/asc-devkit"
        ok "asc-devkit → $INSTALL_BASE/"
    fi
fi

# --- Step 7: Generate manifest + Health check ---
MANIFEST="$CONFIG_ROOT/cannbot-manifest.json"

SKILLS_JSON=$(echo "$ALL_SKILLS" | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
AGENTS_JSON=$(echo "$DIRECT_AGENTS" | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")

cat > "$MANIFEST" << MANIFEST_EOF
{
  "brand": "CANNBot",
  "version": "$VERSION",
  "team": "$TEAM_NAME",
  "level": "$LEVEL",
  "tool": "$TOOL",
  "installed_skills": $SKILLS_JSON,
  "installed_agents": $AGENTS_JSON,
  "brand_dir": "$CONFIG_ROOT",
  "install_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
MANIFEST_EOF

echo ""
step "[7/7] Running health check..."
health_ok=true
health_errors=""

for sub in skills agents; do
  target="$CANNBOT_DIR/$sub"
  if [ -d "$target" ]; then
    count=$(ls -1A "$target" 2>/dev/null | wc -l)
    [ "$count" -eq 0 ] && { health_errors="${health_errors}\n  ${YELLOW}⚠${NC} $sub/ is empty"; }
  else
    health_errors="${health_errors}\n  ${RED}✗${NC} $sub/ missing"
    health_ok=false
  fi
done

if [ ! -d "$ASC_DEVKIT_DIR" ]; then
  health_errors="${health_errors}\n  ${YELLOW}⚠${NC} asc-devkit not available"
fi
# When installed to a custom directory, also check the symlink there
if [ "$LEVEL" = "project" ] && [ "$INSTALL_BASE" != "$SCRIPT_DIR" ] && [ ! -d "$INSTALL_BASE/asc-devkit" ]; then
  health_errors="${health_errors}\n  ${YELLOW}⚠${NC} asc-devkit symlink missing in $INSTALL_BASE"
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

[ -f "$MANIFEST" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} Manifest generation failed"; health_ok=false; }

if [ "$health_ok" = true ] && [ -z "$health_errors" ]; then
  ok "All checks passed"
else
  echo -e "$health_errors"
  [ "$health_ok" = true ] && warn "Some warnings, see above" || err "Some checks failed, see above"
fi

# --- Summary & Quick Start ---
echo ""
echo -e "  ${GREEN}${BOLD}✓ Team $TEAM_NAME installed successfully!${NC}"
echo -e "  ${DIM}Skills: $ALL_SKILL_COUNT | Agents: $DIRECT_AGENT_COUNT${NC}"
echo ""
echo -e "  ${BOLD}Quick Start:${NC}"
if [ "$TOOL" = "opencode" ]; then
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}opencode${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}帮我开发一个 abs 算子，支持 float16 数据类型${NC}"
elif [ "$TOOL" = "trae" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 CLI/IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}帮我开发一个 abs 算子，支持 float16 数据类型${NC}"
elif [ "$TOOL" = "cursor" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 Cursor IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}帮我开发一个 abs 算子，支持 float16 数据类型${NC}"
else
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}claude${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}帮我开发一个 abs 算子，支持 float16 数据类型${NC}"
fi
echo ""
