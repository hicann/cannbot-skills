#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set -e

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
VERSION="1.2.2"

INCLUDED_SKILLS="ops-direct-invoke-flash gitcode-toolkit gitcode-pr-handler gitcode-issue-gen gitcode-issue-handler"
INCLUDED_AGENT_PATTERN="ops-direct-invoke-flash-*"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$SCRIPT_DIR"
LOCAL_AGENT_ROOT="$PLUGIN_ROOT/agents"
LOCAL_SKILL_ROOT="$PLUGIN_ROOT/skills"
INFRA_SKILL_ROOT="$(cd "$PLUGIN_ROOT/../../infra" && pwd)"

show_help() {
    cat << EOF
CANNBot - Ascend C Direct-Invoke Flash Installer

Usage: init.sh [level] [tool] [install_path]

Arguments:
  level        - Installation level: "project" (default) or "global"
  tool         - Target tool: "opencode" (default), "claude", "trae", "cursor", or "copilot"
  install_path - Project-level installation directory (default: current working directory)

Options:
  --help  - Show this help message

Examples:
  init.sh project opencode
  init.sh global claude
  init.sh project trae
EOF
}

LEVEL="project"
TOOL="opencode"
INSTALL_PATH=""

for arg in "$@"; do
    case "$arg" in
        --help) show_help; exit 0 ;;
        global|project) LEVEL="$arg" ;;
        opencode|claude|trae|cursor|copilot) TOOL="$arg" ;;
    esac
done

if [ $# -gt 0 ]; then
    last_arg="${!#}"
    case "$last_arg" in
        --help|global|project|opencode|claude|trae|cursor|copilot) ;;
        *) INSTALL_PATH="$last_arg" ;;
    esac
fi

if [ "$LEVEL" = "global" ]; then
    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$HOME/.config/opencode"
    elif [ "$TOOL" = "trae" ]; then
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
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.cursor"
    elif [ "$TOOL" = "copilot" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.github"
    else
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.claude"
    fi
fi

CANNBOT_DIR="$CONFIG_ROOT"

if [ -e "$CONFIG_ROOT/$BRAND" ] || [ -L "$CONFIG_ROOT/$BRAND" ]; then
    rm -rf "$CONFIG_ROOT/$BRAND"
fi

install_skill_links() {
    local target_root="$1"
    mkdir -p "$target_root"
    local count=0
    for skill in $INCLUDED_SKILLS; do
        local src="$LOCAL_SKILL_ROOT/$skill"
        if [ ! -d "$src" ]; then
            src="$INFRA_SKILL_ROOT/$skill"
        fi
        if [ -d "$src" ]; then
            rm -rf "$target_root/$skill"
            ln -sfn "$(realpath "$src")" "$target_root/$skill"
            count=$((count + 1))
        else
            warn "Skill not found: $skill"
        fi
    done
    # Explicit literal link for the bundled skill: keeps the install resolvable via the
    # plugin's self-named entry and lets the dependency validator (DG-05) discover it.
    ln -sfn "$SCRIPT_DIR/ops-direct-invoke-flash" "$CONFIG_ROOT/skills/ops-direct-invoke-flash"
    ok "Skills: $count linked"
}

install_agent_links() {
    local target_root="$1"
    mkdir -p "$target_root"
    local count=0
    for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
        [ -e "$agent_entry" ] || continue
        local name base
        name=$(basename "$agent_entry")
        base="${name%.md}"
        [[ "$base" != $INCLUDED_AGENT_PATTERN ]] && continue
        rm -f "$target_root/$name"
        ln -sfn "$(realpath "$agent_entry")" "$target_root/$name"
        count=$((count + 1))
    done
    ok "Agents: $count linked"
}

install_config() {
    mkdir -p "$CONFIG_ROOT"
    local config_src="$PLUGIN_ROOT/AGENTS.md"
    local config_target
    if [ "$LEVEL" = "project" ]; then
        if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "copilot" ]; then
            config_target="$PWD/AGENTS.md"
        else
            config_target="$PWD/CLAUDE.md"
        fi
    else
        if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "copilot" ]; then
            config_target="$CONFIG_ROOT/AGENTS.md"
        else
            config_target="$CONFIG_ROOT/CLAUDE.md"
        fi
    fi

    if { [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "copilot" ]; } && [ "$LEVEL" = "project" ] && [ "$PLUGIN_ROOT" = "$PWD" ]; then
        ok "$(basename "$config_target") already in current directory"
    else
        ln -sf "$config_src" "$config_target"
        ok "$(basename "$config_target")"
    fi

    if [ "$LEVEL" = "project" ] && [ "$config_target" != "$CONFIG_ROOT/$(basename "$config_target")" ]; then
        ln -sf "$config_src" "$CONFIG_ROOT/$(basename "$config_target")"
    fi
}

write_manifest() {
    local manifest="$CONFIG_ROOT/cannbot-manifest.json"
    local skills_json agents_json
    skills_json=$(printf '%s\n' $INCLUDED_SKILLS | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
    agents_json=$(find "$LOCAL_AGENT_ROOT" -maxdepth 1 -name 'ops-direct-invoke-flash-*.md' -exec basename {} \; | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
    cat > "$manifest" << EOF
{
  "brand": "CANNBot",
  "version": "$VERSION",
  "team": "ops-direct-invoke-flash",
  "level": "$LEVEL",
  "tool": "$TOOL",
  "installed_skills": $skills_json,
  "installed_agents": $agents_json,
  "brand_dir": "$CONFIG_ROOT",
  "install_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    ok "Manifest: $manifest"
}

step "[1/4] Installing skills and agents..."
mkdir -p "$CANNBOT_DIR"
install_skill_links "$CANNBOT_DIR/skills"
install_agent_links "$CANNBOT_DIR/agents"
echo ""

step "[2/4] Installing configuration..."
install_config
echo ""

step "[3/4] Writing manifest..."
write_manifest
echo ""

step "[4/4] Health check..."
health_ok=true
[ -d "$CANNBOT_DIR/skills" ] || { err "skills/ missing"; health_ok=false; }
[ -d "$CANNBOT_DIR/agents" ] || { err "agents/ missing"; health_ok=false; }
[ -f "$CONFIG_ROOT/cannbot-manifest.json" ] || { err "manifest missing"; health_ok=false; }

if [ "$health_ok" = true ]; then
    ok "All checks passed"
else
    exit 1
fi

echo ""
echo -e "  ${GREEN}${BOLD}✓ ops-direct-invoke-flash installed successfully!${NC}"
echo ""
