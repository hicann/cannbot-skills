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

INCLUDED_SKILLS="ops-direct-invoke-flash ascendc-st-design ascendc-whitebox-design"
INCLUDED_AGENT_PATTERN="ops-direct-invoke-flash-*"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$SCRIPT_DIR"
LOCAL_AGENT_ROOT="$PLUGIN_ROOT/agents"
LOCAL_SKILL_ROOT="$PLUGIN_ROOT/skills"
# Codex custom agents use standalone TOML definitions.
CODEX_AGENT_ROOT="$LOCAL_AGENT_ROOT/codex"
if [ -d "$PLUGIN_ROOT/../../ops" ]; then
    SHARED_SKILL_ROOT="$(cd "$PLUGIN_ROOT/../../ops" && pwd)"
else
    SHARED_SKILL_ROOT=""
fi
if [ -d "$PLUGIN_ROOT/../../infra" ]; then
    INFRA_SKILL_ROOT="$(cd "$PLUGIN_ROOT/../../infra" && pwd)"
else
    INFRA_SKILL_ROOT=""
fi

show_help() {
    cat << EOF
CANNBot - Ascend C Direct-Invoke Flash Installer

Usage: init.sh [level] [tool] [install_path]

Arguments:
  level        - Installation level: "project" (default) or "global"
  tool         - Target tool: "opencode" (default), "claude", "trae", "cursor", "codex", "copilot", or "codearts"
  install_path - Project-level installation directory (default: current working directory)

Options:
  --help  - Show this help message

Examples:
  init.sh project opencode
  init.sh global claude
  init.sh project trae
  init.sh project codex
  init.sh global codex
  init.sh project codearts
EOF
}

LEVEL="project"
TOOL="opencode"
INSTALL_PATH=""

for arg in "$@"; do
    case "$arg" in
        --help) show_help; exit 0 ;;
        global|project) LEVEL="$arg" ;;
        opencode|claude|trae|cursor|codex|copilot|codearts) TOOL="$arg" ;;
    esac
done

if [ $# -gt 0 ]; then
    last_arg="${!#}"
    case "$last_arg" in
        --help|global|project|opencode|claude|trae|cursor|codex|copilot|codearts) ;;
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
    elif [ "$TOOL" = "codex" ]; then
        CONFIG_ROOT="$HOME/.codex"
    elif [ "$TOOL" = "copilot" ]; then
        CONFIG_ROOT="$HOME/.copilot"
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
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.cursor"
    elif [ "$TOOL" = "codex" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.codex"
    elif [ "$TOOL" = "copilot" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.github"
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
            src="$SHARED_SKILL_ROOT/$skill"
        fi
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
    ln -sfn "$SCRIPT_DIR/ops-direct-invoke-flash" "$target_root/ops-direct-invoke-flash"
    ok "Skills: $count linked"
}

install_agent_links() {
    local target_root="$1"
    local count=0
    if [ "$TOOL" = "codex" ]; then
        # Codex may ignore symlinked custom-agent TOML files (openai/codex#15345).
        # Install regular TOML files (with __CANNBOT_AGENT_SOURCE__ resolved) so
        # multiple plugins can coexist in the same .codex/agents/ directory.
        mkdir -p "$target_root"
        for agent_entry in "$CODEX_AGENT_ROOT"/*.toml; do
            [ -f "$agent_entry" ] || continue
            local name base canonical_agent escaped_agent tmpfile
            name=$(basename "$agent_entry")
            base="${name%.toml}"
            [[ "$base" != $INCLUDED_AGENT_PATTERN ]] && continue
            canonical_agent="$LOCAL_AGENT_ROOT/$base.md"
            escaped_agent="$(echo "$canonical_agent" | sed 's/[&|\\]/\\&/g')"
            tmpfile=$(mktemp)
            sed "s|__CANNBOT_AGENT_SOURCE__|$escaped_agent|g" "$agent_entry" > "$tmpfile"
            safe_install_file "$tmpfile" "$target_root/$name" "$name" "$LEVEL"
            count=$((count + 1))
        done
        ok "Agents: $count compatible TOML files"
    else
        mkdir -p "$target_root"
        for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
            [ -e "$agent_entry" ] || continue
            local name base
            name=$(basename "$agent_entry")
            base="${name%.*}"
            [[ "$base" != $INCLUDED_AGENT_PATTERN ]] && continue
            rm -f "$target_root/$name"
            ln -sfn "$(realpath "$agent_entry")" "$target_root/$name"
            count=$((count + 1))
        done
        ok "Agents: $count linked"
    fi
}

install_config() {
    mkdir -p "$CONFIG_ROOT"
    local config_src="$PLUGIN_ROOT/AGENTS.md"
    local config_target
    if [ "$LEVEL" = "project" ]; then
        if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "copilot" ] || [ "$TOOL" = "codex" ] || [ "$TOOL" = "codearts" ]; then
            config_target="$CONFIG_ROOT_BASE/AGENTS.md"
        else
            config_target="$CONFIG_ROOT_BASE/CLAUDE.md"
        fi
    else
        if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "copilot" ] || [ "$TOOL" = "codex" ] || [ "$TOOL" = "codearts" ]; then
            config_target="$CONFIG_ROOT/AGENTS.md"
        else
            config_target="$CONFIG_ROOT/CLAUDE.md"
        fi
    fi

    if { [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "copilot" ] || [ "$TOOL" = "codex" ] || [ "$TOOL" = "codearts" ]; } && [ "$LEVEL" = "project" ] && [ "$PLUGIN_ROOT" = "$CONFIG_ROOT_BASE" ]; then
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
    skills_json=$(ls -d "$SKILL_DISCOVERY_ROOT"/*/ 2>/dev/null | while read d; do d="${d%/}"; echo "${d##*/}"; done | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
    agents_json=$(ls -d "$AGENT_DISCOVERY_ROOT"/* 2>/dev/null | while read d; do echo "${d##*/}"; done | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
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
  "skills_dir": "$SKILL_DISCOVERY_ROOT",
  "agents_dir": "$AGENT_DISCOVERY_ROOT",
  "install_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    ok "Manifest: $manifest"
}

echo ""
echo "  Tool:      $TOOL"
echo "  Level:     $LEVEL"
echo "  Path:      $CONFIG_ROOT"
if [ "$TOOL" = "codex" ]; then
    echo "  Skills:    $SKILL_DISCOVERY_ROOT"
    echo "  Subagents: $AGENT_DISCOVERY_ROOT"
fi
echo ""
step "[1/4] Installing skills and agents..."
mkdir -p "$CANNBOT_DIR"
install_skill_links "$SKILL_DISCOVERY_ROOT"
install_agent_links "$AGENT_DISCOVERY_ROOT"
echo ""

step "[2/4] Installing configuration..."
install_config
echo ""

step "[3/4] Writing manifest..."
write_manifest
echo ""

step "[4/4] Health check..."
health_ok=true
[ -d "$SKILL_DISCOVERY_ROOT" ] || { err "skills/ missing"; health_ok=false; }
[ -d "$AGENT_DISCOVERY_ROOT" ] || { err "agents/ missing"; health_ok=false; }
[ -f "$CONFIG_ROOT/cannbot-manifest.json" ] || { err "manifest missing"; health_ok=false; }

if [ "$health_ok" = true ]; then
    ok "All checks passed"
else
    exit 1
fi

echo ""
echo -e "  ${GREEN}${BOLD}✓ ops-direct-invoke-flash installed successfully!${NC}"
echo ""
