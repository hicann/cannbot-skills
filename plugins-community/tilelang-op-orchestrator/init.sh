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

BRAND="cannbot"
VERSION="1.0.0"

# --- Plugin-specific filters ---
EXCLUDED_SKILL=""
# Skill whitelist pattern - references shared ops-lab/tilelang/skills
INCLUDED_SKILL_PATTERN="tilelang-*"
# Agent whitelist pattern - uses local agents/
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

Usage: init.sh [level] [tool]

Arguments:
  level   - Installation level: "project" (default) or "global"
  tool    - Target tool: "opencode" (default), "claude", "trae", "cursor", or "copilot"

Options:
  --help  - Show this help message

Examples:
  init.sh                      # Project-level, OpenCode
  init.sh project opencode     # Project-level, OpenCode
  init.sh global claude        # Global-level, Claude Code
  init.sh project claude       # Project-level, Claude Code
  init.sh project trae         # Project-level, Trae
  init.sh project cursor       # Project-level, Cursor
  init.sh project copilot      # Project-level, Copilot
  init.sh global copilot       # Global-level, Copilot

Installation paths (CANNBot brand):
  OpenCode: .opencode/{skills,agents}/  (auto-discovered)
  Claude:   .claude/{skills,agents}/    (per-skill symlinks auto-created)
  Copilot:  .github/{skills,agents}/    (project-level)
            ~/.copilot/{skills,agents}/ (global)

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
# Skills: reference shared ops-lab/tilelang/skills directory
SHARED_SKILL_ROOT="$(cd "$PLUGIN_ROOT/../../ops-lab/tilelang/skills" && pwd)"

for arg in "$@"; do
    case "$arg" in
        --help)            show_help; exit 0 ;;
        global|project)    LEVEL="$arg" ;;
        opencode|claude|trae|cursor|copilot)   TOOL="$arg" ;;
        *)  echo "Error: Unknown argument '$arg'. Valid: global, project, opencode, claude, trae, cursor, copilot, --help."
            exit 1 ;;
    esac
done

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

# Determine config root directory
if [ "$LEVEL" = "global" ]; then
    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$HOME/.config/opencode"
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
        detect_trae_variant
        case "$TRAE_VARIANT" in
            plugin) CONFIG_ROOT="$PLUGIN_ROOT/.marscode" ;;
            cli)    CONFIG_ROOT="$PLUGIN_ROOT/.traecli" ;;
            *)      CONFIG_ROOT="$PLUGIN_ROOT/.trae" ;;
        esac
    elif [ "$TOOL" = "copilot" ]; then
        CONFIG_ROOT="$PLUGIN_ROOT/.github"
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_ROOT="$PLUGIN_ROOT/.cursor"
    else
        CONFIG_ROOT="$PLUGIN_ROOT/.claude"
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

# --- Step 1: Create directory symlinks ---
step "[1/5] Setting up CANNBot directory..."
mkdir -p "$CANNBOT_DIR"

step1_summary=""
step1_warns=""
if [ "$TOOL" = "opencode" ]; then
    # OpenCode: per-item symlinks for skills (whitelist filtered)
    mkdir -p "$CANNBOT_DIR/skills"
    # Pre-clean existing skill symlinks
    for skill_dir in "$SHARED_SKILL_ROOT"/*/; do
        [ -d "$skill_dir" ] || continue
        name=$(basename "$skill_dir")
        target="$CANNBOT_DIR/skills/$name"
        [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
    done
    skill_count=0
    for skill_dir in "$SHARED_SKILL_ROOT"/*/; do
        [ -d "$skill_dir" ] || continue
        name=$(basename "$skill_dir")
        [[ "$name" != $INCLUDED_SKILL_PATTERN ]] && continue
        [ -n "$EXCLUDED_SKILL" ] && [ "$name" = "$EXCLUDED_SKILL" ] && continue
        ln -sfn "$(realpath "$skill_dir")" "$CANNBOT_DIR/skills/$name"
        skill_count=$((skill_count + 1))
    done
    step1_summary="skills(${skill_count}) "

    # OpenCode: per-item symlinks for agents (whitelist filtered)
    # Supports both agent directories and standalone .md agent files
    mkdir -p "$CANNBOT_DIR/agents"
    for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
        [ -e "$agent_entry" ] || continue
        name=$(basename "$agent_entry")
        target="$CANNBOT_DIR/agents/$name"
        [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
    done
    agent_count=0
    for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
        [ -e "$agent_entry" ] || continue
        name=$(basename "$agent_entry")
        # Match pattern against name (strip .md suffix for file-based agents)
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
    ok "Prepared: skills/, agents/"
fi
[ -n "$step1_warns" ] && echo -e "$step1_warns"
echo ""

# --- Step 2: Install config file (AGENTS.md / CLAUDE.md) ---
step "[2/5] Installing configuration..."
mkdir -p "$CONFIG_ROOT"

if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "copilot" ]; then
    ln -sf "$PLUGIN_ROOT/AGENTS.md" "$CONFIG_ROOT/AGENTS.md"
    ok "AGENTS.md"
else
    ln -sf "$PLUGIN_ROOT/AGENTS.md" "$CONFIG_ROOT/CLAUDE.md"
    ok "CLAUDE.md"
fi
echo ""

# --- Step 3: Configure tool discovery ---
step "[3/5] Configuring tool discovery..."

if [ "$TOOL" = "opencode" ]; then
    # OpenCode: skills/agents already at auto-scan paths, no extra discovery needed
    ok "Auto-scan: skills/, agents/"
else
    # Claude/Trae/Cursor/Copilot: create per-skill discovery symlinks (with filter, from shared ops-lab/tilelang/skills)
    DISCOVERY="$CONFIG_ROOT/skills"

    # Pre-clean existing skills
    for skill_dir in "$SHARED_SKILL_ROOT"/*/; do
        [ -d "$skill_dir" ] || continue
        name=$(basename "$skill_dir")
        target="$DISCOVERY/$name"
        [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
    done

    link_count=0
    for skill_dir in "$SHARED_SKILL_ROOT"/*/; do
        [ -d "$skill_dir" ] || continue
        name=$(basename "$skill_dir")
        [[ "$name" != $INCLUDED_SKILL_PATTERN ]] && continue
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

    # Claude/Trae/Cursor/Copilot: also create agent discovery symlinks (from local agents/)
    AGENT_DISCOVERY="$CONFIG_ROOT/agents"

    for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
        [ -e "$agent_entry" ] || continue
        name=$(basename "$agent_entry")
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

# Check config file
if [ "$TOOL" = "opencode" ]; then
  [ -f "$CONFIG_ROOT/AGENTS.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} AGENTS.md missing"; health_ok=false; }
else
  [ -f "$CONFIG_ROOT/CLAUDE.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} CLAUDE.md missing"; health_ok=false; }
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
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}帮我写一个 softmax 算子设计文档并开发算子${NC}"
elif [ "$TOOL" = "copilot" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 GitHub Copilot CLI / IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}帮我写一个 softmax 算子设计文档并开发算子${NC}"
elif [ "$TOOL" = "trae" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 CLI/IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}帮我写一个 softmax 算子设计文档并开发算子${NC}"
elif [ "$TOOL" = "cursor" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 Cursor IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}帮我写一个 softmax 算子设计文档并开发算子${NC}"
else
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}claude${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}帮我写一个 softmax 算子设计文档并开发算子${NC}"
fi
echo ""
