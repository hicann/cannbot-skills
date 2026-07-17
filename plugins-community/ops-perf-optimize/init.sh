#!/usr/bin/env bash
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

# --- Team-specific filters ---
INCLUDED_AGENT_PATTERN="ascendc-perf-*"
INCLUDED_SKILLS="ops-simulator ops-profiling ascendc-perf-optimize ascendc-performance-best-practices ascendc-docs-search"
# OpenCode registers markdown agents by filename under agents/; the primary agent
# (AGENTS.md, mode: primary) must be linked there to be loadable. Claude keeps it
# as CLAUDE.md at the config root.
PRIMARY_AGENT_NAME="perf-optimize"

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
  echo -e "  ${BOLD}AscendC Operator Perf Optimization Team${NC}"
  echo ""
}

show_help() {
    cat << EOF
CANNBot - AscendC Operator Performance Optimization Environment Installer

Usage: init.sh [level] [tool]

Arguments:
  level   - Installation level: "project" (default) or "global"
  tool    - Target tool: "opencode" (default), "claude", or "codearts"

Options:
  --help  - Show this help message

Examples:
  cd /path/to/your/project
  bash /path/to/ops-perf-optimize/init.sh                    # Project-level, OpenCode
  bash /path/to/ops-perf-optimize/init.sh project claude     # Project-level, Claude Code
  bash /path/to/ops-perf-optimize/init.sh global claude      # Global-level, Claude Code
  bash /path/to/ops-perf-optimize/init.sh project codearts   # Project-level, CodeArts

Installation paths (CANNBot brand):
  Project-level: $PWD/.claude/ or $PWD/.opencode/
  Global-level:  $HOME/.claude/ or $HOME/.config/opencode/

After installation, launch directly:
  OpenCode: opencode
  Claude:   claude
EOF
}

LEVEL="project"
TOOL="opencode"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$PWD"
PLUGIN_SRC="$SCRIPT_DIR"
if [ -d "$SCRIPT_DIR/../../ops" ]; then
    ASCEND_AGENT_ROOT="$(cd "$SCRIPT_DIR/../../ops" && pwd)"
else
    ASCEND_AGENT_ROOT=""
fi

for arg in "$@"; do
    case "$arg" in
        --help)            show_help; exit 0 ;;
        global|project)    LEVEL="$arg" ;;
        opencode|claude|codearts)   TOOL="$arg" ;;
        *)  echo "Error: Unknown argument '$arg'. Valid: global, project, opencode, claude, codearts, --help."
            exit 1 ;;
    esac
done

# Determine config root directory
if [ "$LEVEL" = "global" ]; then
    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$HOME/.config/opencode"
    elif [ "$TOOL" = "codearts" ]; then
        CONFIG_ROOT="$HOME/.codeartsdoer"
    else
        CONFIG_ROOT="$HOME/.claude"
    fi
else
    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$PROJECT_ROOT/.opencode"
    elif [ "$TOOL" = "codearts" ]; then
        CONFIG_ROOT="$PROJECT_ROOT/.codeartsdoer"
    else
        CONFIG_ROOT="$PROJECT_ROOT/.claude"
    fi
fi

CANNBOT_DIR="$CONFIG_ROOT"

# --- Uninstall previous installation (manifest-based, safe) ---
uninstall_previous() {
    local manifest="$CONFIG_ROOT/cannbot-manifest.json"
    [ -f "$manifest" ] || return 0

    # Remove previously installed skills (only symlinks we created)
    local old_skills
    old_skills=$(python3 -c "
import json
try:
    with open('$manifest') as f:
        data = json.load(f)
    print(' '.join(data.get('installed_skills', [])))
except: pass
" 2>/dev/null)
    for skill in $old_skills; do
        local target="$CANNBOT_DIR/skills/$skill"
        if [ -L "$target" ]; then
            rm -f "$target"
        elif [ -d "$target" ]; then
            rm -rf "$target"
        fi
    done

    # Remove previously installed agents (only symlinks we created)
    local old_agents
    old_agents=$(python3 -c "
import json
try:
    with open('$manifest') as f:
        data = json.load(f)
    print(' '.join(data.get('installed_agents', [])))
except: pass
" 2>/dev/null)
    for agent in $old_agents; do
        local target="$CANNBOT_DIR/agents/$agent"
        [ -L "$target" ] && rm -f "$target"
    done

    # Remove config symlinks (only if they are symlinks, never real files)
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "codearts" ]; then
        [ -L "$CONFIG_ROOT/AGENTS.md" ] && rm -f "$CONFIG_ROOT/AGENTS.md"
    else
        [ -L "$CONFIG_ROOT/CLAUDE.md" ] && rm -f "$CONFIG_ROOT/CLAUDE.md"
    fi
    [ -L "$CONFIG_ROOT/workflows" ] && rm -f "$CONFIG_ROOT/workflows"

    # Remove project-root AGENTS.md trigger (only if it contains our marker,
    # never touch user-customized AGENTS.md)
    if [ "$TOOL" = "opencode" ] && [ "$LEVEL" = "project" ]; then
        local proj_agents="$PROJECT_ROOT/AGENTS.md"
        if [ -f "$proj_agents" ] && grep -q "工作流加载规则" "$proj_agents" 2>/dev/null; then
            rm -f "$proj_agents"
        fi
    fi

    rm -f "$manifest"
}

uninstall_previous

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

# --- Step 1: Create directory symlinks ---
step "[1/5] Setting up CANNBot directory..."
mkdir -p "$CANNBOT_DIR"

step1_summary=""
step1_warns=""
if [ "$TOOL" = "opencode" ]; then
    # OpenCode: per-item symlinks for skills (only required skills)
    SKILLS_SRC="$ASCEND_AGENT_ROOT"
    mkdir -p "$CANNBOT_DIR/skills"
    skill_count=0
    if [ -d "$SKILLS_SRC" ]; then
        for skill_name in $INCLUDED_SKILLS; do
            skill_dir="$SKILLS_SRC/$skill_name"
            [ -d "$skill_dir" ] || continue
            ln -sfn "$(realpath "$skill_dir")" "$CANNBOT_DIR/skills/$skill_name"
            skill_count=$((skill_count + 1))
        done
    fi
    step1_summary="skills(${skill_count}) "

    # OpenCode: per-item symlinks for agents (whitelist filtered)
    mkdir -p "$CANNBOT_DIR/agents"
    # Pre-clean existing agent symlinks
    for agent_file in "$PLUGIN_SRC/agents"/*.md; do
        [ -f "$agent_file" ] || continue
        name=$(basename "$agent_file")
        target="$CANNBOT_DIR/agents/$name"
        [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
    done
    agent_count=0
    for agent_file in "$PLUGIN_SRC/agents"/*.md; do
        [ -f "$agent_file" ] || continue
        name=$(basename "$agent_file")
        base="${name%.md}"
        [[ "$base" != $INCLUDED_AGENT_PATTERN ]] && continue
        ln -sfn "$(realpath "$agent_file")" "$CANNBOT_DIR/agents/$name"
        agent_count=$((agent_count + 1))
    done
    step1_summary="${step1_summary}agents(${agent_count})"
    ok "Linked: $step1_summary"
else
    # Claude: create directories (per-item symlinks handled in Step 3)
    mkdir -p "$CONFIG_ROOT/skills" "$CONFIG_ROOT/agents"
    ok "Prepared: skills/, agents/"
fi
[ -n "$step1_warns" ] && echo -e "$step1_warns"
echo ""

# --- Step 2: Install primary agent (OpenCode) / config file (Claude) ---
step "[2/5] Installing configuration..."
mkdir -p "$CONFIG_ROOT"

if [ "$TOOL" = "opencode" ]; then
    # OpenCode scans agents/*.md to register subagents (filename = agent name).
    # The plugin's AGENTS.md (mode: primary, full workflow definition) is linked
    # under agents/ as perf-optimize.md so it can be dispatched as a subagent.
    mkdir -p "$CANNBOT_DIR/agents"
    config_target="$CANNBOT_DIR/agents/$PRIMARY_AGENT_NAME.md"
elif [ "$TOOL" = "codearts" ]; then
    config_target="$CONFIG_ROOT/AGENTS.md"
else
    config_target="$CONFIG_ROOT/CLAUDE.md"
fi

if [ -e "$config_target" ] && [ ! -L "$config_target" ]; then
    warn "$(basename "$config_target") already exists (not a symlink), skipping — remove manually if you want to overwrite"
elif [ -e "$config_target" ] || [ -L "$config_target" ]; then
    rm -f "$config_target"
    ln -sf "$PLUGIN_SRC/AGENTS.md" "$config_target"
    ok "$(basename "$config_target") (replaced)"
else
    ln -sf "$PLUGIN_SRC/AGENTS.md" "$config_target"
    ok "$(basename "$config_target")"
fi

# Link workflows directory
if [ -d "$PLUGIN_SRC/workflows" ]; then
    workflow_target="$CONFIG_ROOT/workflows"
    if [ -e "$workflow_target" ] && [ ! -L "$workflow_target" ]; then
        warn "workflows/ already exists (not a symlink), skipping — remove manually if you want to overwrite"
    else
        [ -e "$workflow_target" ] || [ -L "$workflow_target" ] && rm -rf "$workflow_target"
        ln -sfn "$(realpath "$PLUGIN_SRC/workflows")" "$workflow_target"
    fi
else
    warn "workflows/ not found, skipping"
fi

# Install project-root AGENTS.md trigger (OpenCode only)
# This file tells the main agent WHEN to dispatch to the perf-optimize subagent.
# It is a COPY (not symlink) so the user can add other instructions alongside.
if [ "$TOOL" = "opencode" ] && [ "$LEVEL" = "project" ]; then
    trigger_src="$PLUGIN_SRC/project-AGENTS.md"
    trigger_target="$PROJECT_ROOT/AGENTS.md"
    if [ -f "$trigger_src" ]; then
        if [ ! -e "$trigger_target" ]; then
            cp "$trigger_src" "$trigger_target"
            ok "AGENTS.md (project trigger) installed"
        elif [ -L "$trigger_target" ]; then
            rm -f "$trigger_target"
            cp "$trigger_src" "$trigger_target"
            ok "AGENTS.md (project trigger) replaced"
        elif grep -q "工作流加载规则" "$trigger_target" 2>/dev/null; then
            cp "$trigger_src" "$trigger_target"
            ok "AGENTS.md (project trigger) updated"
        else
            warn "AGENTS.md already exists (user content), skipping — merge project-AGENTS.md manually if needed"
        fi
    fi
fi
echo ""

# --- Step 3: Configure tool discovery ---
step "[3/5] Configuring tool discovery..."

if [ "$TOOL" = "opencode" ]; then
    # OpenCode: skills/agents already at auto-scan paths, no extra discovery needed
    ok "Auto-scan: skills/, agents/"
else
    # Claude: create per-skill discovery symlinks (only required skills)
    SKILLS_SRC="$ASCEND_AGENT_ROOT"
    DISCOVERY="$CONFIG_ROOT/skills"

    link_count=0
    if [ -d "$SKILLS_SRC" ]; then
        for skill_name in $INCLUDED_SKILLS; do
            skill_dir="$SKILLS_SRC/$skill_name"
            [ -d "$skill_dir" ] || continue
            ln -sfn "$(realpath "$skill_dir")" "$DISCOVERY/$skill_name"
            link_count=$((link_count + 1))
        done
    fi

    # Clean broken symlinks
    for link in "$DISCOVERY"/*/; do
        link="${link%/}"
        [ -L "$link" ] && [ ! -e "$link" ] && rm "$link"
    done

    ok "Skills: $link_count discovery symlinks"

    # Claude: also create agent discovery symlinks
    AGENTS_SRC="$PLUGIN_SRC/agents"
    AGENT_DISCOVERY="$CONFIG_ROOT/agents"

    agent_link_count=0
    for agent_file in "$AGENTS_SRC"/*.md; do
        [ -f "$agent_file" ] || continue
        name=$(basename "$agent_file")
        base="${name%.md}"
        [[ "$base" != $INCLUDED_AGENT_PATTERN ]] && continue
        target="$AGENT_DISCOVERY/$name"
        ln -sfn "$(realpath "$agent_file")" "$target"
        agent_link_count=$((agent_link_count + 1))
    done

    for link in "$AGENT_DISCOVERY"/*.md; do
        [ -L "$link" ] && [ ! -e "$link" ] && rm "$link"
    done

    ok "Agents: $agent_link_count discovery symlinks"
fi
echo ""

# --- Step 4: Setup reference repositories (optional) ---
step "[4/5] Setting up reference repositories..."
ASC_DEVKIT_DIR="$PROJECT_ROOT/asc-devkit"
CANN_SAMPLES_DIR="$PROJECT_ROOT/cann-samples"

# Setup asc-devkit
if [ -d "$ASC_DEVKIT_DIR" ]; then
    cd "$ASC_DEVKIT_DIR"
    git checkout . 2>/dev/null || true
    git pull --quiet 2>/dev/null || warn "asc-devkit git pull failed, using existing version"
    cd "$PROJECT_ROOT"
    ok "asc-devkit updated"
else
    git clone --quiet https://gitcode.com/cann/asc-devkit.git "$ASC_DEVKIT_DIR" 2>/dev/null || warn "git clone failed, skipping asc-devkit (optional)"
    [ -d "$ASC_DEVKIT_DIR" ] && ok "asc-devkit cloned"
fi

# Setup cann-samples
if [ -d "$CANN_SAMPLES_DIR" ]; then
    cd "$CANN_SAMPLES_DIR"
    git checkout . 2>/dev/null || true
    git pull --quiet 2>/dev/null || warn "cann-samples git pull failed, using existing version"
    cd "$PROJECT_ROOT"
    ok "cann-samples updated"
else
    git clone --quiet https://gitcode.com/cann/cann-samples.git "$CANN_SAMPLES_DIR" 2>/dev/null || warn "git clone failed, skipping cann-samples (optional)"
    [ -d "$CANN_SAMPLES_DIR" ] && ok "cann-samples cloned"
fi

if [ -n "$ASCEND_AGENT_ROOT" ] && [ -d "$ASC_DEVKIT_DIR" ] && [ -f "$ASCEND_AGENT_ROOT/ascendc-docs-search/scripts/clean_markdown.py" ]; then
    python3 "$ASCEND_AGENT_ROOT/ascendc-docs-search/scripts/clean_markdown.py" --dir "$ASC_DEVKIT_DIR" --no-backup --quiet > /dev/null 2>&1 || warn "markdown cleanup failed"
fi

# --- Step 5: Generate brand manifest (silent) ---
MANIFEST="$CONFIG_ROOT/cannbot-manifest.json"

# Collect installed skills
SKILLS_JSON="[]"
if [ -d "$CANNBOT_DIR/skills" ]; then
  SKILLS_JSON=$(ls -d "$CANNBOT_DIR/skills"/*/ 2>/dev/null | while read d; do
    d="${d%/}"; echo "${d##*/}"
  done | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
fi

# Collect installed agents
AGENTS_JSON="[]"
if [ -d "$CANNBOT_DIR/agents" ]; then
  AGENTS_JSON=$(ls "$CANNBOT_DIR/agents"/*.md 2>/dev/null | while read f; do
    echo "${f##*/}"
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
  "project_agents_trigger": "$([ "$TOOL" = "opencode" ] && [ "$LEVEL" = "project" ] && echo "$PROJECT_ROOT/AGENTS.md" || echo "")",
  "devkit_dir": "$ASC_DEVKIT_DIR",
  "samples_dir": "$CANN_SAMPLES_DIR",
  "brand_dir": "$CONFIG_ROOT",
  "install_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
MANIFEST_EOF


# --- Step 5: Health check ---
echo ""
step "[5/5] Running health check..."
health_ok=true
health_errors=""

# Check directory symlinks
if [ "$TOOL" = "opencode" ]; then
  HEALTH_SUBS="skills agents"
else
  HEALTH_SUBS="skills agents"
fi
for sub in $HEALTH_SUBS; do
  target="$CANNBOT_DIR/$sub"
  if [ -d "$target" ]; then
    if [ "$sub" = "agents" ]; then
      count=$(ls "$target"/*.md 2>/dev/null | wc -l)
    else
      count=$(ls -d "$target"/*/ 2>/dev/null | wc -l)
    fi
    [ "$count" -eq 0 ] && { health_errors="${health_errors}\n  ${YELLOW}⚠${NC} $sub/ is empty"; }
  else
    health_errors="${health_errors}\n  ${RED}✗${NC} $sub/ missing"
    health_ok=false
  fi
done

# Check skill count consistency (Claude only)
if [ "$TOOL" = "claude" ] && [ -d "$DISCOVERY" ]; then
  expected=$(echo $REQUIRED_SKILLS | wc -w)
  actual=$link_count
  if [ "$actual" -ne "$expected" ]; then
    health_errors="${health_errors}\n  ${YELLOW}⚠${NC} Skill discovery mismatch: $actual/$expected"
  fi
fi

# Check reference repositories
if [ ! -d "$ASC_DEVKIT_DIR" ]; then
  health_errors="${health_errors}\n  ${YELLOW}⚠${NC} asc-devkit not available (optional)"
fi
if [ ! -d "$CANN_SAMPLES_DIR" ]; then
  health_errors="${health_errors}\n  ${YELLOW}⚠${NC} cann-samples not available (optional)"
fi

# Check primary agent / config file
if [ "$TOOL" = "opencode" ]; then
  [ -f "$CANNBOT_DIR/agents/$PRIMARY_AGENT_NAME.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} $PRIMARY_AGENT_NAME primary agent missing"; health_ok=false; }
  # Check project-root AGENTS.md trigger (project-level only)
  if [ "$LEVEL" = "project" ]; then
    [ -f "$PROJECT_ROOT/AGENTS.md" ] || { health_errors="${health_errors}\n  ${YELLOW}⚠${NC} Project AGENTS.md trigger not installed"; }
  fi
elif [ "$TOOL" = "codearts" ]; then
  [ -f "$CONFIG_ROOT/AGENTS.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} AGENTS.md missing"; health_ok=false; }
else
  [ -f "$CONFIG_ROOT/CLAUDE.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} CLAUDE.md missing"; health_ok=false; }
fi

# Check manifest
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
if [ "$TOOL" = "codearts" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 CodeArts CLI / IDE 启动"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}请帮我优化 matmul 算子的性能，Shape M=1024 K=4096 N=2048，数据类型 fp4x2_e2m1_t${NC}"
elif [ "$TOOL" = "opencode" ]; then
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}opencode${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}请帮我优化 matmul 算子的性能，Shape M=1024 K=4096 N=2048，数据类型 fp4x2_e2m1_t${NC}"
else
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}claude${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}请帮我优化 matmul 算子的性能，Shape M=1024 K=4096 N=2048，数据类型 fp4x2_e2m1_t${NC}"
fi
echo ""
