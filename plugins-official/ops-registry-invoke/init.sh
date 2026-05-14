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

BRAND="cannbot"
VERSION="1.0.0"

# --- Plugin-specific filters ---
# Skill whitelist (space-separated list) - references shared ops + local workflow
INCLUDED_SKILLS="ascendc-api-best-practices ascendc-code-review ascendc-crash-debug ascendc-docs-gen ascendc-docs-search ascendc-env-check npu-arch ascendc-performance-best-practices ascendc-precision-debug ascendc-registry-invoke-template ascendc-runtime-debug ascendc-st-design ascendc-tiling-design ascendc-ut-develop ops-precision-standard ops-profiling ops-registry-invoke-workflow"
# Agent whitelist (shell pattern) - uses local agents/
INCLUDED_AGENT_PATTERN="ascendc-ops-*"

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAM_NAME="$(basename "$SCRIPT_DIR")"
PLUGIN_ROOT="$SCRIPT_DIR"
SHARED_SKILL_ROOT="$(cd "$SCRIPT_DIR/../../ops" && pwd)"
LOCAL_AGENT_ROOT="$SCRIPT_DIR/agents"

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

Usage: init.sh [level] [tool]

Arguments:
  level   - Installation level: "project" (default) or "global"
  tool    - Target tool: "opencode" (default), "claude", "trae", or "cursor"

Options:
  --help  - Show this help message

Examples:
  init.sh                      # Project-level, OpenCode
  init.sh project opencode     # Project-level, OpenCode
  init.sh global claude        # Global-level, Claude Code
  init.sh project claude       # Project-level, Claude Code
  init.sh project trae         # Project-level, Trae
  init.sh project cursor       # Project-level, Cursor

Installation paths:
  OpenCode: .opencode/{skills,agents}/  (auto-discovered)
  Claude:   .claude/{skills,agents}/    (per-skill symlinks auto-created)
  Trae:     .trae/{skills,agents}/      (symlinks, project-level only)
  Cursor:   .cursor/{skills,agents}/    (auto-discovered)

After installation, launch directly:
  OpenCode: opencode
  Claude:   claude
  Trae:     通过 CLI 或 IDE 启动
  Cursor:   通过 Cursor IDE 启动
EOF
}

# --- Parse arguments ---
LEVEL="project"
TOOL="opencode"

for arg in "$@"; do
    case "$arg" in
        --help)            show_help; exit 0 ;;
        global|project)    LEVEL="$arg" ;;
        opencode|claude|trae|cursor)   TOOL="$arg" ;;
        *)  echo "Error: Unknown argument '$arg'. Valid: global, project, opencode, claude, trae, cursor, --help."
            exit 1 ;;
    esac
done

# --- Determine config root ---
if [ "$LEVEL" = "global" ]; then
    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$HOME/.config/opencode"
    elif [ "$TOOL" = "trae" ]; then
        echo "Error: Global installation is not supported for Trae. Use project-level instead."
        exit 1
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_ROOT="$HOME/.cursor"
    else
        CONFIG_ROOT="$HOME/.claude"
    fi
else
    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$SCRIPT_DIR/.opencode"
    elif [ "$TOOL" = "trae" ]; then
        CONFIG_ROOT="$SCRIPT_DIR/.trae"
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_ROOT="$SCRIPT_DIR/.cursor"
    else
        CONFIG_ROOT="$SCRIPT_DIR/.claude"
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

# Resolve skill source path (shared skills/ first, then team-local)
# For team-local skills, searches subdirectories for a SKILL.md with matching `name:`.
resolve_skill_src() {
    local skill="$1"
    if [ -d "$SHARED_SKILL_ROOT/$skill" ]; then
        echo "$SHARED_SKILL_ROOT/$skill"
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
    # Trae/Claude/Cursor: create directories (per-item symlinks handled in Step 5)
    mkdir -p "$CONFIG_ROOT/skills" "$CONFIG_ROOT/agents"
    ok "Prepared: skills/, agents/"
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

if [ "$LEVEL" = "project" ]; then
    # Project-level: config file should be in current directory (PWD)
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ]; then
        config_target="$PWD/AGENTS.md"
    else
        config_target="$PWD/CLAUDE.md"
    fi
else
    # Global-level: config file in CONFIG_ROOT
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ]; then
        config_target="$CONFIG_ROOT/AGENTS.md"
    else
        config_target="$CONFIG_ROOT/CLAUDE.md"
    fi
fi
config_src="$SCRIPT_DIR/AGENTS.md"

# Skip only when source file is already at target location
if [ "$config_src" = "$config_target" ]; then
    info "$(basename "$config_target") already at target location"
else
    ln -sf "$config_src" "$config_target"
    ok "$(basename "$config_target")"
fi

# Also ensure CONFIG_ROOT has the config file (for consistency with other init.sh)
if [ "$LEVEL" = "project" ] && [ "$config_target" != "$CONFIG_ROOT/$(basename "$config_target")" ]; then
    ln -sf "$config_src" "$CONFIG_ROOT/$(basename "$config_target")"
fi

link_workflow_skill

echo ""

# --- Step 5: Configure tool discovery ---
step "[5/7] Configuring tool discovery..."

if [ "$TOOL" = "opencode" ]; then
    ok "Auto-scan: skills/, agents/"
else
    # Trae/Claude/Cursor: create per-skill discovery symlinks
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
    git clone --quiet https://gitcode.com/cann/asc-devkit.git "$ASC_DEVKIT_DIR" 2>/dev/null || warn "git clone failed, skipping asc-devkit"
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

if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ]; then
  [ -f "$CONFIG_ROOT/AGENTS.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} AGENTS.md missing"; health_ok=false; }
else
  [ -f "$CONFIG_ROOT/CLAUDE.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} CLAUDE.md missing"; health_ok=false; }
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
