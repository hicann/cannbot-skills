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

BRAND="cannbot"
VERSION="1.0.0"

# --- Plugin-specific filters ---
EXCLUDED_SKILL=""
# Triton plugin skills are excluded from installation
EXCLUDED_PATTERNS="triton"
# Agent whitelist (shell pattern) - uses local agents/
INCLUDED_AGENT_PATTERN="ascend-kernel-developer*"

show_banner() {
  echo ""
  echo -e "${CYAN}"
  cat << 'BANNER'
   ____    _    _   _ _   _ ____        _
  / ___|  / \  | \ | | \ | | __ )  ___ | |_
 | |     / _ \ |  \| |  \| |  _ \ / _ \| __|
 | |___ / ___ \| |\  | |\  | |_) | (_) | |_
  \____/_/   \_\_| \_|_| \_|____/ \___/ \__|

     ___  ___  ___   ___  _   _  ___ ___  ___
    / _ \| _ \/ __| | _ \ | | | || __| _ \/ __|
   | (_) |  _/\__ \ |  _/ | |_| || _||   /\__ \
    \___/|_|  |___/ |_|   \___/ |___|_|_\|___/
BANNER
  echo -e "${NC}"
  echo -e "  ${BOLD}Ascend C Ops Developer Team${NC}"
  echo ""
}

show_help() {
    cat << EOF
CANNBot - Ascend C Ops Developer Environment Installer

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
  init.sh global cursor        # Global-level, Cursor

Installation paths (CANNBot brand):
  OpenCode: .opencode/{skills,agents}/  (auto-discovered)
  Claude:   .claude/{skills,agents}/    (per-skill symlinks auto-created)
  Trae:     .trae/{skills,agents}/      (symlinks, project-level only)
  Cursor:   .cursor/{skills,agents}/    (per-skill symlinks auto-created)

After installation, launch directly:
  OpenCode: opencode
  Claude:   claude
  Trae:     通过 CLI 或 IDE 启动
  Cursor:   通过 Cursor IDE 启动
EOF
}

LEVEL="project"
TOOL="opencode"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$SCRIPT_DIR"
# Agents: use local agents/ directory (migrated with plugin)
LOCAL_AGENT_ROOT="$PLUGIN_ROOT/agents"
# Skills: reference shared ops and ops-lab directories
SHARED_SKILL_ROOT="$(cd "$PLUGIN_ROOT/../../ops" && pwd)"
SHARED_SKILL_ROOT_OPS_LAB="$(cd "$PLUGIN_ROOT/../../ops-lab" && pwd)"

# Parse skill dependencies from AGENTS.md frontmatter
if [ -f "$PLUGIN_ROOT/AGENTS.md" ]; then
  INCLUDED_SKILLS=$(python3 -c "
import re, sys
with open('$PLUGIN_ROOT/AGENTS.md') as f:
    content = f.read()
m = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
if m:
    fm = m.group(1)
    sm = re.search(r'^skills:\n((?:\s+-\s+.+\n?)*)', fm, re.MULTILINE)
    if sm:
        skills = re.findall(r'^\s+-\s+(.+)$', sm.group(0), re.MULTILINE)
        filtered = [s for s in skills if not re.search(r'triton', s, re.IGNORECASE)]
        print(' '.join(filtered))
" 2>/dev/null)
  if [ -z "$INCLUDED_SKILLS" ]; then
    warn "Failed to parse skill dependencies from AGENTS.md, using fallback"
    INCLUDED_SKILLS="npu-arch ascendc-api-best-practices ascendc-docs-search ascendc-tiling-design tilelang2ascend-case-simplifier tilelang2ascend-operator-project-init ops-profiling ascendc-precision-debug tilelang2ascend-precision-tuning tilelang2ascend-tilelang-designer tilelang2ascend-translator tilelang2ascend-trace-recorder"
  fi
else
  err "AGENTS.md not found at $PLUGIN_ROOT/AGENTS.md"
  exit 1
fi

for arg in "$@"; do
    case "$arg" in
        --help)            show_help; exit 0 ;;
        global|project)    LEVEL="$arg" ;;
        opencode|claude|trae|cursor)   TOOL="$arg" ;;
        *)  echo "Error: Unknown argument '$arg'. Valid: global, project, opencode, claude, trae, cursor, --help."
            exit 1 ;;
    esac
done

# Determine config root directory
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
        CONFIG_ROOT="$PLUGIN_ROOT/.opencode"
    elif [ "$TOOL" = "trae" ]; then
        CONFIG_ROOT="$PLUGIN_ROOT/.trae"
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

# --- Step 0: Confirmation before installation ---
step "[0/5] Checking items to be installed..."

# Collect skills to install (from shared ops and ops-lab)
SKILLS_TO_INSTALL=""
SKILL_COUNT=0
for skill_dir in "$SHARED_SKILL_ROOT"/*/ "$SHARED_SKILL_ROOT_OPS_LAB"/*/ "$SHARED_SKILL_ROOT_OPS_LAB"/*/skills/*/; do
    [ -d "$skill_dir" ] || continue
    name=$(basename "$skill_dir")
    echo "$INCLUDED_SKILLS" | tr ' ' '\n' | grep -qxF "$name" || continue
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

if [ "$SKILL_COUNT" -gt 0 ]; then
    echo -e "${CYAN}Skills (${SKILL_COUNT} 项)：${NC}"
    for name in $SKILLS_TO_INSTALL; do
        target="$CANNBOT_DIR/skills/$name"
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
    for name in $AGENTS_TO_INSTALL; do
        target="$CANNBOT_DIR/agents/$name"
        if [ -e "$target" ] || [ -L "$target" ]; then
            echo -e "  ${YELLOW}$name${NC}"
        else
            echo -e "  ${GREEN}$name${NC}"
        fi
    done
    echo ""
fi

echo -e "${CYAN}配置文件：${NC}"
if [ "$LEVEL" = "project" ]; then
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ]; then
        config_target="$PWD/AGENTS.md"
    else
        config_target="$PWD/CLAUDE.md"
    fi
else
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ]; then
        config_target="$CONFIG_ROOT/AGENTS.md"
    else
        config_target="$CONFIG_ROOT/CLAUDE.md"
    fi
fi
config_src="$PLUGIN_ROOT/AGENTS.md"
if { [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ]; } && [ "$LEVEL" = "project" ] && [ "$PLUGIN_ROOT" = "$PWD" ]; then
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
    for skill_dir in "$SHARED_SKILL_ROOT"/*/ "$SHARED_SKILL_ROOT_OPS_LAB"/*/ "$SHARED_SKILL_ROOT_OPS_LAB"/*/skills/*/; do
        [ -d "$skill_dir" ] || continue
        name=$(basename "$skill_dir")
        echo "$INCLUDED_SKILLS" | tr ' ' '\n' | grep -qxF "$name" || continue
        target="$CANNBOT_DIR/skills/$name"
        [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
    done
    skill_count=0
    for skill_dir in "$SHARED_SKILL_ROOT"/*/ "$SHARED_SKILL_ROOT_OPS_LAB"/*/ "$SHARED_SKILL_ROOT_OPS_LAB"/*/skills/*/; do
        [ -d "$skill_dir" ] || continue
        name=$(basename "$skill_dir")
        echo "$INCLUDED_SKILLS" | tr ' ' '\n' | grep -qxF "$name" || continue
        [ -n "$EXCLUDED_SKILL" ] && [ "$name" = "$EXCLUDED_SKILL" ] && continue
        ln -sfn "$(realpath "$skill_dir")" "$CANNBOT_DIR/skills/$name"
        skill_count=$((skill_count + 1))
    done
    step1_summary="skills(${skill_count}) "

    # OpenCode: per-item symlinks for agents (from local agents/, whitelist filtered)
    mkdir -p "$CANNBOT_DIR/agents"
    for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
        [ -e "$agent_entry" ] || continue
        name=$(basename "$agent_entry")
        base_name="${name%.md}"
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
    step1_summary="${step1_summary}agents(${agent_count}) "

    # OpenCode: per-item symlinks for hooks (from local hooks/)
    mkdir -p "$CANNBOT_DIR/hooks"
    for hook_entry in "$PLUGIN_ROOT/hooks"/*; do
        [ -e "$hook_entry" ] || continue
        name=$(basename "$hook_entry")
        target="$CANNBOT_DIR/hooks/$name"
        [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
    done
    hook_count=0
    for hook_entry in "$PLUGIN_ROOT/hooks"/*; do
        [ -e "$hook_entry" ] || continue
        name=$(basename "$hook_entry")
        ln -sfn "$(realpath "$hook_entry")" "$CANNBOT_DIR/hooks/$name"
        hook_count=$((hook_count + 1))
    done
    step1_summary="${step1_summary}hooks(${hook_count})"
    ok "Linked: $step1_summary"
else
    # Trae/Claude/Cursor: create directories
    mkdir -p "$CONFIG_ROOT/skills" "$CONFIG_ROOT/agents" "$CONFIG_ROOT/hooks"
    ok "Prepared: skills/, agents/, hooks/"
fi
[ -n "$step1_warns" ] && echo -e "$step1_warns"

# Copy attention-patterns reference from translator to tilelang-designer
ATTN_SRC="$SHARED_SKILL_ROOT_OPS_LAB/tilelang-to-ascendc/skills/tilelang2ascend-translator/references/attention-patterns"
ATTN_DST="$SHARED_SKILL_ROOT_OPS_LAB/tilelang-to-ascendc/skills/tilelang2ascend-tilelang-designer/references/attention-patterns"
if [ -d "$ATTN_SRC" ]; then
    mkdir -p "$(dirname "$ATTN_DST")"
    rm -rf "$ATTN_DST"
    cp -r "$ATTN_SRC" "$ATTN_DST"
    ok "attention-patterns → tilelang-designer/references/"
else
    warn "attention-patterns source not found, skipping copy"
fi

echo ""

# --- Step 2: Install config file (AGENTS.md / CLAUDE.md) ---
step "[2/5] Installing configuration..."

if [ "$LEVEL" = "project" ]; then
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ]; then
        config_target="$PWD/AGENTS.md"
    else
        config_target="$PWD/CLAUDE.md"
    fi
else
    mkdir -p "$CONFIG_ROOT"
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ]; then
        config_target="$CONFIG_ROOT/AGENTS.md"
    else
        config_target="$CONFIG_ROOT/CLAUDE.md"
    fi
fi

config_src="$PLUGIN_ROOT/AGENTS.md"

if { [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ]; } && [ "$LEVEL" = "project" ] && [ "$PLUGIN_ROOT" = "$PWD" ]; then
    ok "$(basename "$config_target") already in current directory"
else
    if [ "$LEVEL" = "global" ]; then
        [ -e "$config_target" ] || [ -L "$config_target" ] && rm -f "$config_target"
        PLUGIN_ROOT_ABS="$(realpath "$PLUGIN_ROOT")"
        ESCAPED_ROOT="$(echo "$PLUGIN_ROOT_ABS" | sed 's/#/\\#/g')"
        sed \
          -e "s#bash workflows/scripts/#bash ${ESCAPED_ROOT}/workflows/scripts/#g" \
          -e "s#](workflows/#](${ESCAPED_ROOT}/workflows/#g" \
          -e "s#\`workflows/#\`${ESCAPED_ROOT}/workflows/#g" \
          "$config_src" > "$config_target"
        ok "$(basename "$config_target") (absolute paths for global mode)"
    else
        ln -sf "$config_src" "$config_target"
        ok "$(basename "$config_target")"
    fi
fi

if { [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ]; } && [ "$LEVEL" = "project" ]; then
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
    ok "Auto-scan: skills/, agents/"
else
    # Trae/Claude/Cursor: create per-skill discovery symlinks (with filter, from shared ops)
    DISCOVERY="$CONFIG_ROOT/skills"

    for skill_dir in "$SHARED_SKILL_ROOT"/*/ "$SHARED_SKILL_ROOT_OPS_LAB"/*/ "$SHARED_SKILL_ROOT_OPS_LAB"/*/skills/*/; do
        [ -d "$skill_dir" ] || continue
        name=$(basename "$skill_dir")
        echo "$INCLUDED_SKILLS" | tr ' ' '\n' | grep -qxF "$name" || continue
        target="$DISCOVERY/$name"
        [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
    done

    link_count=0
    for skill_dir in "$SHARED_SKILL_ROOT"/*/ "$SHARED_SKILL_ROOT_OPS_LAB"/*/ "$SHARED_SKILL_ROOT_OPS_LAB"/*/skills/*/; do
        [ -d "$skill_dir" ] || continue
        name=$(basename "$skill_dir")
        echo "$INCLUDED_SKILLS" | tr ' ' '\n' | grep -qxF "$name" || continue
        [ -n "$EXCLUDED_SKILL" ] && [ "$name" = "$EXCLUDED_SKILL" ] && continue
        target="$DISCOVERY/$name"
        ln -sfn "$(realpath "$skill_dir")" "$target"
        link_count=$((link_count + 1))
    done

    for link in "$DISCOVERY"/*/; do
        link="${link%/}"
        [ -L "$link" ] && [ ! -e "$link" ] && rm "$link"
    done

    ok "Skills: $link_count discovery symlinks"

    AGENT_DISCOVERY="$CONFIG_ROOT/agents"

    for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
        [ -e "$agent_entry" ] || continue
        name=$(basename "$agent_entry")
        base="${name%.md}"
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

    for link in "$AGENT_DISCOVERY"/*; do
        [ -L "$link" ] && [ ! -e "$link" ] && rm "$link"
    done

    ok "Agents: $agent_link_count discovery symlinks"

    # Trae/Claude/Cursor: create per-hook discovery symlinks (from local hooks/)
    HOOK_DISCOVERY="$CONFIG_ROOT/hooks"

    for hook_entry in "$PLUGIN_ROOT/hooks"/*; do
        [ -e "$hook_entry" ] || continue
        name=$(basename "$hook_entry")
        target="$HOOK_DISCOVERY/$name"
        [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
    done

    hook_link_count=0
    for hook_entry in "$PLUGIN_ROOT/hooks"/*; do
        [ -e "$hook_entry" ] || continue
        name=$(basename "$hook_entry")
        target="$HOOK_DISCOVERY/$name"
        ln -sfn "$(realpath "$hook_entry")" "$target"
        hook_link_count=$((hook_link_count + 1))
    done

    for link in "$HOOK_DISCOVERY"/*; do
        [ -L "$link" ] && [ ! -e "$link" ] && rm "$link"
    done

    ok "Hooks: $hook_link_count discovery symlinks"

	    # --- Generate .claude/settings.json (Claude tool only) ---
	    if [ "$TOOL" = "claude" ]; then
	        SETTINGS_TARGET="$CONFIG_ROOT/settings.json"
	        PLUGIN_ROOT_ABS="$(realpath "$PLUGIN_ROOT")"
	        python3 -c "
import json
with open('$PLUGIN_ROOT/hooks/hooks.json') as f:
    content = f.read()
content = content.replace('\${CLAUDE_PLUGIN_ROOT}', '$PLUGIN_ROOT_ABS')
config = json.loads(content)
with open('$SETTINGS_TARGET', 'w') as f:
    json.dump(config, f, indent=2)
" 2>/dev/null
	        if [ -f "$SETTINGS_TARGET" ]; then
	            ok "Generated settings.json (hooks activated)"
	        else
	            warn "Failed to generate settings.json"
	        fi
	    fi
fi
echo ""

# --- Step 4: Setup asc-devkit ---
step "[4/5] Setting up asc-devkit..."
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
    python3 "$SHARED_SKILL_ROOT/ascendc-docs-search/scripts/clean_markdown.py" --dir "$ASC_DEVKIT_DIR" --no-backup --quiet > /dev/null 2>&1 || warn "markdown cleanup failed"
fi

if [ "$LEVEL" = "global" ] && [ -d "$ASC_DEVKIT_DIR" ]; then
    ln -sfn "$(realpath "$ASC_DEVKIT_DIR")" "$CONFIG_ROOT/asc-devkit"
    ok "asc-devkit → $CONFIG_ROOT/"
fi
echo ""

# --- Step 5: Health check ---
step "[5/5] Running health check..."
health_ok=true
health_errors=""

for sub in skills agents hooks; do
  target="$CANNBOT_DIR/$sub"
  if [ -d "$target" ]; then
    count=$(ls -d "$target"/* 2>/dev/null | wc -l)
    [ "$count" -eq 0 ] && { health_errors="${health_errors}\n  ${YELLOW}⚠${NC} $sub/ is empty"; }
  else
    health_errors="${health_errors}\n  ${RED}✗${NC} $sub/ missing"
    health_ok=false
  fi
done

if [ ! -d "$ASC_DEVKIT_DIR" ]; then
  health_errors="${health_errors}\n  ${YELLOW}⚠${NC} asc-devkit not available"
fi
if [ "$LEVEL" = "global" ] && [ ! -d "$CONFIG_ROOT/asc-devkit" ]; then
  health_errors="${health_errors}\n  ${YELLOW}⚠${NC} asc-devkit symlink missing in $CONFIG_ROOT"
fi

if [ "$LEVEL" = "project" ]; then
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ]; then
        [ -f "$PWD/AGENTS.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} AGENTS.md missing in current directory"; health_ok=false; }
    else
        [ -f "$PWD/CLAUDE.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} CLAUDE.md missing in current directory"; health_ok=false; }
    fi
else
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ]; then
        [ -f "$CONFIG_ROOT/AGENTS.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} AGENTS.md missing"; health_ok=false; }
    else
        [ -f "$CONFIG_ROOT/CLAUDE.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} CLAUDE.md missing"; health_ok=false; }
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

HOOKS_JSON="[]"
if [ -d "$CANNBOT_DIR/hooks" ]; then
  HOOKS_JSON=$(ls -d "$CANNBOT_DIR/hooks"/* 2>/dev/null | while read d; do
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
  "installed_hooks": $HOOKS_JSON,
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
echo -e "  ${GREEN}${BOLD}✓ CANNBot ascendc-ops-lab-developer installed successfully!${NC}"
echo ""
echo -e "  ${BOLD}Quick Start:${NC}"
if [ "$TOOL" = "opencode" ]; then
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}opencode${NC}"
  echo -e "  ${CYAN}2.${NC} 输入需求: ${GREEN}${BOLD}生成ascendC算子，npu=0，算子描述文件为 /path/to/op.py，输出到 /path/to/output/${NC}"
elif [ "$TOOL" = "trae" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 CLI/IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} 输入需求: ${GREEN}${BOLD}生成ascendC算子，npu=0，算子描述文件为 /path/to/op.py，输出到 /path/to/output/${NC}"
elif [ "$TOOL" = "cursor" ]; then
  echo -e "  ${CYAN}1.${NC} 启动 Cursor IDE"
  echo -e "  ${CYAN}2.${NC} 输入需求: ${GREEN}${BOLD}生成ascendC算子，npu=0，算子描述文件为 /path/to/op.py，输出到 /path/to/output/${NC}"
else
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}claude${NC}"
  echo -e "  ${CYAN}2.${NC} 输入需求: ${GREEN}${BOLD}生成ascendC算子，npu=0，算子描述文件为 /path/to/op.py，输出到 /path/to/output/${NC}"
fi
echo ""
