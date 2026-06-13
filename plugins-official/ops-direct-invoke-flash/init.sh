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
VERSION="1.2.2"

# --- Plugin-specific config ---
# This plugin is self-contained: a single skill bundled under skills/, plus a
# bundled review subagent under agents/.
SKILL_NAME="ops-direct-invoke-flash"
# Skill(s) installed by this plugin (bundled locally under skills/).
# Consumed by the CI install checks (INCLUDED_SKILLS / INCLUDED_AGENT_PATTERN).
INCLUDED_SKILLS="ops-direct-invoke-flash"
# Glob pattern for agents bundled under agents/.
INCLUDED_AGENT_PATTERN="ops-direct-invoke-flash-*"

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
  echo -e "  ${BOLD}Ascend C Kernel 从零构建 Skill (Flash)${NC}"
  echo ""
}

show_help() {
    cat << EOF
CANNBot - ops-direct-invoke-flash Skill Installer

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
  init.sh project claude /path/to/proj # Project-level, Claude Code, custom path

Installation paths (CANNBot brand):
  OpenCode: .opencode/skills/
  Claude:   .claude/skills/
  Trae IDE:     .trae/skills/
  Trae Plugin:  .marscode/skills/
  Trae CLI:     .traecli/skills/
  Cursor:       .cursor/skills/
  Copilot:      .github/skills/      (project)  /  ~/.copilot/skills/  (global)

After installation, launch your tool and run:
  /${SKILL_NAME} <源文件或描述>
EOF
}

LEVEL="project"
TOOL="opencode"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$SCRIPT_DIR"
# Skill source: bundled inside this plugin (self-contained, no shared ops/ dependency)
LOCAL_SKILL_ROOT="$PLUGIN_ROOT/skills"
# Agent source: review subagent bundled inside this plugin
LOCAL_AGENT_ROOT="$PLUGIN_ROOT/agents"
# Rules file bundled at plugin root (installed as AGENTS.md / CLAUDE.md)
LOCAL_RULES_FILE="$PLUGIN_ROOT/AGENTS.md"

# Reference repositories fetched at install time (shallow, latest). The workflow
# uses asc-devkit (API docs / examples) and cann-samples (operator samples).
# Cloned into the plugin root and git-ignored; install degrades gracefully offline.
ASC_DEVKIT_DIR="$PLUGIN_ROOT/asc-devkit"
CANN_SAMPLES_DIR="$PLUGIN_ROOT/cann-samples"
# name=git-url pairs (kept as a plain array for bash 3.2 compatibility)
REPO_SPECS=(
    "asc-devkit=https://gitcode.com/cann/asc-devkit.git"
    "cann-samples=https://gitcode.com/cann/cann-samples.git"
)

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
    # Project-level: default to current directory, allow override via install_path arg
    if [ -n "$INSTALL_PATH" ]; then
        INSTALL_BASE="$(cd "$INSTALL_PATH" && pwd)"
    else
        INSTALL_BASE="$PWD"
    fi

    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$INSTALL_BASE/.opencode"
    elif [ "$TOOL" = "trae" ]; then
        detect_trae_variant
        case "$TRAE_VARIANT" in
            plugin) CONFIG_ROOT="$INSTALL_BASE/.marscode" ;;
            cli)    CONFIG_ROOT="$INSTALL_BASE/.traecli" ;;
            *)      CONFIG_ROOT="$INSTALL_BASE/.trae" ;;
        esac
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_ROOT="$INSTALL_BASE/.cursor"
    elif [ "$TOOL" = "copilot" ]; then
        CONFIG_ROOT="$INSTALL_BASE/.github"
    else
        CONFIG_ROOT="$INSTALL_BASE/.claude"
    fi
fi

CANNBOT_DIR="$CONFIG_ROOT"

# Clean up legacy cannbot subdirectory from previous installations
if [ -e "$CONFIG_ROOT/$BRAND" ] || [ -L "$CONFIG_ROOT/$BRAND" ]; then
    rm -rf "$CONFIG_ROOT/$BRAND"
fi

show_banner
echo "  Tool:      $TOOL"
echo "  Level:     $LEVEL"
echo "  Path:      $CONFIG_ROOT"
echo ""

if [ "$TOOL" = "trae" ]; then
    case "$TRAE_VARIANT" in
        ide)     info "Detected: TRAE IDE (.trae-cn / .trae)" ;;
        plugin)  info "Detected: TRAE Plugin (.marscode)" ;;
        cli)     info "Detected: TRAE CLI (.traecli)" ;;
        unknown)
            warn "TRAE variant not detected; defaulting to IDE path"
            warn "If you use TRAE Plugin, ensure ~/.marscode exists before re-running"
            warn "If you use TRAE CLI, ensure ~/.traecli exists before re-running"
            ;;
    esac
    echo ""
fi

# --- Sanity check: skill source must exist ---
if [ ! -d "$LOCAL_SKILL_ROOT/$SKILL_NAME" ]; then
    err "Skill source not found: $LOCAL_SKILL_ROOT/$SKILL_NAME"
    err "请在插件目录内运行 init.sh（plugins-official/ops-direct-invoke-flash/）"
    exit 1
fi

# --- Step 0: Installation plan ---
step "[0/5] Checking items to be installed..."
echo ""
echo -e "${BOLD}以下内容将被安装/替换：${NC}"
echo ""
echo -e "${CYAN}Skill (1 项)：${NC}"
target="$CANNBOT_DIR/skills/$SKILL_NAME"
if [ -e "$target" ] || [ -L "$target" ]; then
    echo -e "  ${YELLOW}$SKILL_NAME${NC} (将被替换)"
else
    echo -e "  ${GREEN}$SKILL_NAME${NC} (将创建)"
fi
echo ""

# Agents bundled under agents/ matching the include pattern
plan_agents=()
for agent_file in "$LOCAL_AGENT_ROOT"/$INCLUDED_AGENT_PATTERN.md; do
    [ -f "$agent_file" ] || continue
    plan_agents+=("$(basename "$agent_file")")
done
if [ ${#plan_agents[@]} -gt 0 ]; then
    echo -e "${CYAN}Agent (${#plan_agents[@]} 项)：${NC}"
    for agent_base in "${plan_agents[@]}"; do
        if [ -e "$CANNBOT_DIR/agents/$agent_base" ] || [ -L "$CANNBOT_DIR/agents/$agent_base" ]; then
            echo -e "  ${YELLOW}${agent_base%.md}${NC} (将被替换)"
        else
            echo -e "  ${GREEN}${agent_base%.md}${NC} (将创建)"
        fi
    done
    echo ""
fi

# Rules file (AGENTS.md, or CLAUDE.md for Claude)
if [ "$TOOL" = "claude" ]; then plan_rules="CLAUDE.md"; else plan_rules="AGENTS.md"; fi
if [ "$LEVEL" = "global" ]; then plan_rules_dir="$CONFIG_ROOT"; else plan_rules_dir="$INSTALL_BASE"; fi
echo -e "${CYAN}规则文件 (1 项)：${NC}"
echo -e "  ${GREEN}${plan_rules}${NC} → ${DIM}${plan_rules_dir}${NC}"
echo ""

# Reference repositories (fetched / updated at install time)
echo -e "${CYAN}参考仓库 (${#REPO_SPECS[@]} 项，首装拉取，离线则跳过)：${NC}"
for spec in "${REPO_SPECS[@]}"; do
    repo_name="${spec%%=*}"
    if [ -d "$PLUGIN_ROOT/$repo_name" ]; then
        echo -e "  ${YELLOW}${repo_name}${NC} (将更新)"
    else
        echo -e "  ${GREEN}${repo_name}${NC} (将拉取)"
    fi
done
echo ""

echo -e "${BOLD}${YELLOW}注意：仅安装/替换上述内容，不影响其他已存在的 skills / agents${NC}"
echo ""
ok "开始安装..."
echo ""

# --- Step 1: Prepare skills directory ---
step "[1/5] Preparing skills directory..."
mkdir -p "$CANNBOT_DIR/skills"
ok "Prepared: $CANNBOT_DIR/skills"
echo ""

# --- Step 2: Install skill (per-item symlink) ---
step "[2/5] Installing skill..."
target="$CANNBOT_DIR/skills/$SKILL_NAME"
if [ -e "$target" ] || [ -L "$target" ]; then
    rm -rf "$target"
fi
ln -sfn "$(realpath "$LOCAL_SKILL_ROOT/$SKILL_NAME")" "$target"
ok "$SKILL_NAME → $target"
echo ""

# --- Step 3: Install review subagent(s) + rules file ---
step "[3/5] Installing agent & rules file..."
installed_agents=()
if [ -d "$LOCAL_AGENT_ROOT" ]; then
    mkdir -p "$CANNBOT_DIR/agents"
    for agent_file in "$LOCAL_AGENT_ROOT"/$INCLUDED_AGENT_PATTERN.md; do
        [ -f "$agent_file" ] || continue
        agent_base="$(basename "$agent_file")"
        agent_target="$CANNBOT_DIR/agents/$agent_base"
        if [ -e "$agent_target" ] || [ -L "$agent_target" ]; then
            rm -rf "$agent_target"
        fi
        ln -sfn "$(realpath "$agent_file")" "$agent_target"
        installed_agents+=("${agent_base%.md}")
        ok "agent: $agent_base"
    done
fi

# Rules file: project mode → project root (INSTALL_BASE); global mode → config root.
# Claude reads CLAUDE.md; all other tools read AGENTS.md. Installed as a symlink
# back to the plugin's bundled AGENTS.md.
if [ "$LEVEL" = "global" ]; then
    rules_dest_dir="$CONFIG_ROOT"
else
    rules_dest_dir="$INSTALL_BASE"
fi
if [ "$TOOL" = "claude" ]; then
    rules_name="CLAUDE.md"
else
    rules_name="AGENTS.md"
fi
if [ -f "$LOCAL_RULES_FILE" ]; then
    rules_target="$rules_dest_dir/$rules_name"
    src_real="$(realpath "$LOCAL_RULES_FILE")"
    tgt_real="$(realpath "$rules_target" 2>/dev/null || true)"
    if [ "$tgt_real" = "$src_real" ]; then
        # Running in-place (target already is/points to the bundled source).
        ok "$rules_name already in place"
    else
        # Preserve a user's pre-existing real file before replacing it.
        # Only proceed with the replacement if the backup succeeds — never
        # delete the original on a failed backup.
        rules_ok=1
        if [ -e "$rules_target" ] && [ ! -L "$rules_target" ]; then
            backup="$rules_target.bak.$(date +%Y%m%d%H%M%S)"
            if cp "$rules_target" "$backup"; then
                warn "已备份原有 $rules_name → $(basename "$backup")"
            else
                err "无法备份原有 $rules_name，已跳过安装以保护原文件：$rules_target"
                rules_ok=0
            fi
        fi
        if [ "$rules_ok" = 1 ]; then
            rm -f "$rules_target"
            ln -sfn "$src_real" "$rules_target"
            ok "$rules_name → $rules_target"
        fi
    fi
fi
echo ""

# --- Step 4: Set up reference repositories (asc-devkit, cann-samples) ---
step "[4/5] Setting up reference repositories..."
installed_repos=()
for spec in "${REPO_SPECS[@]}"; do
    repo_name="${spec%%=*}"
    repo_url="${spec#*=}"
    repo_dir="$PLUGIN_ROOT/$repo_name"

    if [ -d "$repo_dir/.git" ]; then
        # Existing real clone: best-effort refresh, never fail the install.
        ( cd "$repo_dir" && git pull --quiet 2>/dev/null ) || warn "$repo_name: git pull 失败，使用现有版本"
        ok "$repo_name updated"
    elif [ -d "$repo_dir" ]; then
        # Directory exists without .git (e.g. CI fake repo) — leave it as-is.
        ok "$repo_name present"
    else
        # Shallow clone latest; degrade gracefully when offline.
        if git clone --quiet --depth 1 "$repo_url" "$repo_dir" 2>/dev/null; then
            ok "$repo_name cloned"
        else
            warn "$repo_name: git clone 失败（离线？），已跳过"
        fi
    fi
    [ -d "$repo_dir" ] && installed_repos+=("$repo_name")
done

# Make repos discoverable outside the plugin dir. Literal repo names are used in
# the symlink targets below on purpose: the CI install test statically greps this
# script's symlink-into-config-root lines to learn the repo names, so the names
# must appear as literals (not via a loop variable).
#   - global mode → symlink into CONFIG_ROOT
#   - project mode with a custom INSTALL_BASE → symlink into INSTALL_BASE
if [ "$LEVEL" = "global" ]; then
    if [ -d "$ASC_DEVKIT_DIR" ]; then
        ln -sfn "$(realpath "$ASC_DEVKIT_DIR")" "$CONFIG_ROOT/asc-devkit"
        ok "asc-devkit → $CONFIG_ROOT/"
    fi
    if [ -d "$CANN_SAMPLES_DIR" ]; then
        ln -sfn "$(realpath "$CANN_SAMPLES_DIR")" "$CONFIG_ROOT/cann-samples"
        ok "cann-samples → $CONFIG_ROOT/"
    fi
elif [ -n "${INSTALL_BASE:-}" ] && [ "$INSTALL_BASE" != "$PLUGIN_ROOT" ]; then
    if [ -d "$ASC_DEVKIT_DIR" ]; then
        ln -sfn "$(realpath "$ASC_DEVKIT_DIR")" "$INSTALL_BASE/asc-devkit"
        ok "asc-devkit → $INSTALL_BASE/"
    fi
    if [ -d "$CANN_SAMPLES_DIR" ]; then
        ln -sfn "$(realpath "$CANN_SAMPLES_DIR")" "$INSTALL_BASE/cann-samples"
        ok "cann-samples → $INSTALL_BASE/"
    fi
fi
echo ""

# --- Step 5: Health check & manifest ---
step "[5/5] Running health check..."
health_ok=true
health_errors=""

# Reference repos are best-effort; missing ones are warnings, not failures.
for spec in "${REPO_SPECS[@]}"; do
    repo_name="${spec%%=*}"
    if [ ! -d "$PLUGIN_ROOT/$repo_name" ]; then
        health_errors="${health_errors}\n  ${YELLOW}⚠${NC} 参考仓库缺失（离线安装？）：$repo_name"
    elif [ "$LEVEL" = "global" ] && [ ! -d "$CONFIG_ROOT/$repo_name" ]; then
        health_errors="${health_errors}\n  ${YELLOW}⚠${NC} $repo_name 软链缺失：$CONFIG_ROOT/$repo_name"
    fi
done

if [ ! -e "$target/SKILL.md" ]; then
    health_errors="${health_errors}\n  ${RED}✗${NC} SKILL.md not reachable via symlink"
    health_ok=false
fi

# Ensure bundled hooks are executable
for hook in "$LOCAL_SKILL_ROOT/$SKILL_NAME"/hooks/*.sh; do
    [ -f "$hook" ] || continue
    chmod +x "$hook" 2>/dev/null || warn "无法设置可执行权限: $(basename "$hook")"
done

# Build installed_agents JSON array for the manifest
if [ ${#installed_agents[@]} -gt 0 ]; then
    AGENTS_JSON="$(printf '"%s",' "${installed_agents[@]}")"
    AGENTS_JSON="[${AGENTS_JSON%,}]"
else
    AGENTS_JSON="[]"
fi

# Build installed_repos JSON array (reference repos actually present)
if [ ${#installed_repos[@]} -gt 0 ]; then
    REPOS_JSON="$(printf '"%s",' "${installed_repos[@]}")"
    REPOS_JSON="[${REPOS_JSON%,}]"
else
    REPOS_JSON="[]"
fi

# Generate brand manifest
MANIFEST="$CONFIG_ROOT/cannbot-manifest.json"
cat > "$MANIFEST" << MANIFEST_EOF
{
  "brand": "CANNBot",
  "version": "$VERSION",
  "team": "$(basename "$SCRIPT_DIR")",
  "level": "$LEVEL",
  "tool": "$TOOL",
  "installed_skills": ["$SKILL_NAME"],
  "installed_agents": $AGENTS_JSON,
  "installed_repos": $REPOS_JSON,
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
echo -e "  ${GREEN}${BOLD}✓ CANNBot ops-direct-invoke-flash installed successfully!${NC}"
echo ""
echo -e "  ${BOLD}Quick Start:${NC}"
if [ "$TOOL" = "opencode" ]; then
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}opencode${NC}"
elif [ "$TOOL" = "trae" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 CLI / IDE 启动${NC}"
elif [ "$TOOL" = "cursor" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 Cursor IDE 启动${NC}"
elif [ "$TOOL" = "copilot" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 GitHub Copilot CLI / IDE 启动${NC}"
else
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}claude${NC}"
fi
echo -e "  ${CYAN}2.${NC} 运行命令: ${GREEN}${BOLD}/${SKILL_NAME} <源文件或描述>${NC}"
echo -e "  ${DIM}     示例：/${SKILL_NAME} 帮我实现一个 abs 算子，float16，shape [1,128]/[4,2048]${NC}"
echo ""
