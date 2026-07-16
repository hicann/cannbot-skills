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
# cuda2ascend 基类 init（构造函数）
# 职责：
#   1. 创建 .cannbot 中间目录
#   2. 链接配置文件 AGENTS.md 到工作区
#   3. 链接子 Agent（architect / developer / qa，全部链成平级）
#   4. 按 Agent 中注册的 skill 链接 skill 目录（插件 skills/ 优先，再 ops/、infra/）
#      并把 opencode 权限插件链接到 .opencode/plugin/
#   5. clone asc-devkit 仓到 .cannbot/ 下（可用 --asc-devkit 指定本地已有仓）
#   6. clone cann-samples 仓到 .cannbot/ 下（可用 --cann-samples 指定本地已有仓）
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

# Validate that a path exists (directory), exit with unified error if not.
# $1 = path to check  $2 = parameter name for error message
validate_path() {
    local path="$1"
    local param_name="$2"
    if [ ! -d "$path" ]; then
        err "$param_name path not found: $path"
        exit 1
    fi
}

# Safe install config file with backup and conflict handling.
# $1 = generated temp file path  $2 = target  $3 = display name  $4 = level
safe_install_file() {
    local tmpfile="$1"
    local target="$2"
    local name="$3"
    local level="$4"

    if [ -e "$target" ] && diff -q "$tmpfile" "$target" > /dev/null 2>&1; then
        info "$name already up to date"
        rm -f "$tmpfile"
        return 0
    fi

    if [ -e "$target" ] || [ -L "$target" ]; then
        local backup
        backup="${target}.bak.$(date +%Y%m%d_%H%M%S)"
        cp -a "$target" "$backup"
        warn "$name already exists, backed up to $(basename "$backup")"

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

    mv "$tmpfile" "$target"
    if [ "$level" = "global" ]; then
        ok "$name (absolute paths for global mode)"
    else
        ok "$name (absolute paths for project mode)"
    fi
}

BRAND="cannbot"
VERSION="1.0.0"

# ============================================================
# Third-party Repository Registry
# ============================================================
# Format: "name|url|ref|update_strategy|post_hook"
#   name: repo name (used in --{name} option and DIR variable)
#   url: git clone URL
#   ref: branch/tag/commit to checkout (empty = no checkout)
#   update_strategy: "fetch_merge" | "pull" | "pull_rebase"
#   post_hook: function to call after setup (empty = none)
REPO_REGISTRY=(
    "asc-devkit|https://gitcode.com/cann/asc-devkit.git|31f3ab38|fetch_merge|post_setup_asc_devkit"
    "cann-samples|https://gitcode.com/cann/cann-samples.git||pull|"
    "ops-tensor|https://gitcode.com/cann/ops-tensor.git||pull|"
)

# Post-setup hook for asc-devkit: clean markdown files
post_setup_asc_devkit() {
    local repo_dir="$1"
    CLEAN_SCRIPT="$SHARED_SKILL_ROOT/ascendc-docs-search/scripts/clean_markdown.py"
    if [ -f "$CLEAN_SCRIPT" ]; then
        python3 "$CLEAN_SCRIPT" --dir "$repo_dir" --no-backup --quiet > /dev/null 2>&1 \
            || warn "markdown cleanup failed"
    else
        warn "clean_markdown.py not found, skipping"
    fi
}

# Setup a third-party repository
# $1 = repo name  $2 = url  $3 = ref  $4 = update_strategy  $5 = post_hook  $6 = local_path (optional)
setup_repo() {
    local name="$1"
    local url="$2"
    local ref="$3"
    local strategy="$4"
    local post_hook="$5"
    local local_path="$6"
    
    local repo_dir="$CANNBOT_MID_DIR/$name"
    
    # Use local path if provided
    if [ -n "$local_path" ]; then
        validate_path "$local_path" "--$name"
        rm -rf "$repo_dir"
        ln -sfn "$(realpath "$local_path")" "$repo_dir"
        ok "$name → $(realpath "$local_path") (local, linked)"
        
        # Run post-hook if specified
        if [ -n "$post_hook" ] && type "$post_hook" >/dev/null 2>&1; then
            "$post_hook" "$repo_dir"
        fi
        return 0
    fi
    
    # Update existing repo
    if [ -d "$repo_dir/.git" ]; then
        cd "$repo_dir"
        git checkout . 2>/dev/null || true
        
        case "$strategy" in
            fetch_merge)
                git fetch --quiet 2>/dev/null || true
                if [ -n "$ref" ]; then
                    git checkout --quiet "$ref" 2>/dev/null || warn "checkout $ref failed, using existing"
                fi
                ;;
            pull)
                git pull --quiet 2>/dev/null || warn "git pull failed, using existing version"
                ;;
            pull_rebase)
                git pull --quiet --rebase 2>/dev/null || warn "git pull --rebase failed, using existing version"
                ;;
        esac
        
        cd "$SCRIPT_DIR"
        if [ -n "$ref" ]; then
            ok "$name updated (@$ref)"
        else
            ok "$name updated"
        fi
    else
        # Clone new repo
        local clone_opts=""
        [ "$strategy" = "pull" ] && clone_opts="--depth 1"
        
        if git clone --quiet $clone_opts "$url" "$repo_dir" 2>/dev/null; then
            if [ -n "$ref" ]; then
                cd "$repo_dir" && git checkout --quiet "$ref" 2>/dev/null || warn "checkout $ref failed"
                cd "$SCRIPT_DIR"
                ok "$name cloned (@$ref)"
            else
                ok "$name cloned"
            fi
        else
            warn "git clone failed, skipping $name"
            return 0
        fi
    fi
    
    # Run post-hook if specified
    if [ -n "$post_hook" ] && type "$post_hook" >/dev/null 2>&1; then
        "$post_hook" "$repo_dir"
    fi
}

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAM_NAME="$(basename "$SCRIPT_DIR")"
PLUGIN_ROOT="$SCRIPT_DIR"
LOCAL_AGENT_ROOT="$PLUGIN_ROOT/agents"
LOCAL_SKILL_ROOT="$PLUGIN_ROOT/skills"
SHARED_SKILL_ROOT="$(cd "$PLUGIN_ROOT/../../ops" 2>/dev/null && pwd || true)"
INFRA_SKILL_ROOT="$(cd "$PLUGIN_ROOT/../../infra" 2>/dev/null && pwd || true)"

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
  echo -e "  ${BOLD}cuda2ascend Installer${NC}"
  echo ""
}

show_help() {
    cat << EOF
cuda2ascend - 基类工作流安装器

Usage: init.sh [level] [tool] [install_path] [options]

目标工具仅支持 OpenCode。

Arguments:
  level        - 安装级别: "project" (默认) 或 "global"
  tool         - 目标工具 (目前仅支持 "opencode")
  install_path - project 级安装目录 (默认: 当前工作目录)

Options:
  --override <dir>        - 子仓覆写根目录：等价于 --override-skills <dir>/skills
  --override-skills <dir> - 用 <dir> 下的 skill 覆盖已链接的同名 skill（基类没有的则新增）
  --repo <name>:<path>    - 使用本地已 clone 的第三方仓（直接链接，不再 clone）
                            可多次指定，如：--repo asc-devkit:... --repo cann-samples:...
                            支持的仓库名：asc-devkit / cann-samples / ops-tensor
  --help                  - 显示本帮助

Examples:
  init.sh                                  # project 级
  init.sh global                           # global 级
  init.sh project /path/to/proj            # 指定安装目录
  init.sh --repo asc-devkit:~/repos/asc-devkit --repo cann-samples:~/repos/cann-samples
  init.sh project /path/to/repo --override /path/to/repo/agent

Installation paths:
  .opencode/{skills,agents}/   + AGENTS.md in project root

中间文件目录:
  .cannbot/              流程中间文件、状态文件
  .cannbot/asc-devkit    asc-devkit 仓
  .cannbot/cann-samples  cann-samples 仓
  .cannbot/ops-tensor    ops-tensor 仓
EOF
}

# --- Parse arguments ---
LEVEL="project"
TOOL="opencode"   # 仅支持 opencode
INSTALL_PATH=""
OVERRIDE_SKILL_DIR=""
OVERRIDE_ROOT=""

# Local path overrides for registered repos, keyed by repo name.
declare -A REPO_LOCAL

# Check whether a name is a registered repo (returns 0 if yes)
is_registered_repo() {
    local target="$1"
    local entry
    for entry in "${REPO_REGISTRY[@]}"; do
        [ "${entry%%|*}" = "$target" ] && return 0
    done
    return 1
}

POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --help)          show_help; exit 0 ;;
        --override)         OVERRIDE_ROOT="$2"; shift 2; continue ;;
        --override=*)       OVERRIDE_ROOT="${1#*=}"; shift; continue ;;
        --override-skills)   OVERRIDE_SKILL_DIR="$2"; shift 2; continue ;;
        --override-skills=*) OVERRIDE_SKILL_DIR="${1#*=}"; shift; continue ;;
        --repo)
            local_arg="$2"; shift 2
            if [[ "$local_arg" != *:* ]]; then
                err "--repo format invalid: must be name:/path, got: $local_arg"
                exit 1
            fi
            local_name="${local_arg%%:*}"
            local_path="${local_arg#*:}"
            if ! is_registered_repo "$local_name"; then
                err "--repo unknown name: $local_name (valid: ${REPO_REGISTRY[*]%%|*})"
                exit 1
            fi
            REPO_LOCAL["$local_name"]="$local_path"
            continue ;;
        --repo=*)
            local_arg="${1#*=}"; shift
            if [[ "$local_arg" != *:* ]]; then
                err "--repo format invalid: must be name:/path, got: $local_arg"
                exit 1
            fi
            local_name="${local_arg%%:*}"
            local_path="${local_arg#*:}"
            if ! is_registered_repo "$local_name"; then
                err "--repo unknown name: $local_name (valid: ${REPO_REGISTRY[*]%%|*})"
                exit 1
            fi
            REPO_LOCAL["$local_name"]="$local_path"
            continue ;;
        --*)
            err "unknown option: $1"
            exit 1 ;;
        global|project)  LEVEL="$1"; shift; continue ;;
        opencode)        # 兼容旧用法，仅支持 opencode
                        shift; continue ;;
        claude|trae|cursor|copilot|codearts)
                        err "暂不支持 $1，当前仅支持 opencode"
                        exit 1 ;;
        *) POSITIONAL+=("$1"); shift; continue ;;
    esac
done

# Last unrecognized positional arg is install_path
if [ ${#POSITIONAL[@]} -gt 0 ]; then
    INSTALL_PATH="${POSITIONAL[${#POSITIONAL[@]}-1]}"
fi

# --override <dir> is shorthand for --override-skills <dir>/skills.
# Explicit --override-skills takes precedence if both given.
if [ -n "$OVERRIDE_ROOT" ]; then
    validate_path "$OVERRIDE_ROOT" "--override"
    if [ -z "$OVERRIDE_SKILL_DIR" ]; then
        if [ -d "$OVERRIDE_ROOT/skills" ]; then
            OVERRIDE_SKILL_DIR="$OVERRIDE_ROOT/skills"
        else
            err "--override path has no skills/ subdirectory: $OVERRIDE_ROOT"
            exit 1
        fi
    fi
fi

# Normalize & validate override dir (absolute path for stable symlinks)
if [ -n "$OVERRIDE_SKILL_DIR" ]; then
    validate_path "$OVERRIDE_SKILL_DIR" "--override-skills"
    OVERRIDE_SKILL_DIR="$(cd "$OVERRIDE_SKILL_DIR" && pwd)"
fi

# --- Determine config root (OpenCode) ---
if [ "$LEVEL" = "global" ]; then
    CONFIG_ROOT="$HOME/.config/opencode"
    INSTALL_BASE="$HOME"
else
    if [ -n "$INSTALL_PATH" ]; then
        validate_path "$INSTALL_PATH" "install_path"
        INSTALL_BASE="$(cd "$INSTALL_PATH" && pwd)"
    else
        INSTALL_BASE="$PWD"
    fi
    CONFIG_ROOT="$INSTALL_BASE/.opencode"
fi

CANNBOT_DIR="$CONFIG_ROOT"
CANNBOT_MID_DIR="$INSTALL_BASE/.cannbot"

# --- Clean up legacy ---
if [ -e "$CONFIG_ROOT/$BRAND" ] || [ -L "$CONFIG_ROOT/$BRAND" ]; then
    rm -rf "$CONFIG_ROOT/$BRAND"
fi
if [ -L "$CONFIG_ROOT/teams" ]; then
    rm -f "$CONFIG_ROOT/teams"
fi

# ============================================================
# Dependency resolution
# ============================================================

# Parse YAML list from a config file (handles `skills:` / `agents:` sections)
parse_yaml_list() {
    local file="$1"
    local key="$2"
    awk -v key="$key" '
        $0 ~ "^"key":" { flag=1; next }
        /^[^ ]/ { flag=0 }
        flag && /^ +- / { sub(/^ +- /, ""); print }
    ' "$file" 2>/dev/null | grep -v '^$' || true
}

# Collect all agent files under agents/ (recurse into subdirs), flattened.
collect_agents() {
    AGENT_FILES=()
    while IFS= read -r f; do
        AGENT_FILES+=("$f")
    done < <(find "$LOCAL_AGENT_ROOT" -type f -name '*.md' 2>/dev/null | sort)
}

# Resolve skill source path: local skills/ first, then shared ops/, then infra/.
resolve_skill_src() {
    local skill="$1"
    if [ -d "$LOCAL_SKILL_ROOT/$skill" ]; then
        echo "$LOCAL_SKILL_ROOT/$skill"; return
    fi
    if [ -n "$SHARED_SKILL_ROOT" ] && [ -d "$SHARED_SKILL_ROOT/$skill" ]; then
        echo "$SHARED_SKILL_ROOT/$skill"; return
    fi
    if [ -n "$INFRA_SKILL_ROOT" ] && [ -d "$INFRA_SKILL_ROOT/$skill" ]; then
        echo "$INFRA_SKILL_ROOT/$skill"; return
    fi
    echo ""
}

# Collect skills to link: union of
#   - all skills under the plugin's skills/ (linked by default), and
#   - skills registered by AGENTS.md and every agent's frontmatter `skills:`.
# Deduplicated.
collect_skills() {
    local raw=""
    # 1) All local skills under skills/ (default-linked)
    if [ -d "$LOCAL_SKILL_ROOT" ]; then
        for skill_dir in "$LOCAL_SKILL_ROOT"/*/; do
            [ -d "$skill_dir" ] || continue
            raw="$raw
$(basename "$skill_dir")"
        done
    fi
    # 2) Skills registered by AGENTS.md + each agent frontmatter
    local agents_file="$PLUGIN_ROOT/AGENTS.md"
    [ -f "$agents_file" ] && raw="$raw
$(parse_yaml_list "$agents_file" "skills")"
    for f in "${AGENT_FILES[@]}"; do
        local s
        s="$(parse_yaml_list "$f" "skills")"
        [ -n "$s" ] && raw="$raw
$s"
    done
    ALL_SKILLS=$(echo "$raw" | sort -u | grep -v '^$' || true)
    if [ -z "$ALL_SKILLS" ]; then
        ALL_SKILL_COUNT=0
    else
        ALL_SKILL_COUNT=$(echo "$ALL_SKILLS" | grep -c '.')
    fi
    return 0
}

# ============================================================
# Main
# ============================================================

show_banner
echo "  Team:      $TEAM_NAME"
echo "  Tool:      $TOOL"
echo "  Level:     $LEVEL"
echo "  Config:    $CONFIG_ROOT"
echo "  .cannbot:    $CANNBOT_MID_DIR"
echo ""

collect_agents
collect_skills

# --- Step 1: Create .cannbot directory ---
step "[1/6] Creating .cannbot directory..."
mkdir -p "$CANNBOT_MID_DIR"
ok ".cannbot → $CANNBOT_MID_DIR"
echo ""

# --- Step 2: Install config file (AGENTS.md) ---
step "[2/6] Installing configuration..."
mkdir -p "$CONFIG_ROOT"

config_src="$PLUGIN_ROOT/AGENTS.md"
config_name="AGENTS.md"
if [ "$LEVEL" = "project" ]; then
    config_target="$INSTALL_BASE/$config_name"
else
    config_target="$CONFIG_ROOT/$config_name"
fi

if [ ! -f "$config_src" ]; then
    warn "AGENTS.md not found in plugin root, skipping config"
elif [ "$config_src" = "$config_target" ]; then
    info "$config_name already at target location"
else
    ln -sf "$config_src" "$config_target"
    ok "$config_name → $config_target"
fi
echo ""

# --- Step 3: Link agents (flattened; architect / developer / qa) ---
step "[3/6] Linking agents..."
AGENTS_LINK_DIR="$CANNBOT_DIR/agents"
rm -rf "$AGENTS_LINK_DIR"
mkdir -p "$AGENTS_LINK_DIR"

agent_count=0
for f in "${AGENT_FILES[@]}"; do
    link_name="$(basename "$f")"
    if [ -e "$AGENTS_LINK_DIR/$link_name" ] || [ -L "$AGENTS_LINK_DIR/$link_name" ]; then
        warn "agent name conflict, overwriting: $link_name"
    fi
    ln -sfn "$(realpath "$f")" "$AGENTS_LINK_DIR/$link_name"
    agent_count=$((agent_count + 1))
done
ok "Agents: $agent_count linked (flattened)"
echo ""

# --- Step 4: Link skills registered by agents ---
step "[4/6] Linking skills..."
SKILLS_LINK_DIR="$CANNBOT_DIR/skills"
rm -rf "$SKILLS_LINK_DIR"
mkdir -p "$SKILLS_LINK_DIR"

skill_count=0
missing_skills=""
for skill in $ALL_SKILLS; do
    src="$(resolve_skill_src "$skill")"
    if [ -n "$src" ]; then
        ln -sfn "$(realpath "$src")" "$SKILLS_LINK_DIR/$skill"
        skill_count=$((skill_count + 1))
    else
        missing_skills="$missing_skills $skill"
    fi
done
ok "Skills: $skill_count linked"
[ -n "$missing_skills" ] && warn "Missing skills:$missing_skills"

# Override pass: replace/add domain skills from --override-skill dir (subclass repos).
# Runs right after the base skills are linked, so overriding skills win by same name.
if [ -n "$OVERRIDE_SKILL_DIR" ]; then
    override_replaced=0
    override_added=0
    for skill_dir in "$OVERRIDE_SKILL_DIR"/*/; do
        [ -d "$skill_dir" ] || continue
        [ -f "$skill_dir/SKILL.md" ] || continue   # only real skill dirs
        name="$(basename "$skill_dir")"
        target="$SKILLS_LINK_DIR/$name"
        if [ -e "$target" ] || [ -L "$target" ]; then
            rm -rf "$target"
            ln -sfn "$(realpath "$skill_dir")" "$target"
            override_replaced=$((override_replaced + 1))
        else
            ln -sfn "$(realpath "$skill_dir")" "$target"
            skill_count=$((skill_count + 1))
            override_added=$((override_added + 1))
        fi
    done
    if [ "$override_replaced" -eq 0 ] && [ "$override_added" -eq 0 ]; then
        warn "Override dir has no skills (expected subdirs with SKILL.md): $OVERRIDE_SKILL_DIR"
    else
        info "Overridden from $OVERRIDE_SKILL_DIR: $override_replaced replaced, $override_added added"
    fi
fi
echo ""

# --- Step 4+: Link opencode permission-guard plugin ---
# opencode 自动加载 <install>/.opencode/plugin/ 下的插件。基类把动态权限插件链进去。
OC_PLUGIN_SRC="$PLUGIN_ROOT/hooks/opencode/permission-guard.js"
if [ -f "$OC_PLUGIN_SRC" ]; then
    PLUGIN_LINK_DIR="$CONFIG_ROOT/plugin"
    mkdir -p "$PLUGIN_LINK_DIR"
    ln -sfn "$(realpath "$OC_PLUGIN_SRC")" "$PLUGIN_LINK_DIR/permission-guard.js"
    ok "opencode plugin: permission-guard.js linked"
else
    warn "opencode permission-guard.js not found, skipping"
fi
echo ""

# --- Step 5: Setup third-party repositories ---
step "[5/6] Setting up third-party repositories..."
repo_total=${#REPO_REGISTRY[@]}
repo_idx=1
for entry in "${REPO_REGISTRY[@]}"; do
    IFS='|' read -r name url ref strategy post_hook <<< "$entry"
    
    info "($repo_idx/$repo_total) $name"
    
    # Get local path override if provided
    local_path="${REPO_LOCAL[$name]:-}"
    
    # Setup the repo
    setup_repo "$name" "$url" "$ref" "$strategy" "$post_hook" "$local_path"
    
    # Export {NAME}_DIR variable for downstream use
    repo_dir="$CANNBOT_MID_DIR/$name"
    name_upper=$(echo "$name" | tr '[:lower:]' '[:upper:]' | tr '-' '_')
    eval "${name_upper}_DIR=\"$repo_dir\""
    
    repo_idx=$((repo_idx + 1))
done
echo ""

# --- Step 6: Generate manifest + health check ---
step "[6/6] Generating manifest and running health check..."
MANIFEST="$CONFIG_ROOT/cannbot-manifest.json"
SKILLS_JSON=$(echo "$ALL_SKILLS" | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
AGENTS_JSON=$(printf '%s\n' "${AGENT_FILES[@]}" | while read -r f; do [ -n "$f" ] && basename "$f" .md; done | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")

# Build repo paths JSON from registry (key = {name}_dir with dashes → underscores)
REPO_DIRS_JSON=""
for entry in "${REPO_REGISTRY[@]}"; do
    name="${entry%%|*}"
    key="$(echo "$name" | tr '-' '_')_dir"
    REPO_DIRS_JSON="${REPO_DIRS_JSON}  \"$key\": \"$CANNBOT_MID_DIR/$name\",
"
done

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
  "mid_dir": "$CANNBOT_MID_DIR",
${REPO_DIRS_JSON}  "install_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
MANIFEST_EOF

health_ok=true
health_errors=""

for sub in skills agents; do
  target="$CANNBOT_DIR/$sub"
  if [ -d "$target" ]; then
    count=$(ls -1A "$target" 2>/dev/null | wc -l)
    [ "$count" -eq 0 ] && health_errors="${health_errors}\n  ${YELLOW}⚠${NC} $sub/ is empty"
  else
    health_errors="${health_errors}\n  ${RED}✗${NC} $sub/ missing"
    health_ok=false
  fi
done

[ -d "$CANNBOT_MID_DIR" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} .cannbot/ missing"; health_ok=false; }

# Check all registered repos
for entry in "${REPO_REGISTRY[@]}"; do
    name="${entry%%|*}"
    repo_dir="$CANNBOT_MID_DIR/$name"
    [ -d "$repo_dir" ] || health_errors="${health_errors}\n  ${YELLOW}⚠${NC} $name not available"
done

[ -f "$config_target" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} $config_name missing"; health_ok=false; }
[ -f "$MANIFEST" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} Manifest generation failed"; health_ok=false; }

if [ "$health_ok" = true ] && [ -z "$health_errors" ]; then
  ok "All checks passed"
else
  echo -e "$health_errors"
  [ "$health_ok" = true ] && warn "Some warnings, see above" || err "Some checks failed, see above"
fi

# --- Summary ---
echo ""
echo -e "  ${GREEN}${BOLD}✓ $TEAM_NAME installed!${NC}"
echo -e "  ${DIM}Skills: $skill_count | Agents: $agent_count${NC}"
echo ""
echo -e "  ${BOLD}Quick Start:${NC}"
echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}opencode${NC}"
echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}帮我开发一个 abs 算子，支持 float16，shape 主要是 [1,128]、[4,2048]${NC}"
echo ""
