#!/usr/bin/env bash
set -euo pipefail

BRAND="autoresearch"
VERSION="1.0.1"
INCLUDED_SKILLS=""
INCLUDED_AGENT_PATTERN="ar-*"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$SCRIPT_DIR"
PROJECT_SOURCE="$PLUGIN_ROOT"

show_help() {
  cat <<'EOF'
AutoResearch 插件安装器

用法：
  init.sh [project|global] [claude] [install_path]

参数：
  project       项目级安装（默认），复制完整运行时到目标项目。
  global        全局注册 Claude/OpenCode 使用指导；运行态仍由项目级安装创建。
  claude        目标工具（支持 claude / opencode / trae / cursor / copilot）。
  install_path  目标项目根目录，默认是当前目录。

目标项目根会持有运行态：workspace/、ar_tasks/、.session_tasks/、
.task_dir_pointers/ 和 config.yaml。
EOF
}

LEVEL="project"
TOOL="claude"
INSTALL_PATH="$PWD"

for arg in "$@"; do
  case "$arg" in
    --help|-h) show_help; exit 0 ;;
    project|global) LEVEL="$arg" ;;
    claude) TOOL="$arg" ;;
    opencode) TOOL="$arg" ;;
    trae) TOOL="$arg" ;;
    cursor) TOOL="$arg" ;;
    copilot) TOOL="$arg" ;;
    *) INSTALL_PATH="$arg" ;;
  esac
done

normalize_path() {
  local path="$1"
  if [[ "$path" =~ ^[A-Za-z]:[\\/] ]]; then
    if command -v wslpath >/dev/null 2>&1; then
      wslpath -u "$path" 2>/dev/null && return 0
    fi
    if command -v cygpath >/dev/null 2>&1; then
      cygpath -u "$path" 2>/dev/null && return 0
    fi
    local drive="${path:0:1}"
    local rest="${path:2}"
    drive="$(printf '%s' "$drive" | tr 'A-Z' 'a-z')"
    rest="${rest//\\//}"
    printf '/mnt/%s%s\n' "$drive" "$rest"
    return 0
  fi
  printf '%s\n' "$path"
}

INSTALL_PATH="$(normalize_path "$INSTALL_PATH")"
if [ ! -d "$PROJECT_SOURCE/scripts" ] || [ ! -d "$PROJECT_SOURCE/src/op_autoresearch" ]; then
  echo "插件根目录缺少完整 op-autoresearch 项目" >&2
  exit 1
fi

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

# CANNBot's installer contract expects the selected tool's config root to
# expose a skills/ directory, even when this workflow installs no skills.
if [ "$LEVEL" = "global" ]; then
  if [ "$TOOL" = "trae" ]; then
    echo "Global installation is not supported for Trae; use project installation." >&2
    exit 2
  fi
  PROJECT_ROOT=""
  if [ "$TOOL" = "claude" ]; then
    CONFIG_ROOT="$HOME/.claude"
  elif [ "$TOOL" = "opencode" ]; then
    CONFIG_ROOT="$HOME/.config/opencode"
  elif [ "$TOOL" = "cursor" ]; then
    CONFIG_ROOT="$HOME/.cursor"
  elif [ "$TOOL" = "copilot" ]; then
    CONFIG_ROOT="$HOME/.copilot"
  fi
else
  mkdir -p "$INSTALL_PATH"
  PROJECT_ROOT="$(cd "$INSTALL_PATH" && pwd)"
  mkdir -p "$PROJECT_ROOT/workspace" "$PROJECT_ROOT/ar_tasks" \
    "$PROJECT_ROOT/.session_tasks" "$PROJECT_ROOT/.task_dir_pointers"
  if [ "$TOOL" = "claude" ]; then
    CONFIG_ROOT="$PROJECT_ROOT/.claude"
  elif [ "$TOOL" = "opencode" ]; then
    CONFIG_ROOT="$PROJECT_ROOT/.opencode"
  elif [ "$TOOL" = "trae" ]; then
    detect_trae_variant
    case "$TRAE_VARIANT" in
      plugin) CONFIG_ROOT="$PROJECT_ROOT/.marscode"; echo "Detected: TRAE Plugin" ;;
      cli) CONFIG_ROOT="$PROJECT_ROOT/.traecli"; echo "Detected: TRAE CLI" ;;
      *) CONFIG_ROOT="$PROJECT_ROOT/.trae"; echo "Detected: TRAE IDE" ;;
    esac
  elif [ "$TOOL" = "cursor" ]; then
    CONFIG_ROOT="$PROJECT_ROOT/.cursor"
  elif [ "$TOOL" = "copilot" ]; then
    CONFIG_ROOT="$PROJECT_ROOT/.github"
  fi
fi

backup_existing() {
  local target="$1"
  if [ -e "$target" ] || [ -L "$target" ]; then
    local backup="${target}.bak.$(date +%Y%m%d_%H%M%S)"
    mv "$target" "$backup"
    echo "  已备份 $(basename "$target") -> $(basename "$backup")"
  fi
}

copy_dir() {
  local src="$1"
  local target="$2"
  backup_existing "$target"
  cp -a "$src" "$target"
  echo "  已复制 $(basename "$target")"
}

copy_file() {
  local src="$1"
  local target="$2"
  backup_existing "$target"
  cp "$src" "$target"
  echo "  已安装 $(basename "$target")"
}

install_guidance() {
  local target="$1"
  local source_marker="<!-- AutoResearch source: $PLUGIN_ROOT -->"
  mkdir -p "$(dirname "$target")"
  if [ -e "$target" ] || [ -L "$target" ]; then
    if { cat "$PROJECT_SOURCE/AGENTS.md"; printf '\n%s\n' "$source_marker"; } | cmp -s - "$target"; then
      echo "  已保留 $(basename "$target")（内容未变化）"
      return
    fi
    backup_existing "$target"
  fi
  {
    cat "$PROJECT_SOURCE/AGENTS.md"
    printf '\n%s\n' "$source_marker"
  } > "$target"
  echo "  已安装 $(basename "$target")"
}

echo "AutoResearch $VERSION"
echo "  插件目录：$PLUGIN_ROOT"
echo "  安装级别：$LEVEL"
echo "  目标工具：$TOOL"

if [ "$LEVEL" = "project" ]; then
  echo "  项目目录：$PROJECT_ROOT"
  # 插件根目录就是完整的独立项目，不维护 app/ 或第二套裁剪运行时。
  if [ "$PROJECT_ROOT" != "$PROJECT_SOURCE" ]; then
    for name in .claude .opencode ar_examples scripts src tests; do
      if [ -d "$PROJECT_SOURCE/$name" ]; then
        copy_dir "$PROJECT_SOURCE/$name" "$PROJECT_ROOT/$name"
      fi
    done

    for name in .gitignore AUTORESEARCH.md LICENSE LICENSE-APACHE \
                NOTICE README.md README.OpenSource SOURCE_REVISION config.yaml \
                package.json pyproject.toml requirements-worker.txt; do
      if [ -f "$PROJECT_SOURCE/$name" ]; then
        copy_file "$PROJECT_SOURCE/$name" "$PROJECT_ROOT/$name"
      fi
    done
    install_guidance "$PROJECT_ROOT/AGENTS.md"
  fi
  if [ "$TOOL" = "claude" ]; then
    install_guidance "$PROJECT_ROOT/CLAUDE.md"
  fi
else
  echo "  配置目录：$CONFIG_ROOT"
  case "$TOOL" in
    claude) install_guidance "$CONFIG_ROOT/CLAUDE.md" ;;
    *) install_guidance "$CONFIG_ROOT/AGENTS.md" ;;
  esac
fi

mkdir -p "$CONFIG_ROOT/skills"
PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "需要 Python 3 才能写入安装清单。" >&2
  exit 1
fi
"$PYTHON_BIN" - "$CONFIG_ROOT/cannbot-manifest.json" "$BRAND" "$VERSION" \
  "$LEVEL" "$TOOL" "$CONFIG_ROOT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" <<'PY'
import json
import sys

path, brand, version, level, tool, brand_dir, install_time = sys.argv[1:]
with open(path, "w", encoding="utf-8") as stream:
    json.dump(
        {
            "brand": brand,
            "version": version,
            "team": brand,
            "level": level,
            "tool": tool,
            "installed_skills": [],
            "installed_agents": [],
            "brand_dir": brand_dir,
            "install_time": install_time,
        },
        stream,
        ensure_ascii=False,
        indent=2,
    )
    stream.write("\n")
PY

if [ "$LEVEL" = "project" ]; then
  echo "安装完成。请在项目根执行 python -m pip install -e \".[worker]\"，"
  echo "然后启动 $TOOL：$PROJECT_ROOT"
else
  echo "全局使用指导已注册。运行 AutoResearch 前，请在目标工作区执行 project 安装。"
fi
