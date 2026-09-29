#!/bin/bash
set -euo pipefail

PLUGIN="catlass-cpp-generator"
VERSION="0.1.0"
MARKER="generated-by-cannbot-catlass-cpp-generator-init"
SKILLS=(
    catlass-cpp-interface
    catlass-cpp-reference
    catlass-cpp-design
    catlass-cpp-develop
    catlass-cpp-test
    catlass-cpp-knowledge
)

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
action="install"
level="project"
tool="opencode"
install_path=""

usage() {
    cat <<'EOF'
Usage:
  init.sh [project|global] [tool] [install_path]
  init.sh --check [project|global] [tool] [install_path]
  init.sh --uninstall [project|global] [tool] [install_path]

tool: opencode|claude|trae|cursor|codex|copilot|codearts
EOF
}

for arg in "$@"; do
    case "$arg" in
        --check) action="check" ;;
        --uninstall) action="uninstall" ;;
        --help|-h) usage; exit 0 ;;
        project|global) level="$arg" ;;
        opencode|claude|trae|cursor|codex|copilot|codearts) tool="$arg" ;;
        *) install_path="$arg" ;;
    esac
done

if [ "$level" = "global" ] && [ "$tool" = "trae" ]; then
    echo "global Trae installation is not supported" >&2
    exit 2
fi

if [ -n "$install_path" ]; then
    project_root="$(cd "$install_path" && pwd)"
else
    project_root="$PWD"
fi

if [ "$level" = "global" ]; then
    case "$tool" in
        opencode) config_root="$HOME/.config/opencode" ;;
        claude) config_root="$HOME/.claude" ;;
        cursor) config_root="$HOME/.cursor" ;;
        copilot) config_root="$HOME/.copilot" ;;
        codearts) config_root="$HOME/.codeartsdoer" ;;
        *) config_root="$HOME/.agents" ;;
    esac
    config_target="$config_root/AGENTS.md"
    [ "$tool" = "claude" ] && config_target="$config_root/CLAUDE.md"
else
    case "$tool" in
        opencode) config_root="$project_root/.opencode" ;;
        claude) config_root="$project_root/.claude" ;;
        trae)
            if [ -d "$HOME/.marscode" ]; then
                config_root="$project_root/.marscode"
            elif [ -d "$HOME/.traecli" ]; then
                config_root="$project_root/.traecli"
            else
                config_root="$project_root/.trae"
            fi
            ;;
        cursor) config_root="$project_root/.cursor" ;;
        copilot) config_root="$project_root/.github" ;;
        codearts) config_root="$project_root/.codeartsdoer" ;;
        *) config_root="$project_root/.agents" ;;
    esac
    config_target="$project_root/AGENTS.md"
    [ "$tool" = "claude" ] && config_target="$project_root/CLAUDE.md"
fi

skills_root="$config_root/skills"
bundle_link="$config_root/catlass-cpp-knowledge"
manifest="$config_root/catlass-cpp-generator-manifest.json"

same_target() {
    [ -L "$1" ] && [ "$(readlink -f "$1")" = "$(readlink -f "$2")" ]
}

install_link() {
    local source="$1"
    local target="$2"
    if [ -e "$target" ] || [ -L "$target" ]; then
        if same_target "$target" "$source"; then
            return
        fi
        echo "refusing to replace foreign path: $target" >&2
        exit 1
    fi
    ln -s "$(realpath "$source")" "$target"
}

health_check() {
    local failed=0
    for name in "${SKILLS[@]}"; do
        if ! same_target "$skills_root/$name" "$script_dir/skills/$name"; then
            echo "BROKEN skill: $name" >&2
            failed=1
        fi
    done
    if ! same_target "$bundle_link" "$script_dir/knowledge"; then
        echo "BROKEN knowledge bundle link" >&2
        failed=1
    fi
    if [ ! -f "$config_target" ] || ! head -20 "$config_target" | grep -qF "$MARKER"; then
        echo "BROKEN generated config" >&2
        failed=1
    fi
    if [ ! -f "$manifest" ] || ! grep -q '"plugin": "catlass-cpp-generator"' "$manifest"; then
        echo "BROKEN manifest" >&2
        failed=1
    fi
    [ "$failed" -eq 0 ] || return 1
    echo "PASS: $PLUGIN installation is healthy"
}

uninstall() {
    for name in "${SKILLS[@]}"; do
        target="$skills_root/$name"
        if same_target "$target" "$script_dir/skills/$name"; then
            rm -f "$target"
        fi
    done
    if same_target "$bundle_link" "$script_dir/knowledge"; then
        rm -f "$bundle_link"
    fi
    if [ -f "$config_target" ] && head -20 "$config_target" | grep -qF "$MARKER"; then
        rm -f "$config_target"
    fi
    if [ -f "$manifest" ] && grep -q '"plugin": "catlass-cpp-generator"' "$manifest"; then
        rm -f "$manifest"
    fi
    rmdir "$skills_root" 2>/dev/null || true
    rmdir "$config_root" 2>/dev/null || true
    echo "PASS: removed only $PLUGIN-owned entries"
}

if [ "$action" = "check" ]; then
    health_check
    exit
fi
if [ "$action" = "uninstall" ]; then
    uninstall
    exit
fi

source "$script_dir/skills/init.sh"
install_catlass_cpp_skills "$config_root"
install_link "$script_dir/knowledge" "$bundle_link"

mkdir -p "$(dirname "$config_target")"
if [ -f "$config_target" ] && ! head -20 "$config_target" | grep -qF "$MARKER"; then
    cp -a "$config_target" "$config_target.bak.$(date +%Y%m%d_%H%M%S)"
fi
cp "$script_dir/AGENTS.md" "$config_target"

skills_json="$(printf '%s\n' "${SKILLS[@]}" | python3 -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')"
cat > "$manifest" <<EOF
{
  "brand": "CANNBot",
  "plugin": "$PLUGIN",
  "version": "$VERSION",
  "level": "$level",
  "tool": "$tool",
  "installed_skills": $skills_json,
  "knowledge": "$bundle_link",
  "config_file": "$config_target"
}
EOF

health_check
