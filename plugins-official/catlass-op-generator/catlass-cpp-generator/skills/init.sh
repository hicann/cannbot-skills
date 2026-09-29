#!/bin/bash
set -euo pipefail

_catlass_cpp_prepare_skill_target() {
    local source="$1"
    local target="$2"
    [ -f "$source/SKILL.md" ] || { echo "missing skill source: $source" >&2; exit 1; }
    if [ -e "$target" ] || [ -L "$target" ]; then
        if same_target "$target" "$source"; then
            return
        fi
        echo "refusing to replace foreign path: $target" >&2
        exit 1
    fi
}

install_catlass_cpp_skills() {
    local config_root="$1"
    local SCRIPT_DIR
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    mkdir -p "$config_root/skills"

    _catlass_cpp_prepare_skill_target "$SCRIPT_DIR/catlass-cpp-interface" "$config_root/skills/catlass-cpp-interface"
    ln -sfn "$(realpath "$SCRIPT_DIR/catlass-cpp-interface")" "$config_root/skills/catlass-cpp-interface"
    _catlass_cpp_prepare_skill_target "$SCRIPT_DIR/catlass-cpp-reference" "$config_root/skills/catlass-cpp-reference"
    ln -sfn "$(realpath "$SCRIPT_DIR/catlass-cpp-reference")" "$config_root/skills/catlass-cpp-reference"
    _catlass_cpp_prepare_skill_target "$SCRIPT_DIR/catlass-cpp-design" "$config_root/skills/catlass-cpp-design"
    ln -sfn "$(realpath "$SCRIPT_DIR/catlass-cpp-design")" "$config_root/skills/catlass-cpp-design"
    _catlass_cpp_prepare_skill_target "$SCRIPT_DIR/catlass-cpp-develop" "$config_root/skills/catlass-cpp-develop"
    ln -sfn "$(realpath "$SCRIPT_DIR/catlass-cpp-develop")" "$config_root/skills/catlass-cpp-develop"
    _catlass_cpp_prepare_skill_target "$SCRIPT_DIR/catlass-cpp-test" "$config_root/skills/catlass-cpp-test"
    ln -sfn "$(realpath "$SCRIPT_DIR/catlass-cpp-test")" "$config_root/skills/catlass-cpp-test"
    _catlass_cpp_prepare_skill_target "$SCRIPT_DIR/catlass-cpp-knowledge" "$config_root/skills/catlass-cpp-knowledge"
    ln -sfn "$(realpath "$SCRIPT_DIR/catlass-cpp-knowledge")" "$config_root/skills/catlass-cpp-knowledge"
}
