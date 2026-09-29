#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <operator_name>" >&2
    exit 2
fi

operator_name="$1"
case "$operator_name" in
    *[!a-z0-9_]*|""|[0-9]*)
        echo "operator_name must be safe snake_case" >&2
        exit 2
        ;;
esac
case "$operator_name" in
    *catlass*) ;;
    *)
        echo "operator_name must contain catlass" >&2
        exit 2
        ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
plugin_root="$(cd "$script_dir/../../.." && pwd)"
legacy_init="$plugin_root/../workflows/scripts/init_operator_project.sh"
operator_dir="$PWD/operators/$operator_name"
workflow_file="$operator_dir/docs/workflow.json"

if [ ! -d "$operator_dir" ]; then
    if [ ! -f "$legacy_init" ]; then
        echo "legacy project initializer not found: $legacy_init" >&2
        exit 1
    fi
    bash "$legacy_init" "$operator_name"
elif [ -f "$workflow_file" ]; then
    python3 "$script_dir/validate_workflow.py" --workflow "$workflow_file"
elif find "$operator_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "refusing to convert an existing legacy project without a dedicated workflow marker" >&2
    exit 1
else
    mkdir -p "$operator_dir/docs" "$operator_dir/reference" "$operator_dir/op_host" "$operator_dir/op_kernel" "$operator_dir/build" "$operator_dir/test" "$operator_dir/scripts"
    printf '# %s\n' "$operator_name" > "$operator_dir/README.md"
fi

mkdir -p "$operator_dir/docs" "$operator_dir/reference"
copy_missing() {
    local source="$1"
    local target="$2"
    if [ ! -e "$target" ]; then
        cp "$source" "$target"
    fi
}

copy_missing "$plugin_root/skills/catlass-cpp-interface/templates/workflow.json" "$operator_dir/docs/workflow.json"
copy_missing "$plugin_root/skills/catlass-cpp-reference/templates/definition.template.json" "$operator_dir/reference/definition.template.json"
copy_missing "$plugin_root/skills/catlass-cpp-reference/templates/precision-policy.json" "$operator_dir/reference/precision-policy.json"

python3 "$script_dir/validate_workflow.py" --workflow "$operator_dir/docs/workflow.json"
printf 'Initialized %s with workflow catlass-linear-attention-v1\n' "$operator_dir"
