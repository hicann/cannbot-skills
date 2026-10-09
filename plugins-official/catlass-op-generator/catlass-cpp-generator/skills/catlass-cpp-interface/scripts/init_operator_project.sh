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

set -euo pipefail

if { [ "$#" -ne 1 ] && [ "$#" -ne 3 ]; } || { [ "$#" -eq 3 ] && [ "$2" != "--algorithm-family" ]; }; then
    echo "Usage: $0 <operator_name> [--algorithm-family linear_attention|block_sparse_attention|sparse_flash_mla]" >&2
    exit 2
fi

operator_name="$1"
algorithm_family="${3:-linear_attention}"
case "$algorithm_family" in
    linear_attention|block_sparse_attention|sparse_flash_mla) ;;
    *) echo "unsupported algorithm family: $algorithm_family" >&2; exit 2 ;;
esac
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
workflow_template="$plugin_root/skills/catlass-cpp-interface/templates/workflow.json"
if [ "$algorithm_family" = sparse_flash_mla ]; then
    workflow_template="$plugin_root/knowledge/operator/sparse-flash-mla/workflow.json"
fi

if [ ! -d "$operator_dir" ]; then
    if [ ! -f "$legacy_init" ]; then
        echo "legacy project initializer not found: $legacy_init" >&2
        exit 1
    fi
    bash "$legacy_init" "$operator_name"
elif [ -f "$workflow_file" ]; then
    python3 "$script_dir/validate_workflow.py" --workflow "$workflow_file"
    algorithm_family="$(python3 - "$workflow_file" "${3:-}" <<'PY'
import json
import sys
from pathlib import Path

family = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["algorithm_family"]
if sys.argv[2] and sys.argv[2] != family:
    raise SystemExit("requested algorithm family does not match existing workflow marker")
print(family)
PY
)"
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

if [ ! -e "$workflow_file" ]; then
    python3 - "$script_dir/validate_workflow.py" "$workflow_template" "$workflow_file" "$algorithm_family" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("workflow_validator", sys.argv[1])
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
state = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
state["algorithm_family"] = sys.argv[4]
state["workflow_id"] = validator.WORKFLOWS[sys.argv[4]]
errors = validator.validate_workflow(state)
if errors:
    raise SystemExit("; ".join(errors))
Path(sys.argv[3]).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
PY
fi
copy_missing "$plugin_root/skills/catlass-cpp-reference/templates/definition.template.json" "$operator_dir/reference/definition.template.json"
copy_missing "$plugin_root/skills/catlass-cpp-reference/templates/precision-policy.json" "$operator_dir/reference/precision-policy.json"

python3 "$script_dir/validate_workflow.py" --workflow "$operator_dir/docs/workflow.json"
printf 'Initialized %s with %s workflow\n' "$operator_dir" "$algorithm_family"
