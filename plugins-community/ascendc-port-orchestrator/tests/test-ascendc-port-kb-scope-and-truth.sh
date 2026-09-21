#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PLUGIN_REL="plugins-community/ascendc-port-orchestrator"
PLUGIN_ROOT="$REPO_ROOT/$PLUGIN_REL"
KB_ROOT="$PLUGIN_ROOT/kb"
GATE_CONTRACT="$KB_ROOT/shared/GATE_CONTRACT.md"
FA_TEMPLATES="$PLUGIN_ROOT/templates/fa_class"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_contains() {
    local file="$1"
    local text="$2"
    grep -Fq -- "$text" "$file" || fail "$file is missing: $text"
}

assert_tracked_plugin_has_no_match() {
    local pattern="$1"
    local label="$2"
    local matches
    # 豁免（自引用守卫与特性文件）：插件自己的测试为证明"绝不走外部加速器路径"而
    # 命名这些 family；model_reference 特性按设备名同步用户提供的 torch 模型。
    if matches=$(git -C "$REPO_ROOT" grep -n -i -E "$pattern" -- "$PLUGIN_REL" 2>/dev/null \
            | grep -vE "^$PLUGIN_REL/([^:]*/)*tests/" \
            | grep -vE "^$PLUGIN_REL/([^:]*/)*test_[^/]*" \
            | grep -vE "^$PLUGIN_REL/([^:]*/)*[^/]*model_reference[^/]*"); then
        echo "$matches" >&2
        fail "$label"
    fi
}

[ -d "$PLUGIN_ROOT" ] || fail "plugin directory is missing"

# Keep the plugin scoped to AscendC migration/backward generation. These backends and workbenches
# are unrelated even though AscendC target prior-art is intentionally retained below.
# `tilelang` is boundary-scoped for the same reason tests/test-ascendc-port-scope.sh scopes it:
# the compound route name `tilelang2ascendc` names an AscendC source *format* the plugin consumes
# (an already-lowered `model_new_ascendc.py` + `kernel/` custom-op project), not a foreign backend
# the plugin can run. A bare `tilelang` token stays forbidden.
assert_tracked_plugin_has_no_match \
    'cuda|nvidia|ptx|triton|\btilelang\b|mc2' \
    "unrelated backend knowledge is present"
assert_tracked_plugin_has_no_match \
    'port_tilelang|port_triton|opgen_mc2|port_mc2|port_pytorch|benchmark[-_ ]?mode|mode=benchmark' \
    "removed independent product mode is present"

# Target archives, prior-art and pre-staged branches are admissible research/generation inputs.
for file in \
    "$FA_TEMPLATES/op_host/GE_HOST_TEMPLATE.md" \
    "$FA_TEMPLATES/op_host/flash_attention_score_tiling.cpp" \
    "$FA_TEMPLATES/op_kernel/kernel_common.h" \
    "$FA_TEMPLATES/op_kernel/wholeport/wp_fa_entry.h"; do
    [ -f "$file" ] || fail "required FA prior-art template is missing: $file"
done

# The truth-boundary wording lives in kb/shared/GATE_CONTRACT.md (still shipped);
# the OL-284..292 generic-fact cards, the npu-ut-* safety-net cards, OL-141 and the
# porter toolchain/playbook cards moved to the external cannbot-knowledge repo
# (commit 223ee980) and are no longer checkable here.
assert_contains "$GATE_CONTRACT" 'selected arch22 source 的 source-arch NPU capture'
assert_contains "$GATE_CONTRACT" '`FAIL_NO_INDEPENDENT_TRUTH`'
assert_tracked_plugin_has_no_match \
    'CANN-bit-match|Reference is .*target NPU|target NPU.*Reference is' \
    "target implementation is still advertised as final truth"

# The plugin carries no non-CANN payload, so it ships no licence or provenance files of its own and simply
# inherits the repository-root CANN-2.0 LICENSE. That is the repo standard: 22 of the 24 plugins ship none,
# including the closest comparison -- tilelang2ascendc-ops-generator, which ships derived FlashAttention
# op_host/op_kernel templates carrying only a plain CANN-2.0 header. autoresearch is the sole exception, and
# only because it genuinely vendors Apache-2.0 MindSpore AKG code.
for stray in LICENSE NOTICE README.OpenSource SOURCE_REVISION LICENSE-APACHE; do
    if [ -e "$PLUGIN_ROOT/$stray" ]; then
        fail "plugin re-added $stray; it has no non-CANN payload and must inherit the root LICENSE"
    fi
done

# The target-derived FA host templates stay CANN-2.0 and must keep their upstream copyright notice
# (CANN-2.0 section 3.2 forbids removing it).
fa_host="$FA_TEMPLATES/op_host"
for tmpl in flash_attention_score_def flash_attention_score_infershape flash_attention_score_tiling; do
    assert_contains "$fa_host/$tmpl.cpp" 'Copyright (c) 2025 Huawei Technologies Co., Ltd.'
    assert_contains "$fa_host/$tmpl.cpp" 'CANN Open Software License Agreement Version 2.0'
done

# No non-CANN third-party payload may reappear: the unreferenced Apache-2.0 hardware snapshot was removed,
# and with it the need for a second license. Note that the `AscendOpGenAgent` token itself stays legal here
# -- it is also the workspace directory name used by the deploy/build scripts -- so this guard targets
# license declarations, not the path.
if [ -e "$KB_ROOT/workbench_imports" ]; then
    fail "the removed Apache-2.0 workbench snapshot reappeared"
fi
assert_tracked_plugin_has_no_match \
    'Apache-2\.0|Apache License|LICENSE-APACHE' \
    "a non-CANN third-party license declaration is present"

echo "PASS: AscendC port scope, retained templates, and truth boundaries are consistent"
