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
OKF_ROOT="$KB_ROOT/okf/runbooks"
OKF_REFERENCE="$KB_ROOT/okf/reference"
# OKF-only migration (2026-08-31): legacy kb/target/ + kb/KB_INDEX.md removed.
# OL/CAND entries live as OKF cards under kb/okf/runbooks/**;
# FA prior-art templates → templates/fa_class/.
#
# reference/ reorganisation (2026-09-05): the flat top-level docs and migration/**
# moved under bundles — porter/ for this plugin's own methodology,
# asc-devkit-vendored/ for the upstream API copy. Paths pinned here so the guard
# fails loudly if they move again rather than silently passing on a `grep` miss.
NPU_VALIDATION="$OKF_REFERENCE/porter/toolchain/npu_ut_validation.md"
SOURCE_SKILL="$OKF_REFERENCE/porter/playbook/source_skill.md"
MIGRATION_README="$OKF_REFERENCE/porter/playbook/overview.md"
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

for id in $(seq 3 14); do
    card="$(find "$OKF_ROOT" -type f -name "cand-fa-cv-$id-*.md" -print -quit)"
    [ -n "$card" ] || fail "CAND-FA-CV-$id OKF card is missing"
    assert_contains "$card" "original_id: CAND-FA-CV-$id"
done

# OL-284..292 are the nine RFC-relevant generic facts retained across the restoration.
ol_card() { find "$OKF_ROOT" -type f -name "ol-$1-*.md" -print -quit; }
for id in $(seq 284 292); do
    card="$(ol_card "$id")"
    [ -n "$card" ] || fail "OL-$id OKF card is missing"
    assert_contains "$card" "original_id: OL-$id"
done

assert_contains "$(ol_card 284)" 'Source CANN set_env.sh outside shell strict mode'
assert_contains "$(ol_card 285)" 'Recompute contiguous strides after leading-one rank padding'
assert_contains "$(ol_card 286)" 'Precompute permuted source strides on host'
assert_contains "$(ol_card 287)" 'Form a safe GM address before any potentially out-of-bounds load'
assert_contains "$(ol_card 288)" 'Deterministic variable-size output uses count, exclusive offsets, then ordered emit'
assert_contains "$(ol_card 289)" 'Verify nested outputs structurally and require every leaf to pass'
assert_contains "$(ol_card 290)" 'Short-circuit exact non-finite equality before tolerance arithmetic'
assert_contains "$(ol_card 291)" 'Duplicate-index CPU scatter may be schedule-dependent'
assert_contains "$(ol_card 292)" 'Probe option-sensitive CPU and NPU reference semantics'

# Prior-art may guide implementation; final truth remains source-NPU for migration and CPU fp64 for backward.
assert_contains "$NPU_VALIDATION" 'Target archives, prior-art implementations,'
assert_contains "$NPU_VALIDATION" 'must not be copied verbatim into a deliverable and declared successful'
assert_contains "$NPU_VALIDATION" 'Migration ordinary truth:'
assert_contains "$NPU_VALIDATION" 'source-arch NPU.'
assert_contains "$NPU_VALIDATION" 'Backward truth:'
assert_contains "$NPU_VALIDATION" 'CPU fp64 autograd oracle.'
assert_contains "$NPU_VALIDATION" 'must never be promoted to migration or backward truth.'
assert_contains "$GATE_CONTRACT" 'selected arch22 source 的 source-arch NPU capture'
assert_contains "$GATE_CONTRACT" '`FAIL_NO_INDEPENDENT_TRUTH`'
assert_tracked_plugin_has_no_match \
    'CANN-bit-match|Reference is .*target NPU|target NPU.*Reference is' \
    "target implementation is still advertised as final truth"

npu_cards=(
    npu-ut-a3-recipes-are-template-on-a5.md
    npu-ut-aclinit-500000-not-device-lock.md
    npu-ut-attr-toggle-oracle.md
    npu-ut-cann-overlay-gotchas.md
    npu-ut-device-exception-oob-read.md
    npu-ut-dispatch-confirmation-probe.md
    npu-ut-gm-canary-oob-write.md
    npu-ut-installed-vs-scanned.md
    npu-ut-source-rebuild-overlay-override-proof.md
    npu-ut-value-diff-cpu-reference.md
    npu-ut-verification-decision-order.md
)
for card in "${npu_cards[@]}"; do
    [ -f "$OKF_ROOT/operator-optimization/$card" ] || fail "NPU safety-net card is missing: $card"
done
assert_contains "$OKF_ROOT/operator-optimization/npu-ut-installed-vs-scanned.md" \
    'may be consulted as prior-art'
assert_contains "$OKF_ROOT/operator-optimization/npu-ut-source-rebuild-overlay-override-proof.md" \
    'final validation gate must load the task-owned clean build'
assert_contains "$OKF_ROOT/operator-optimization/npu-ut-value-diff-cpu-reference.md" \
    'Migration truth is the declared contract plus current selected-arch22 source NPU capture.'
assert_contains "$OKF_ROOT/operator-optimization/npu-ut-value-diff-cpu-reference.md" \
    'Backward truth is CPU fp64 autograd'

# Public wording stays arch22 -> arch35; A3/A5 remains only as compatible internal route/hardware aliases.
assert_contains "$SOURCE_SKILL" 'name: ascendc-operator-arch22-arch35-migration'
assert_contains "$SOURCE_SKILL" '# AscendC arch22 → arch35 算子迁移与反向算子生成'
assert_contains "$SOURCE_SKILL" '它们是'
assert_contains "$SOURCE_SKILL" 'advisory 输入，不能逐字复制目标实现后直接宣告生成成功'
assert_contains "$MIGRATION_README" '# arch22 → arch35 AscendC Migration and Backward-Generation Reference'
if grep -Eq 'A3[[:space:]]*(→|->|to)[[:space:]]*A5' "$SOURCE_SKILL"; then
    fail "public migration description was renamed away from arch22 -> arch35"
fi

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

# Active retrieval guidance (the OL-141 OKF card; the legacy KB_INDEX.md row and
# OPERATIONAL_KNOWLEDGE.md entry were removed in the OKF-only migration) must preserve
# target templates as advisory prior art without allowing a target-body mirror, target
# output, or upstream-presence skip to become generation success.
OL141="$(find "$OKF_ROOT" -type f -name 'ol-141-*.md' -print -quit)"
[ -n "$OL141" ] || fail "OL-141 OKF card is missing"
assert_contains "$OL141" 'advisory prior art'
assert_contains "$OL141" 'never authorizes a verbatim target-body mirror'
assert_contains "$OL141" 'It is a diagnostic and review input, not the migration oracle'
if grep -rFq 'Upstream `arch35/` present → SKIP harness regen' "$KB_ROOT/okf"; then
    fail "KB still advertises target-presence skip"
fi
if grep -rFq '**Ship artifact** (`op_host/` + `op_kernel/`): COPY upstream verbatim' "$KB_ROOT/okf"; then
    fail "OL-164 still advertises a verbatim target ship artifact"
fi
FA_CLASS_TEMPLATE="$OKF_REFERENCE/porter/patterns/fa_class_template.md"
[ -f "$FA_CLASS_TEMPLATE" ] || fail "$FA_CLASS_TEMPLATE is missing (a bare grep on a moved file passes silently)"
if grep -Fq 'copy IS the template mechanism' "$FA_CLASS_TEMPLATE"; then
    fail "FA template still advertises target-body copying as generation"
fi

# OKF-only migration (2026-08-31): the legacy bundled KB layout (kb/target/ +
# kb/KB_INDEX.md) is physically deleted; the bundled b-tier is OKF cards under
# kb/okf/** only. Pin the layout so a stray restore or an emptied OKF tree
# fails this gate. (kb/target may legitimately be named in migration-provenance
# comments inside kb/okf/** and engine sources — this guards the LAYOUT, not
# the string.)
if [ -e "$KB_ROOT/target" ]; then
    fail "Legacy kb/target tree reappeared (OKF-only migration removed it)"
fi
if [ -e "$KB_ROOT/KB_INDEX.md" ]; then
    fail "Legacy kb/KB_INDEX.md reappeared (OKF-only migration removed it)"
fi
for okf_dir in "$OKF_ROOT" "$OKF_REFERENCE"; do
    [ -d "$okf_dir" ] || fail "OKF knowledge directory is missing: $okf_dir"
    find "$okf_dir" -type f -name '*.md' -print -quit | grep -q . \
        || fail "OKF knowledge directory has no cards: $okf_dir"
done

echo "PASS: AscendC port KB scope, prior-art, generic facts, and truth boundaries are consistent"
