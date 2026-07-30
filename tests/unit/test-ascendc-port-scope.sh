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
# Test: AscendC port orchestrator scope contract
#
# The plugin exposes exactly two entry capabilities and must not contain
# implementation, documentation, routing, or path residue from another
# accelerator family.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PLUGIN_DIR="$REPO_ROOT/plugins-community/ascendc-port-orchestrator"

forbidden_re='\b(c[u]da|nvid[i]a|a[1]00|h[1]00|nv[c]c|p[t]x|p[t]xas|c[u]bin|c[u]tl[a]ss|t[e]nsor[r]t|n[c]cl|f[a]tbin|s[a]ss|n[s]ight|c[u]dnn|c[u]blas|g[p]u)\b|[.,{](c[u]|c[u]h)([^[:alnum:]_]|$)|g[p]u_'

if ! printf 'kernel/*.{h,cpp,c%s}\n' 'u' | rg -q -i "$forbidden_re"; then
    echo "[FAIL] Forbidden-extension detector misses brace-list entries"
    exit 1
fi

for sample in \
    "$(printf 'c%stl%sss' 'u' 'a')" \
    "$(printf 't%snsor%st' 'e' 'r')" \
    "$(printf 'n%scl' 'c')" \
    "$(printf 'p%sxas' 't')" \
    "$(printf 'f%stbin' 'a')" \
    "$(printf 's%sss' 'a')"; do
    if ! printf '%s\n' "$sample" | rg -q -i "$forbidden_re"; then
        echo "[FAIL] Forbidden-family detector misses a compiled-stack term"
        exit 1
    fi
done

if rg -n -i --hidden "$forbidden_re" "$PLUGIN_DIR"; then
    echo "[FAIL] Found forbidden accelerator-family content in the plugin"
    exit 1
fi

if find "$PLUGIN_DIR" -print | rg -n -i "$forbidden_re"; then
    echo "[FAIL] Found a forbidden accelerator-family path in the plugin"
    exit 1
fi

legacy_mode_re='port_tilelang|port_triton|opgen_mc2|port_mc2|port_pytorch|aog-benchmark-sync|benchmark[-_ ]mode|mode:[[:space:]]*benchmark|await_tilelang|resolve_tilelang|force_tilelang|\b(triton|tilelang|mc2|workbench)\b|MC²'
if rg -n -i --hidden "$legacy_mode_re" "$PLUGIN_DIR"; then
    echo "[FAIL] Found an RFC-external product mode or specialized route"
    exit 1
fi

python3 - "$PLUGIN_DIR" <<'PY'
import json
import pathlib
import re
import sys

plugin_dir = pathlib.Path(sys.argv[1])
expected = ["ascendc-cross-gen-port", "ascendc-backward-gen"]

agents_text = (plugin_dir / "AGENTS.md").read_text(encoding="utf-8")
frontmatter = agents_text.split("---", 2)[1]
match = re.search(r"(?ms)^skills:\n((?:  - [^\n]+\n)+)", frontmatter)
actual = re.findall(r"^  - (.+)$", match.group(1), re.MULTILINE) if match else []
if sorted(actual) != sorted(expected):
    raise SystemExit(f"[FAIL] Entry skills must be exactly {expected}, got {actual}")

for skill in expected:
    if not (plugin_dir / "skills" / skill / "SKILL.md").is_file():
        raise SystemExit(f"[FAIL] Missing entry skill: {skill}")

manifest = json.loads(
    (plugin_dir / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
)
description = manifest.get("description", "")
if "arch22→arch35" not in description:
    raise SystemExit("[FAIL] Plugin description must preserve arch22→arch35 wording")
if "A3→A5" in description or "A3-to-A5" in description:
    raise SystemExit("[FAIL] Plugin description must not rebrand the migration scope")

agents = {pathlib.Path(item).stem for item in manifest.get("agents", [])}
if "aog-cann-learner" not in agents:
    raise SystemExit("[FAIL] CANN learner must remain registered for the feedback loop")

init_text = (plugin_dir / "init.sh").read_text(encoding="utf-8")
for required in ("aog-cann-learner", "aog-prior-art-verify"):
    if required not in init_text:
        raise SystemExit(f"[FAIL] Installer must retain {required}")

prior_art = plugin_dir / "skills" / "aog-prior-art-verify"
if not (prior_art / "SKILL.md").is_file() or not (prior_art / "scripts" / "stage_candidate.py").is_file():
    raise SystemExit("[FAIL] Provenance-bound prior-art staging skill is incomplete")

prior_art_text = (prior_art / "SKILL.md").read_text(encoding="utf-8")
for required in (
    "fresh arch22 source-NPU capture is always required",
    "Candidate evidence remains in `verification_prior_art.json`",
    "Only the standard O4/O5 path may establish customer success",
):
    if required not in prior_art_text:
        raise SystemExit(f"[FAIL] Prior-art safety invariant missing: {required}")

classify_text = (prior_art / "scripts" / "classify.py").read_text(encoding="utf-8")
for forbidden in ("PRIOR_ART_PASS_ARCHIVED", "SKIP_UPSTREAM", "finalize_skip"):
    if forbidden in classify_text:
        raise SystemExit(f"[FAIL] Prior-art classifier may short-circuit final gates: {forbidden}")
if "CANDIDATE_PASS" not in classify_text or "continue mandatory fresh" not in classify_text:
    raise SystemExit("[FAIL] Passing prior art must remain advisory")

fsm_text = (plugin_dir / "workflows" / "opgen_state_machine.yaml").read_text(encoding="utf-8")
required_fsm_text = (
    "Optionally prestage provenance-recorded target, sibling or archive candidates",
    "migration truth is a fresh source-arch NPU capture",
    "FA/L4 uses the registered standard AscendC worker",
)
for required in required_fsm_text:
    if required not in fsm_text:
        raise SystemExit(f"[FAIL] FSM lost required RFC behavior: {required}")

public_files = [
    plugin_dir / ".claude-plugin" / "plugin.json",
    plugin_dir / "AGENTS.md",
    plugin_dir / "README.md",
    plugin_dir / "quickstart.md",
    plugin_dir / "docs" / "ARCHITECTURE.md",
    plugin_dir / "docs" / "USAGE.md",
    plugin_dir / "skills" / "ascendc-cross-gen-port" / "SKILL.md",
    plugin_dir / "skills" / "aog-a3-author" / "SKILL.md",
    plugin_dir / "workflows" / "opgen_state_machine.yaml",
]
rebranded = re.compile(r"A3\s*(?:→|->|to)\s*A5", re.IGNORECASE)
for path in public_files:
    if rebranded.search(path.read_text(encoding="utf-8")):
        raise SystemExit(f"[FAIL] Public migration wording was rebranded in {path}")

a3_author_text = (
    plugin_dir / "skills" / "aog-a3-author" / "SKILL.md"
).read_text(encoding="utf-8")
for required in (
    "one source-supported representative rank",
    "MAX_CASE_TENSOR_BYTES",
    "`ascendcl`, `nnopbase`, and `opapi`",
    "must not define generic macros such as `CHECK_RET`",
):
    if required not in a3_author_text:
        raise SystemExit(f"[FAIL] A3 author safety contract missing: {required}")
if "ascend_hal" in a3_author_text:
    raise SystemExit("[FAIL] A3 author must not prescribe the devlib-only ascend_hal link")

worker_text = (plugin_dir / "agents" / "aog-kernel-worker.md").read_text(encoding="utf-8")
for required in (
    "source and destination resolve to the same dtype",
    "if constexpr (!std::is_same_v<T, float>)",
    "fold that coefficient into the AscendC derivative kernel",
    "Pybind remains output allocation + stream handoff + kernel launch only",
):
    if required not in worker_text:
        raise SystemExit(f"[FAIL] Kernel worker safety contract missing: {required}")

worker_hook = (plugin_dir / "hooks" / "v3" / "check_worker.sh").read_text(encoding="utf-8")
for required in (
    r"\.cpu[[:space:]]*\(",
    r"\.to[[:space:]]*\([[:space:]]*at::kCPU",
    "torch::ones_like",
    "at::ones_like",
):
    if required not in worker_hook:
        raise SystemExit(f"[FAIL] Worker hook pybind blacklist missing: {required}")

print("[PASS] AscendC port orchestrator scope contract")
PY
