---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "op_host/ archive incomplete — patches shipped instead of complete files [V351+all-modes, finalize-pipeline]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=all; phase=finalize"
phenomenon: build_failure
signal:
  - "Downstream reviewers / CANN team / cann/ops-nn PR审核 cannot apply our archived ops because op_host/ either:"
confidence: single_run
original_id: PB-33
timestamp_inferred: true
tags: [ascendc, pb-33]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=all; phase=finalize`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: User feedback 2026-05-14T02:55Z (Discord) — "op_host 代码似乎有一些缺失"`

**Symptom**: Downstream reviewers / CANN team / `cann/ops-nn` PR审核 cannot apply our archived ops because `op_host/` either:
1. **Missing entirely** — no `op_host/` directory in the archive (5 archived ops at scan time: rms_norm_quant, group_norm_silu_quant), OR
2. **Only has `<op>_def.cpp.patch`** + `config/` — patches without the master they were generated against (ctc_loss_v3, gather_elements_v2, apply_adam_w_quant)

PR4778 packaging REQUIRES a complete task-owned `op_host/` layout: `<op>_def.cpp` +
`<op>_tiling.cpp` + `<op>_tiling.h` + `CMakeLists.txt` +
`config/ascend950/{*_binary.json,_simplified_key.ini}` (plus optional `op_api/<op>.{cpp,h}` +
`<op>_infershape.cpp`). Patches are review aids only. Target layout may guide directory completeness,
but copied target bodies do not satisfy generation or validation.

**Root cause**: kw_brief Phase B.4 (pre-2026-05-14) only required `workspace/{op}/op_host/<op>_def.cpp.patch`. Worker complied — wrote only the patch. `_tiling.cpp/.h`, `CMakeLists.txt`, `op_api/` were never required to be produced. For ops without ANY worker-written op_host/ (rms_norm_quant), even the patch wasn't required because workspace was Path B cpp-binary style which skipped `op_host/` setup entirely.

**Fix (2026-05-14, commit pending)**:

1. **kw_brief Phase B.4 rewritten** to mandate complete files (not patches). Patches still produced as review-aid trail, but the SHIP artifact is the complete `<op>_def.cpp` + `_tiling.{cpp,h}` + `CMakeLists.txt` + optional `op_api/`.

2. **finalize_pipeline `_check_op_host_completeness` gate added** (`GateID.OP_HOST_COMPLETENESS`). Counts non-config / non-patch files in `workspace/op_host/`; < 3 → ROLLBACK to `await_worker`. Apply it when the selected arch22→arch35 packaging contract requires `op_host/`; standalone backward pybind builds use their explicit carve-out.

3. **`briefs/_common.py:fixed_layout_block()` added** + injected into ALL 6 agent briefs (kw / ko / fo / pp / ar / da). Every agent now sees the same PR4778 contract; no agent can produce or modify an op without knowing the required output layout.

**Detection signature** (post-finalize audit):

```bash
# Count non-patch, non-config files in archive/op_host/
n=$(find <archive>/op_host -type f \
    -not -path '*/config/*' -not -name '*.patch' | wc -l)
[ $n -lt 3 ] && echo "PB-33 violation: only $n files"
```

**Evidence**:
- 5 archived ops at scan time (2026-05-14) violated PR4778:
  - ctc_loss_v3 / gather_elements_v2 / apply_adam_w_quant: only `.patch` + `config/`
  - rms_norm_quant / group_norm_silu_quant: no `op_host/` at all
- User feedback (Discord 02:55Z 2026-05-14): "op_host 的代码似乎有一些缺失，有的算子给了 ctc_loss_v3_def.cpp.patch，有的算子产物没有 op_host"
- expand_into_jagged_permute 2026-05-17 (port_a3_to_a5, kw-1): first port_a3 op produced post-fix. Workspace ships 6 complete `op_host/` files (`<op>_def.cpp` + `<op>_tiling.{cpp,h}` + `CMakeLists.txt` + 2 `config/ascend950/` files) + 2 `op_kernel/` files (`arch35/<op>.h` + `<op>_apt.cpp`). No `.patch`-only artifact. Reviewer can diff against current upstream without needing the original master snapshot. Confirms kw_brief Phase B.4 rewrite + `_check_op_host_completeness` gate enforce the contract.
- fatrelu_mul 2026-05-17 (port_a3_to_a5, kw-1): mirrored `op_host/` + `op_kernel/arch35/` + `<op>_apt.cpp` files for PR4778 layout-contract completeness BUT NOT BUILT — actual ship + verify path runs the standalone pybind kernel at `workspace/fatrelu_mul/kernel/` via ACLRT_LAUNCH_KERNEL (per P140 pivot). The op_host/arch35 artifacts serve as upstream-ready review aids: a CANN-team reviewer can apply these to ops-nn directly with no algorithmic changes (arch35 `.h` would need V220 implementation body copied verbatim — only macro guards and includes changed in the L1 mechanical edit). Demonstrates the layout contract can be satisfied as review trail even when the active build path is pybind/ACLRT_LAUNCH_KERNEL rather than the full ops-nn pipeline.

**Mitigation gate**: `finalize_pipeline.check_finalize_eligibility` returns `GateID.OP_HOST_COMPLETENESS` on `op_host/` insufficient. Tests at `test_finalize_gate_contract.py::test_op_host_missing_dir` + `::test_op_host_only_patch_no_complete_files`.

**Other instances (predicted)**: every future op-gen run. Without this gate, regress to the shipping-incomplete pattern is silent (precision/perf gates don't catch missing artifacts).

**Historical backfill note (superseded)**: the old proposal copied target tiling files into five
incomplete archives. Current policy requires regenerating complete task-owned files from the selected
source contract and rerunning build/provenance/truth gates.

**Shared-common layout guidance (adaptive_avg_pool3d witness)**: when target prior art uses a shared
`<family>_common/op_kernel/arch35/` layout, use it to infer package dependencies, not to copy bodies.
A strict local-only layout would otherwise tempt one of:
- (a) copying shared kernel files into the op's local `op_kernel/arch35/`, OR
- (b) rewriting `<op>_apt.cpp`'s `#include "../<family>_common/arch35/..."` to `arch35/...`

Both break on-host build deployment because upstream's tree expects the shared-common path; siblings (`adaptive_max_pool3d` etc.) co-rely on that same dir. **Permitted layout for Mode B ports**: workspace MAY include a sibling `<family>_common/` directory mirroring upstream's structure, when ALL of:

1. The op's `op_kernel/arch35/` is empty (no `<op>_*.{h,cpp}` files at that path)
2. Upstream has a sibling `<family>_common/op_kernel/arch35/` containing `<op>_*` files (Mode B per OL-141)
3. The op's `<op>_apt.cpp` references the shared path via `#include "../<family>_common/arch35/..."`
4. Only the target op's arch35 dependencies are shipped from the sibling dir (other family-member ops MUST be excluded from the archive — one op per archive, even when they share kernel sources)

In this case PB-33's completeness intent is satisfied by task-owned `op_kernel/` + `op_host/` and the
declared shared dependency surface. The `_check_op_host_completeness` gate still counts `op_host/`;
a kernel-completeness gate must understand shared dependencies without accepting a target mirror.

Detection signature for Mode B archive:
```bash
# Mode B if op_kernel/arch35/ is empty AND a sibling *_common/op_kernel/arch35 has <op>_*
op_arch35=$(ls <archive>/op_kernel/arch35/*.{h,cpp} 2>/dev/null | wc -l)
sibling_arch35=$(ls <archive>/../*_common/op_kernel/arch35/<op>_*.{h,cpp} 2>/dev/null | wc -l)
[ "$op_arch35" -eq 0 ] && [ "$sibling_arch35" -gt 0 ] && echo "Mode B archive"
```

**Cross-reference**:
- PR4778 spec (CANN ops-nn) — the canonical layout being mirrored
- `kw_brief.py §Phase B.4` (rev 2026-05-14) — the producer-side rule
- `_common.py:fixed_layout_block()` — the shared contract block
- `finalize_pipeline._check_op_host_completeness` — the enforcement

<!-- 迁移自 porter kb/target/ascendc/（PB-33，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
