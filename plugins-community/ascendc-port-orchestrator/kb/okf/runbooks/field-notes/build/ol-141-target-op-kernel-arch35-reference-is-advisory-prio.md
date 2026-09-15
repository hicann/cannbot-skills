---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Target `op_kernel/arch35/` reference is advisory prior art — never skip task-owned generation or independent truth [arch22→arch35]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2.5"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2.5"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-141
timestamp_inferred: true
tags: [skip_upstream_has_reference, ascendc, ol-141]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2.5`
`verified_on: soc=Ascend950PR; cann=9.0.0`

**Current rule**: during arch22→arch35 migration, Phase O2.5 inventories target-side `arch35/`,
shared-common, and unified-cpp variants as provenance-bearing prior art. Their presence never yields
`SKIP_UPSTREAM_HAS_REFERENCE`, never authorizes a verbatim target-body mirror, and never replaces the
task-owned build. Generate from the selected arch22 source and declared contract, then validate the
current binary against a capture from that same source on the source-arch NPU.

Only a pre-staged arch35 branch that is part of the selected arch22 source tree and traceable to that
selected-source contract may be reused as selected-source code; it still must build and pass the same
independent truth gate.

**Why inventory it**: target prior art can expose public-API shapes, required variants, packaging
layout, and risk hypotheses. It is a diagnostic and review input, not the migration oracle. A copied
target implementation can match the target by construction while proving nothing about translation
from the selected source.

**Concrete example (ada_layer_norm, 2026-05-13)**: upstream `op_kernel/arch35/` has
`{welford.h, full_load.h, impl.h, common.h}` (1463 lines, Welford one-pass + full-load
fast path, tilingkey dispatcher). Our brief pointed kw at `op_kernel/<op>.h` (A3
implementation) without first checking arch35/. kw wrote 923-line hand-rolled kernel
using `AscendC::LayerNorm<T,T>` vendor primitive — precision 7/8 bit-exact (vendor saved
us) but perf 0.38× (we abandoned Welford + full-load). Cost: $69.20 + 244 min wall,
entirely wasted (postmortem at `output/a3_to_a5_port/src/kernels/ada_layer_norm.postmortem-handrolled-partial/`).

**Mechanical detection**:

```python
def has_upstream_a5_reference(op_dir: Path) -> bool:
    op_name = op_dir.name  # e.g. "adaptive_avg_pool3d"
    # 1. Single-op variant — arch35/ lives inside the op's own op_kernel/
    arch35 = op_dir / "op_kernel" / "arch35"
    if arch35.is_dir() and (any(arch35.glob("*.h")) or any(arch35.glob("*.cpp"))):
        return True
    # 2. Shared-common variant — arch35/ lives in a SIBLING <family>_common/ dir
    #    and op-specific _apt.cpp references it via `#include "../<family>_common/arch35/..."`.
    #    Greppable by op-name prefix on the file basename.
    parent = op_dir.parent
    for sibling in parent.glob("*_common"):
        shared_arch35 = sibling / "op_kernel" / "arch35"
        if not shared_arch35.is_dir():
            continue
        if any(shared_arch35.glob(f"{op_name}_*.h")) or any(shared_arch35.glob(f"{op_name}_*.cpp")):
            return True
    # 3. Unified-cpp variant — no arch35/ subdir, but op's single .cpp gates the
    #    A5 binary branch on __NPU_ARCH__ == 3510 and op_host registers ascend950.
    main_cpp = op_dir / "op_kernel" / f"{op_name}.cpp"
    def_cpp = op_dir / "op_host" / f"{op_name}_def.cpp"
    cfg_dir = op_dir / "op_host" / "config" / "ascend950"
    if main_cpp.is_file() and def_cpp.is_file() and cfg_dir.is_dir():
        src = main_cpp.read_text()
        if "__CCE_AICORE__" in src and "3003" in src and "3113" in src:
            # Negative-guard form: `#if __CCE_AICORE__ >= 220 && !(__NPU_ARCH__ == 3003 || == 3113)`
            # enables the branch on A5 (3510) — equivalent to "shipped for ascend950".
            return True
    return False
```

Three upstream-layout shapes covered:
- **Mode A (single-op)**: `<backward>/op_kernel/arch35/{*.h,*.cpp}` — used by `ada_layer_norm`, `fused_quant_mat_mul`.
- **Mode B (shared-common)**: `<backward>/../<family>_common/op_kernel/arch35/<op_name>_*.{h,cpp}` with the op's local `op_kernel/<op>_apt.cpp` doing `#include "../<family>_common/arch35/..."`. Used by `pooling/adaptive_avg_pool3d` (and sibling `adaptive_max_pool3d`) which share `pooling/adaptive_pool3d_common/op_kernel/arch35/`.
- **Mode C (unified-cpp + negative-guard)**: NO `arch35/` subdir. Upstream `op_kernel/<op>.cpp` gates dtype branches with `#if __CCE_AICORE__ >= 220 && !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))` — the negative guard EVALUATES TRUE on A5 (`__NPU_ARCH__ == 3510`), so the branch IS shipped. Confirmation signals (all three must be present): (i) the negative-guard pattern in `op_kernel/<op>.cpp`, (ii) `op_host/<op>_def.cpp` contains `AddConfig("ascend950")` (either flat or inside an `if (regbase)` block), (iii) `op_host/config/ascend950/<op>_{binary,simplified_key}.{json,ini}` files exist. Used by `foreach_reciprocal` (and the broader foreach unary family that pre-dates the arch35/ split convention).

Phase A.1.0 MUST inventory all three shapes and classify each finding as external target prior art or
as a pre-staged branch inside the selected source. Mode C is easy to miss because a literal
`ls arch35/` returns empty. No inventory result is a skip or acceptance verdict.

**What to do with the inventory**:

1. Record source location, revision, license, and whether the branch belongs to the selected source.
2. Extract API, layout, variant-coverage, and failure hypotheses without copying target bodies.
3. Generate or adapt a task-owned artifact from the selected arch22 source and declared contract.
4. Prove clean-build/current-binary provenance and compare with selected-source NPU truth.
5. Distill only the independently supported transformation pattern into the KB.

**Historical evidence (pre-RFC skip/mirror policy; retained for measurements and failure analysis,
not current prescriptive guidance)**:
- ada_layer_norm 2026-05-13: brief bug let kw write inferior version; corrective rule
  surfaced + applied retroactively (postmortem archived). Algorithm-level split (single
  → welford + full_load), introduces tilingkey dispatcher. Layout shape: Mode A (single-op `op_kernel/arch35/`).
- fused_quant_mat_mul 2026-05-13: upstream has `{<op>.cpp, _tiling_data.h, _tilingkey.h}`
  in arch35/; same SKIP rule applies. Layout shape: Mode A.
- adaptive_avg_pool3d 2026-05-14: **first Mode B (shared-common) witness**. Upstream
  arch35 kernel files live in sibling `pooling/adaptive_pool3d_common/op_kernel/arch35/`
  (shared with `adaptive_max_pool3d`), not in the op's own `op_kernel/arch35/` (which is
  empty). Op-specific `op_kernel/adaptive_avg_pool3d_apt.cpp` does `#include
  "../adaptive_pool3d_common/arch35/..."`. SKIP rule applies — extended `has_upstream_a5_reference()`
  above (Mode B branch) catches it. L-tier judgment: L2+L3 hybrid (MicroAPI Register-based
  big_kernel + parall_pool + SIMT) — first pooling-family witness for OL-153.
- foreach_reciprocal 2026-05-16: **first Mode C (unified-cpp + negative-guard) witness**. Literal `ls op_kernel/arch35/` returned EMPTY, yet upstream ships A5 binary via `op_kernel/foreach_reciprocal.cpp` with `#if __CCE_AICORE__ >= 220 && !(__NPU_ARCH__ == 3003 || == 3113)` enabling all three dtype branches on `__NPU_ARCH__ == 3510`, plus `op_host/foreach_reciprocal_def.cpp` calling `AICore().AddConfig("ascend950")` and `op_host/config/ascend950/{binary.json,simplified_key.ini}` present. The brief's `Step A.1.0` arch35-only check would have falsely classified this as Case B (greenfield), driving an unnecessary regen. With Mode C added, classification is correctly `SKIP_UPSTREAM_HAS_REFERENCE_VERIFIED` — L1 verbatim mirror of foreach_sqrt with orthogonal Sqrt→Reciprocal substitution is structurally cleaner restatement of upstream, not a missing port. Same shape predicted for the rest of the foreach unary family (erf, neg, sqrt itself, cos, sin, exp, tan).
- apply_adam_w_v2 2026-05-14: **Mode A + on-host prebuilt extreme case**. Upstream PR4778
  ships full `op_kernel/arch35/{DAG.h, _tiling_data.h, _tilingkey.h}` + `op_kernel/<op>_apt.cpp`
  + `op_host/{<op>_def.cpp, <op>_tiling_arch35.{cpp,h}, config/ascend950/{binary.json,simplified_key.ini}}`.
  AND the A5 host's installed CANN at `/data/cann_b103/cann-9.0.0/opp/built-in/.../ascendc/apply_adam_w_v2/`
  is **bit-identical (md5 matched)** to upstream, with 10 prebuilt `.o` files (tiling keys 101..110)
  already deployed under `opp/.../ascend950/ops_nn/apply_adam_w_v2/`. kw work collapsed to
  verbatim mirror + python aclnn runner + A5-vs-A3 output compare — no source edit at all.
  Layout shape: Mode A. Confirms the SKIP rule's outer bound: when the CANN install ALREADY
  ships compiled binaries, even a "regen for measurement" lane is wasteful — the spawn is
  purely a verification pass.
- ada_layer_norm 2026-05-17 (port_a3_to_a5, kw-1, **C43 postmortem replay**): **Mode A SKIP applied cleanly to the canonical postmortem op**. The same op whose 2026-05-13 hand-rolled regen produced the OL-141 motivating disaster (244 min wall, $69.20, perf 0.38×) was re-run with the SKIP path active. kw-1 produced a complete PR4778 ship layout — verbatim mirror of upstream `op_kernel/arch35/{welford.h, full_load.h, impl.h, common.h}` + `op_host/{<op>_def.cpp, _tiling.{cpp,h}, CMakeLists.txt, config/ascend950/{binary.json, simplified_key.ini}}` — in a single iter with NO kernel-level work, NO compute path written by the worker. verification.json verdict `SKIP_UPSTREAM_HAS_REFERENCE_VERIFIED` with precision.status=N/A per L1 verbatim-mirror semantics. Confirms OL-141's rule is correctly load-bearing: with the SKIP gate active, the C43 postmortem class becomes a 1-iter zero-regen pass.
- foreach_neg 2026-05-18 (port_a3_to_a5, kw-1→kw-2): **Mode A + P140 pybind, 2nd foreach-family witness**. Upstream `op_kernel/foreach_neg.cpp` uses negative-form `__NPU_ARCH__` guard (`!(NPU_ARCH==3003||3113)`) for BF16 — on A5 (`__NPU_ARCH__=3510`) the BF16 branch is enabled and the kernel compiles cleanly. Host-side `AddConfig("ascend950")` and `config/ascend950/foreach_neg_binary.json` already shipped upstream. Workspace `op_host/` + `op_kernel/` are verbatim mirrors (Mode A — no `arch35/` directory). Standalone pybind kernel adapted from 4_Abs template (348-line `kernel.h`, 4 dtype variants `ForeachNegSimd{F32,F16,BF16,I32}`) — adapted by: (i) rename namespaces+symbols (`AbsSimd*→ForeachNegSimd*`); (ii) substitute `Abs(y,x,count)` with the negation primitive — kw-1 used `Muls(y,x,T(-1),count)` and hit a signed-zero bit-pattern mismatch on case 8 (7/8 T1 + 1/8 T2-only), kw-2 switched to `Sub(y, zero_buf, x, count)` with a one-shot `Duplicate(zero_buf, T(0), TILE)` in `Init()` matching CANN's `ForeachImplictOutputLevelZeroApi<..., Sub, ...>` choice → 8/8 T1 bit-exact (see **OL-166** for the IEEE-754 rationale); (iii) add a 4th `I32` variant since 4_Abs only ships fp32/fp16/bf16; (iv) wrap pybind per-tensor call in `std::vector<at::Tensor>` loop for list semantics. Result: **8/8 cases bit-exact** across fp32/fp16/bf16/int32 (cleaner than foreach_sqrt — Neg is exact arithmetic, no transcendental drift), Option-2 wrapper-inclusive perf ratio **2.41× geomean** vs A3 `torch._foreach_neg` baseline (range 1.32× at 5-tiny-tensors → 3.05× at single bf16 1024-element). Confirms the foreach-family kw-1-first-iter-clean recipe holds for exact-arithmetic primitives too, not just transcendentals.
- foreach_sqrt 2026-05-17 (port_a3_to_a5, kw-1): **Mode A + SKIP→PASS_WITHIN_TOLERANCE elevation via P140 pybind**. Upstream `op_kernel/arch35/foreach_sqrt_regbase.h` (57 lines) uses `ForeachSqrtRegbase<T, Tiling, Self>` with `Compute(inLocal, outLocal, dataCount)` body = Cast→Sqrt→Cast for fp16/bf16, direct Sqrt for fp32. Ship-layout artifact: verbatim mirror of upstream `op_host/` + `op_kernel/arch35/` files (no diffs). The novel contribution beyond foreach_reciprocal's Mode C SKIP precedent is the **empirical verify path**: a standalone pybind kernel under `workspace/foreach_sqrt/kernel/` derived mechanically from the upstream regbase by (i) stripping `ForeachRegbase<...>` base class, substituting per-tensor `TQue<VECIN>+TQue<VECOUT>+TBuf` pipeline (1_GELU pattern with QDEPTH=4, TILE_F32=4096, TILE_HALF=8192); (ii) copying the regbase `Compute()` body verbatim; (iii) wiring to `ACLRT_LAUNCH_KERNEL` via pybind11 with the Python loop driving foreach semantics. Result: 8/8 cases PASS at T2 within fp32-ULP, 5/8 ALSO bit-exact T1 (all 3/3 bf16 cases bit-exact; 3 non-bit-exact cases are textbook 1-ULP cross-platform drift per OL-103 + P-P88), Option-2 wrapper-inclusive perf ratio 1.95–2.47× median across all 8 cases vs A3 `torch._foreach_sqrt` baseline. P140 path bypasses the libcust_opapi.so install blocker (protobuf/GLIBCXX deficit) that forced foreach_reciprocal into `SKIP_UPSTREAM_HAS_REFERENCE_VERIFIED` (precision.status="N/A"). **Generalization**: ANY upstream-shipped foreach unary op (Mode A or Mode C) can follow this kw-1-first-iter-clean recipe — mechanical mirror + standalone pybind wrap + edge_runner comparison — and earn `PASS_WITHIN_TOLERANCE` verdict instead of falling back to SKIP. Estimated budget: 1 kw iter per sibling foreach op (erf, neg, cos, sin, exp, tan, ...). The heavy lifting is upstream's; our job is mechanical mirror + pybind wrap + edge_dataset compare.

**Current generalization**: every future arch22→arch35 port inventories target prior art before
generation, but source capture and task-owned generation remain mandatory regardless of target-side
availability.

**Cross-reference**:
- OL-134 (Phase O2.5 port-complexity estimator) — extend it to inventory arch35/ prior art
- OL-139 (arch35 dispatch-branch enumeration) — applies to greenfield ports where
  arch35/ doesn't yet exist
- `kw_brief.py §A.1.0` — mandatory upstream arch35/ list as first read

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-141（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
