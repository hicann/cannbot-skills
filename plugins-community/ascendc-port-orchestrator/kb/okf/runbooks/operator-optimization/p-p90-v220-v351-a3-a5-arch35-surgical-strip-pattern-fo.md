---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "V220→V351 (arch22→arch35) — surgical strip pattern for op_kernel port"
description: "> W10 (2026-05-12, ROADMAP §1.5) — extracted from ctc_loss_v3_a5_migration_plan.md §7 (PR4778 docs) + diff op_kernel/ctc_loss_v3.h vs op_kernel/arch35/ctc_loss_v3.h (831 vs 830 lines — surgical change"
confidence: single_run
original_id: P-P90
timestamp_inferred: true
tags: [platform_compat, optimization, a3_to_a5_port, gather_elements_v2, welfordupdate, dav_c220, __npu_arch__, p-p90, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

> **W10** (2026-05-12, ROADMAP §1.5) — extracted from `ctc_loss_v3_a5_migration_plan.md` §7 (PR4778 docs) + diff `op_kernel/ctc_loss_v3.h` vs `op_kernel/arch35/ctc_loss_v3.h` (831 vs 830 lines — surgical change, not rewrite).
>
> **ID rename note (2026-05-12)**: this entry was originally P-P89 in commit `267667a` (W8-W11 batch). Renamed to P-P90 in commit (this commit) to resolve collision with pre-existing `PATTERN_INDEX.md:148` P-P89 ("GM workspace contract for fused ops") from commit `21882d4`. References in OL-131/OL-132 cross-refs + KB_INDEX A3→A5 section + W12 op_taxonomy + test_port_a3_integration_smoke.py + ascend950pr.md + output/a3_to_a5_port/ project docs updated in same commit.

### Trigger conditions

- Op-class tag: `a3_to_a5_port`
- Phase: B.1 of port_from_a3_ascendc kw_brief (writing `op_kernel/arch35/<op>.h`)
- Input: existing `op_kernel/<op>.h` (A3/V220 kernel) — the algorithm spec

### Pattern

The V220→V351 port is mostly a **strip** operation, not a rewrite. Take the A3 kernel as starting point; remove or adjust these specific items:

1. **Strip V220 reg-primitives include**:
   ```cpp
   // REMOVE this line (V220-only):
   #include "impl/dav_c220/kernel_operator_reg_others_impl.h"
   ```
   A5 (arch35) provides reg-based primitives via default `kernel_operator.h`; the V220 explicit include conflicts on V351.

2. **Strip BF16 conditional compile blocks**:
   ```cpp
   // REMOVE:
   #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
       // V220-specific BF16 codepath
   #else
       // generic codepath
   #endif
   ```
   On V351, reg-based VEC ops support BF16/FP16/FP32 unconditionally — the V220 conditional was masking V220 BF16 quirks that don't exist on V351. Keep the body of the `#else` branch (the generic codepath); delete the `#if`/`#endif` and the V220 branch.

3. **`ToFloat<>` audit** — A5 restricts `ToFloat<T>` to `T ∈ {bfloat16_t, fp8_e5m2_t, fp8_e4m3fn_t}`. For FP16 source values, insert `.template ReinterpretCast<bfloat16_t>()` first:
   ```cpp
   // V220 (works on A3 with FP16 directly):
   float v = ToFloat(logProbTensor.GetValue(0));     // T = half

   // V351 (must reinterpret to bfloat16 first):
   float v = ToFloat(logProbTensor.template ReinterpretCast<bfloat16_t>().GetValue(0));
   ```
   See **W11** for the full ToFloat<> A5 restriction reference.

4. **Tiling header includes — generally unchanged**:
   - `kernel_operator.h` — keep (A5 native)
   - `kernel_tiling/kernel_tiling.h` — keep (target-agnostic)
   - The arch35/ kernel typically needs no new includes beyond these two; everything else was V220-specific noise.

### Anti-pattern (DO NOT)

- Do NOT delete the algorithm body. The A3 algorithm is what we want; only the platform plumbing changes.
- Do NOT add a `#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 300` wrapper. The arch35/ file is V351-only by virtue of its directory; conditional compile inside is redundant + confusing.
- Do NOT include `impl/dav_v300/*` headers explicitly unless a specific primitive requires it (rare; usually `kernel_operator.h` covers everything via target macros).
- Do NOT use the V220 conditional as a "where to insert V351 code" landmark and just flip the macro — that pattern leaves dead `#elif __CCE_AICORE__ == 200` (V220 single-die) branches that will compile on V200 builds.

### Diff size sanity-check

For L1 / L2 ports (gather_elements_v2, ctc_loss_v3, rms_norm_quant, top_k_top_p_sample_v2, group_norm_silu_quant from PR4778), `wc -l op_kernel/<op>.h` vs `wc -l op_kernel/arch35/<op>.h` typically differs by < 10 lines (just strip operations). If your `arch35/<op>.h` is more than ~10% different in line count, you're likely rewriting instead of porting — re-check Phase A analysis.md.

### Two-stage authoring pattern (recommended over derivation-strip)

Added 2026-05-12 from `gather_elements_v2` kw-1 finding: PR4778's arch35 kernel files are NOT a V220 strip-and-edit — they are **freshly authored** from the algorithm spec. Audit:
```bash
grep -nE "dav_c220|__CCE_AICORE__ == 220|ToFloat<|__NPU_ARCH__" \
    gather_elements_v2/op_kernel/arch35/*.h gather_elements_v2/op_kernel/arch35/*.cpp
# → zero matches
```

This is the **better** authoring pattern when feasible:
- **Derivation-strip** (this entry's main body): the arch35 kernel is derived from the V220 kernel by applying the 4 strip operations. Lowers cognitive load (algorithm body is copied) but inherits any V220 plumbing quirks unless every strip rule is applied carefully. Use when the V220 algorithm is the authoritative spec and no clean arch35 reference exists.
- **Fresh authoring** (gather_elements_v2 model): the arch35 kernel is written from scratch using the algorithm spec + arch35 native primitives (reg-based MicroAPI per CAND-A3A5-5, `WelfordUpdate` per OL-135, etc.) — V220 kernel kept in place untouched for backward compat. Cleaner end state; zero V220-leftover risk; reviewer-friendly. **Prefer this when the algorithm spec is independently authoritative AND the author can sustain the cost of writing two parallel kernels.**

Decision rule: if a `git diff master..FETCH_HEAD -- <op>/op_kernel/arch35/` reveals the arch35 files are NEW (not derived), apply audit but do NOT apply strip rules — they're already not present. If the diff reveals the arch35 files share most of the V220 body, apply the 4 strip rules above.

### Evidence

- ctc_loss_v3 (PR4778): **Derivation-strip path** — A3 = 831 lines → A5 arch35 = 830 lines (1-line strip of `impl/dav_c220/` include + `__CCE_AICORE__ == 220` conditional removal; rest identical). User-verified via `git show FETCH_HEAD:loss/ctc_loss_v3/op_kernel/arch35/ctc_loss_v3.h` vs `loss/ctc_loss_v3/op_kernel/ctc_loss_v3.h` on 2026-05-12.
- gather_elements_v2 (PR4778): **Fresh-authoring path** — 4 arch35 `.h` files (common + scalar + transpose + last_dim) are independently authored, no derivation from the master V220 `.cpp`. Audit grep returned 0 matches for `dav_c220`/`__CCE_AICORE__ == 220`/`ToFloat<`/`__NPU_ARCH__`. V220 master kept in place unchanged. 8/8 T1 bit-exact PASS on A5 pipeline-wiring verify (2026-05-12 kw-1).

### Cross-ref

- **W8** `ops_nn_layout/ops_nn_a5_artifact_layout.md` — what the arch35/<op>.h fits into
- **W11** `hardware/target/ascend950pr.md §Reg-based intrinsics restrictions` — full ToFloat<> rule
- **W9** OL-131 (cross-op router) — orthogonal host-side change for v2/v3-shared-aclnn ops

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/platform_compat.md（P-P90，convert_patterns_to_okf.py）。confidence 未升格。 -->
