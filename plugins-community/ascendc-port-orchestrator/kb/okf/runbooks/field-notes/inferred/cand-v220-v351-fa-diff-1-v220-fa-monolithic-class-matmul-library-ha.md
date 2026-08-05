---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V220 FA monolithic-class + Matmul-library handoff vs V351 FA per-engine block-types + ASCEND_IS_AIC/AIV gates — the structural pattern for porting FA-class kernels across V220/V351"
description: "applies_to: soc=Ascend910_V220 ↔ Ascend950PR cross-arch port; cann=9.0.0+; op_class=fused_attention_port_a3_to_a5 / FlashAttention_arch22_to_arch35 derived-from: cann-source (FA reference V220 arch22/"
phenomenon: build_failure
signal:
  - "Porting a V220 FA-class kernel (single templated class with IterateBmm1 / ProcessVec1 / IterateBmm2 / ProcessVec2 methods, 3-deep per-stage info array ping-pong"
confidence: inferred
status: stub
original_id: CAND-V220-V351-FA-DIFF-1
timestamp_inferred: true
tags: [candidate, inferred, iteratebmm1, processvec1, iteratebmm2, processvec2, arch22, cand-v220-v351-fa-diff-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_V220 ↔ Ascend950PR cross-arch port; cann=9.0.0+; op_class=fused_attention_port_a3_to_a5 / FlashAttention_arch22_to_arch35`
`derived-from: cann-source (FA reference V220 arch22/* + V351 arch35/*, 2026-05-24 cl-fa-diff)`
`evidence_family: V351-SYNC-MODE / port_a3_to_a5_FA`
`verified_on: cann ops-transformer FA reference V220 + V351 top-level Process structure + entry macros + template_tiling_key dispatch tables`

**Trigger**: Porting a V220 FA-class kernel (single templated class with `IterateBmm1` / `ProcessVec1` / `IterateBmm2` / `ProcessVec2` methods, 3-deep per-stage info array ping-pong, monolithic source) to V351 / Ascend950PR. Temptation: copy V220 source verbatim, swap `arch22` includes for `arch35` equivalents, hope for the best.

**Why "candidate"**: cross-arch structural pattern derived from comparing V220 and V351 FA reference. The per-engine block-type shape on V351 is mandatory architecturally (V351 toolchain expects `ASCEND_IS_AIC/AIV` partition); structural recommendation is unambiguous from source.

**Recommendation**: V220 and V351 FA reference share the SAME outer algorithm (online softmax, ping-pong pipeline, GQA index decode, KV-shared dispatch) but differ in 4 specific structural axes. When porting, refactor along these axes:

1. **Per-engine block-types**: V351 expects two distinct template classes — one with AIC-only methods, one with AIV-only methods — instantiated via `std::conditional<g_coreType == AscendC::AIC, CubeBlockType, VecBlockType>`. V220's monolithic class with internal `if constexpr` cube/vec guards is NOT the V351 shape; refactor to split.
2. **`ASCEND_IS_AIC` / `ASCEND_IS_AIV` source guards**: V351 Process() body uses `if ASCEND_IS_AIC { cubeBlock.IterateBmm1(...); }` / `if ASCEND_IS_AIV { vecBlock.ProcessVec1(...); }`. Same .o file compiles both paths; the guard runs at compile time. V220's pattern (one class, one Process method) does not transfer; the V351 source must be explicitly partitioned.
3. **Pipeline depth**: V220 = 3-deep (per-stage info array of size 3, indexed via `taskId % 3`); V351 = 4-deep (per-stage info array of size 4, indexed via `taskId & 3`). When porting, add one more pipeline slot — V351's wider Matmul/Vec capacity expects it.
4. **D-template-key range**: V220 supports D-template values from a narrow set ({5, 6, 8} indices, ~3 specific D sizes); V351 supports a much wider set ({16, 32, 48, 64, 80, 96, 128, 160, 192, 256, 768}). When porting, re-validate the D-template-key dispatch — D=512 falls into the "general path" on V220 but might dispatch to the D=768 template on V351, which uses different inner UB layout assumptions.

**Concrete anchor** (V351 per-engine partition skeleton, public-API only):
```cpp
// V351 (Ascend950PR / arch35) top-level Process — public-API shape
namespace MyOp {

template <typename CubeBlockType, typename VecBlockType>
class MyFusedKernel : public BaseFAClass<MyFusedKernel<CubeBlockType, VecBlockType>, CubeBlockType, VecBlockType> {
public:
    __aicore__ inline void Process() {
        RunInfo runInfo[4];  // 4-deep ping-pong
        int64_t taskId = 0;
        for (int64_t outer = 0; outer < outerLimit + 3; ++outer) {
            bool notLast = (outer < outerLimit);
            // ... pipeline stage gating ...

            if (notLast) {
                this->ComputeAxisIdx(outer, runInfo[taskId & 3]);
                if ASCEND_IS_AIC {
                    this->cubeBlock.IterateBmm1(runInfo[taskId & 3]);
                }
            }
            if (taskId > 0 && notLast) {
                if ASCEND_IS_AIV {
                    this->vecBlock.ProcessVec1(runInfo[(taskId + 3) & 3]);
                }
            }
            if (taskId > 1 && notLast) {
                if ASCEND_IS_AIC {
                    this->cubeBlock.IterateBmm2(runInfo[(taskId + 2) & 3]);
                }
            }
            if (taskId > 2) {
                if ASCEND_IS_AIV {
                    this->vecBlock.ProcessVec2(runInfo[(taskId + 1) & 3]);
                }
            }
            ++taskId;
        }
    }
};

}  // namespace MyOp

// Entry macro — public AscendC kernel registration with MIX_AIC_1_2
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
```

Critical: the `if ASCEND_IS_AIC` / `if ASCEND_IS_AIV` are NOT runtime branches — they expand to compile-time conditional that gates which engine compiles which sub-tree. The .o file built with `__DAV_C310_CUBE__` only contains the AIC sub-tree; the one built with `__DAV_C310_VEC__` only contains the AIV sub-tree. The MIX_AIC_1_2 launch invokes both .o files paired.

**Reject_cond**: do NOT use this V220→V351 refactor pattern when:
- The op is **not** fused-attention class (doesn't have the bmm1/softmax/bmm2 3-stage structure). Other fused ops use different cross-engine patterns; CAND-FA1 / CAND-FA-VEC-D-TILE-* still apply per-arch but the per-engine block-type split here is FA-specific.
- The port target is V220 (the current canonical KB direction is V351 → V220 not the reverse). For V220 ports, this candidate doesn't apply.
- The op uses the high-level `matmul::Matmul<>` library on V220 — the library on V351 handles per-engine internally, so the explicit per-engine split is redundant. This candidate's value is for ops that fully decompose into kernel-level AIC and AIV phases (FA, MoE-finalize, fused-MLA).

**Symptom anchor**: independent prototype `fa_v220` (V220) is currently the only working A3 FA kernel; any future V351/A5 port (post DEBT-FA-V351) will hit the per-engine partition wall. The kw-5 silent-hang in CAND-FA1 / PB-34 was on V220, but the structural shape mismatch (monolithic class + MatmulImpl + manual CrossCore) was a V351-style anti-pattern on V220. Honoring this candidate's per-engine partition during a V351 port avoids inheriting that anti-pattern.

**Other-instances-predicted**: any V220→V351 port of an L4-fused op decomposing into cube+vec phases (MoE-finalize, fused-MLA, fused-RMSNorm-then-attention). Same per-engine block-type pattern applies.

**Promote when**: a successful V220→V351 FA port lands using this structural shape (per-engine block-types + ASCEND_IS_AIC/AIV gates + 4-deep pipeline + V351 D-template-key re-validation) AND a second port-a3_to_a5 fused op (e.g. fused-MLA) reuses the same shape.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-V220-V351-FA-DIFF-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
