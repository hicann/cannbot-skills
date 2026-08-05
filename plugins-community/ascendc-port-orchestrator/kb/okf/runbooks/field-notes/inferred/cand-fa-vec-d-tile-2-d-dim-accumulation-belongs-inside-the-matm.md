---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "D-dim accumulation belongs INSIDE the Matmul library (V220) / inside FA BaseApi (V351), NOT in kernel-level d-tile loop with manual SetFlag/WaitFlag"
description: "applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=fused_attention_with_large_head_dim_D = 256 / FlashAttention_D_512_or_768 derived-from: cann-source (FA reference V220 bn2gs1s2_b.h"
phenomenon: build_failure
signal:
  - "Implementing FlashAttention (forward) with head-dim D ≥ 256 on V220 or D ≥ 512 on V351. Temptation: write an explicit d-tile loop in the kernel for (d_tile = 0;"
confidence: inferred
status: stub
original_id: CAND-FA-VEC-D-TILE-2
timestamp_inferred: true
tags: [candidate, inferred, d_base_size, flash_attention_score_template_tiling_key.h, matmulimpl, crosscoresetflag, fa_v220, cand-fa-vec-d-tile-2]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=fused_attention_with_large_head_dim_D >= 256 / FlashAttention_D_512_or_768`
`derived-from: cann-source (FA reference V220 bn2gs1s2_b.h + V351 kernel_train, 2026-05-24 cl-fa-diff)`
`evidence_family: FA-GQA-DIM`
`verified_on: cann ops-transformer FA reference V220 + V351 dispatch tables`

**Trigger**: Implementing FlashAttention (forward) with head-dim D ≥ 256 on V220 or D ≥ 512 on V351. Temptation: write an explicit d-tile loop in the kernel `for (d_tile = 0; d_tile < D / D_TILE; ++d_tile) { ... DataCopy Q[s1, d_tile*D_TILE:(d_tile+1)*D_TILE] ...; bmm1_partial; accumulate; }` and synchronize d-tile boundaries with manual `SetFlag<HardEvent::V_V> / WaitFlag<HardEvent::V_V>` or `PipeBarrier<PIPE_V>`. This is the path the independent prototype row-tiled fp16 kernel takes for D=512 with D_TILE=128.

**Why "candidate"**: structural pattern derived from how CANN's FA reference dispatches large-D shapes; symptom-link to independent prototype DEBT-FA-GQA is hypothesis, not validated. Need one more L4-FA port to confirm.

**Recommendation**: The CANN FA reference does NOT do kernel-level d-tile splitting. Instead:
- V220 large-D path (when D doesn't fit a single template-dispatch key): tiling table sets a per-core `d_base_size` field consumed by the matmul library tiling policy; the kernel calls `bmm1.SetTail(s1_real, d_size, s2_real); bmm1.IterateAll(workspace_ping, ...)` with d-direction accumulation handled by the high-level `matmul::Matmul<>` library internally. Kernel level has NO d-loop — only outer (b/n2/g/s1) loops.
- V351 large-D path: `flash_attention_score_template_tiling_key.h` includes a discrete D-template key for `D=768`, indicating UB-resident Q[s1_base, 768] is feasible without splitting. The kernel relies on the FA BaseApi base class to manage d-tile accumulation if needed.

For a VEC-only port (when AIC + Matmul library is unavailable, e.g. AIV-only fallback path used in DEBT-FA-GQA), kernel-level d-tile loop IS unavoidable. In that case the safe shape is:
1. Use **`HardEvent::MTE2_V` SetFlag/WaitFlag PER d-tile iteration** between the DataCopy of Q[s1, d_tile] and the Mul/Madd of scores += Q[s1, d_tile] * K^T[d_tile, s2].
2. Do NOT use `PipeBarrier<PIPE_V>` between d-tile iterations of the same s1 row — see PB-21 (V220 PipeBarrier<PIPE_ALL>-on-TBuf silent crash 507015). Use explicit event flags.
3. Keep K^T tile in UB across the full d-loop (load once per s1-block, reuse for all d-tiles); only Q gets reloaded per d-tile.

**Concrete anchor** (public-API VEC-only d-tile loop for fp16 D=512 row-tiled FA; only used when AIC path is unavailable):
```cpp
// One Q-row block in flight; K^T full-S2 × full-D-tile in UB (re-used across d-tiles)
LocalTensor<half> q_tile = q_que.AllocTensor<half>();    // sized for [s1_base, d_tile_size]
LocalTensor<half> kt_tile = kt_buf.Get<half>();           // sized for [d_tile_size, s2_real]
LocalTensor<float> scores = scores_buf.Get<float>();     // [s1_base, s2_real], persistent across d-tiles
Duplicate(scores, 0.0f, s1_base * s2_real);
event_t e_mte2_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

for (int d_tile = 0; d_tile < d_size / d_tile_size; ++d_tile) {
    DataCopy(q_tile, q_gm[q_off + d_tile * d_tile_size], s1_base * d_tile_size);
    DataCopy(kt_tile, kt_gm[kt_off + d_tile * d_tile_size * s2_real], d_tile_size * s2_real);
    SetFlag<HardEvent::MTE2_V>(e_mte2_v);
    WaitFlag<HardEvent::MTE2_V>(e_mte2_v);
    // Accumulate scores += q_tile @ kt_tile^T  using VEC MMA-style Mul+Add chain
    // (kept abstract — the chain depends on s1_base × s2_real shape; emit Madd / Mul + Add as fits.)
}
PipeBarrier<PIPE_V>();  // OK here: post-d-loop, before softmax; not between d-tiles.
// scores fully accumulated; continue with softmax + bmm2.
```

**Reject_cond**: do NOT apply this pattern when:
- The op is using high-level `matmul::Matmul<>` / `MatmulImpl<>` — the library handles d-tile internally; kernel-level d-loop is wrong-direction.
- D ≤ 128 single-template dispatch path — the full Q tile fits in UB without splitting; no d-loop needed.
- The kernel mixes `MatmulImpl` + manual `CrossCoreSetFlag` — that's the PB-34 / CAND-FA1 deadlock zone, deal with that first.

**Symptom anchor**: independent prototype `fa_v220` D=512 D_TILE=128 (single-iter d-loop) hang at `LaunchAscendKernel 507035`. HYPOTHESIS: the hang is sync-related (likely the d-tile boundary's MTE2→V handoff missing or wrong-event-ID), NOT UB-budget-related (UB at 192 KB easily fits 64-row × D=512 fp16 = 64 KB). The reject_cond above flags that if the kernel ALSO uses MatmulImpl + manual CrossCore, the d-tile fix won't resolve the underlying deadlock.

**Other-instances-predicted**: any large-D attention port (D=256, 384, 512, 768), any L4-fused kernel decomposing into a `for d_tile { mm1_partial; accumulate; }` shape on the VEC-only path.

**Promote when**: independent prototype DEBT-FA-GQA resolves with this d-loop sync shape AND a separate D=768 FA-class op verifies the same shape works.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-VEC-D-TILE-2，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
