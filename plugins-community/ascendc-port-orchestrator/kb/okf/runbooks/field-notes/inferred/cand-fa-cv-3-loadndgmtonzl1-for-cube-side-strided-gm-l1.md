---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "LoadNdGmToNzL1 for Cube-side strided GM→L1 with dimension-aware tiling"
description: "applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; op_class=cube_matmul_with_strided_input derived-from: cv-agent flash_attention_cube.h:57 — LoadQ implementation verified_on: cv-agent stock FA 16/16"
phenomenon: build_failure
signal:
  - "Cube matmul (Mmad/Fixpipe) needs to load input tensors from GM→L1 where the input layout may be strided (e.g., Q[batch, head, seq, dim] → cube processes per-(b,"
confidence: inferred
status: stub
original_id: CAND-FA-CV-3
timestamp_inferred: true
tags: [candidate, inferred, datacopy, tque, loadndgmtonzl1, cand-fa-cv-3]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; op_class=cube_matmul_with_strided_input`
`derived-from: cv-agent flash_attention_cube.h:57 — LoadQ implementation`
`verified_on: cv-agent stock FA 16/16 PASS (cube matmul1 Q@K^T + matmul2 P@V both use this pattern)`

**Trigger**: Cube matmul (Mmad/Fixpipe) needs to load input tensors from GM→L1 where the input layout may be strided (e.g., Q[batch, head, seq, dim] → cube processes per-(b,h) tile with stride=dim in GM).

**Pattern**: `LoadNdGmToNzL1(dst_L1, gm_src, M, K, stride)` loads M×K block from GM into L1 in Nz format (V220 cube-native layout), handling dimension-aware stride. Followed by `SetWaitFlag<HardEvent::MTE2_MTE1>()` to signal L1 data ready for cube pipeline.

**Our gap**: a5_ops matmul ops (1_BatchMatmul, 3_MatmulBothTrans) use `DataCopy` or `TQue` for GM→UB loads, then manually reorganize for cube. `LoadNdGmToNzL1` combines load + layout transform in one hardware operation.

**Detection**: grep for `DataCopy.*L1\|TQue.*A1` in kernel .h files. If cube matmul path uses `DataCopy` or `TQue` for L1 staging instead of `LoadNdGmToNzL1` → pattern missing.

**Evidence**: cv-agent `flash_attention_cube.h:57` — `LoadNdGmToNzL1(qL1, qGm_[qOffset], BLOCK_M, dim, dim)` at LoadQ entry.

**Cross-ref**: P-P47 (VEC halving reduction), EC-61 (scalar-pipe accumulator → VEC pipe)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-CV-3，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
