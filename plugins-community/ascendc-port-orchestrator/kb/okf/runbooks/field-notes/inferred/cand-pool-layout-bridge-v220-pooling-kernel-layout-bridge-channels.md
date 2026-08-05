---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V220 pooling kernel layout bridge — channels-last (NDHWC) algorithm vs PyTorch channels-first (NCDHW) via pybind permute"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=pooling verified_on: adaptive_avg_pool3d (V220→A5 L1 port, 2026-06-16) unverified_on: other V220 pooling ops (adaptive_max_pool3d, avg_po"
phenomenon: build_failure
signal:
  - "A V220 pooling algorithm (e.g., adaptive_avg_pool3d SplitC/SplitW/MultiW) assumes channels-last memory layout (D, H, W, C — NDHWC), but PyTorch's tensor convent"
confidence: inferred
status: stub
original_id: CAND-POOL-LAYOUT-BRIDGE
timestamp_inferred: true
tags: [candidate, inferred, cand-pool-layout-bridge]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=pooling`
`verified_on: adaptive_avg_pool3d (V220→A5 L1 port, 2026-06-16)`
`unverified_on: other V220 pooling ops (adaptive_max_pool3d, avg_pool2d, max_pool2d); non-pooling V220 ops with channels-last native layout`

**Trigger**: A V220 pooling algorithm (e.g., adaptive_avg_pool3d SplitC/SplitW/MultiW) assumes channels-last memory layout (D, H, W, C — NDHWC), but PyTorch's tensor convention is channels-first (C, D, H, W — NCDHW). A direct port of the V220 kernel reads memory in the wrong order, producing wrong outputs.

**Recommendation**: Do NOT rewrite the kernel algorithm to use channels-first. Instead, bridge the layout mismatch in the pybind layer:
1. `input_ncdhw.permute(0, 2, 3, 4, 1).contiguous()` → NDHWC before kernel launch
2. Launch the (unchanged) V220-algorithm kernel on NDHWC input
3. `output_ndhwc.permute(0, 4, 1, 2, 3).contiguous()` → NCDHW after kernel returns

**Evidence**: adaptive_avg_pool3d L1 V220→A5 port (2026-06-16): initial direct port produced wrong outputs on all 30 cases. Root cause identified as NCDHW→NDHWC layout mismatch. Fix in pybind (permute before kernel + after kernel) resolved all 30/30 cases to bit-exact vs CPU truth. The V220 kernel algorithm was preserved unchanged — only the pybind interface adapted.

**Hard do-not-apply**:
- Do NOT use this pattern when the V220 algorithm HAS a channels-first code path — prefer the native path over the pybind bridge.
- Do NOT permute inside the kernel (wastes UB on layout transform) — keep it in host/pybind where it is a one-time cost per launch.
- Do NOT apply blindly to non-pooling V220 ops — verify the algorithm's assumed layout first by reading the op_host/kernel source.

**Cross-reference**: OL-141 (target arch35 is advisory; generate from selected arch22 source rather
than wrapping target code); selected-source pre-stage policy.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-POOL-LAYOUT-BRIDGE，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
