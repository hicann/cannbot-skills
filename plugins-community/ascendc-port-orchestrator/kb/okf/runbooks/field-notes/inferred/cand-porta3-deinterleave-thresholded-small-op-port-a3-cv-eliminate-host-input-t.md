---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "small-op port_a3 CV — eliminate host input-transpose overhead via THRESHOLDED in-kernel de-interleave"
description: "applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=cv-box/port_a3-vec verified_on: soc=Ascend950PR; cann=9.1.0 (A5 Ascend950PR_9579) unverified_on: soc=Ascend910_V220 Principle: For small-"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=cv-box/port_a3-vec"
confidence: inferred
status: stub
original_id: CAND-PORTA3-DEINTERLEAVE-THRESHOLDED
timestamp_inferred: true
tags: [candidate, inferred, boxmajor, gather, datacopypad, srcstride, cand-porta3-deinterleave-thresholded]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=cv-box/port_a3-vec`
`verified_on: soc=Ascend950PR; cann=9.1.0 (A5 Ascend950PR_9579)`
`unverified_on: soc=Ascend910_V220`

**Principle**: For small-op port_a3 CV kernels, the host `.t().contiguous()` input marshaling ((m,4)↔(4,m) coord-major↔box-major, mirroring the op's `op_api/aclnn_*.cpp` `l0op::Transpose`) can DOMINATE the per-call launch-overhead floor — for iou_v2, the 2 input transpose kernels = ~40% of a flat ~0.10-0.12ms floor (the IoU compute is invisible; device-time flat vs m). Eliminate them with a THRESHOLDED in-kernel de-interleave: for small m (≤128) pybind passes box-major `(m,4)` directly + a `boxMajor` flag, the kernel contiguous-loads `(cnt,4)` then does a scalar UB-shuffle `dst.SetValue(p*cnt+i, src.GetValue(i*4+p))` over p∈0..3 (bit-exact, ~30us faster); for large m (>128) KEEP the host transpose — the scalar-shuffle is O(m) and loses past the crossover (m=64..1009), and large m is already 30-57×.

**Confirmed-WRONG primitives (do NOT repeat)**: (1) AscendC `Gather` intrinsic — RUNTIME 507035 (aivec 271, scalar internal-buffer OOB) on V351, both byte and element offset-unit forms; (2) strided multi-block `DataCopyPad` (byte-gap `srcStride`) gather — precision regressed 33/33→7/33; the byte-gap `srcStride` semantics PB-22 verified for *contiguous-tail* copies do NOT hold for a sub-32B multi-block *gather* (OL-85 overfit signature).

**Evidence**: iou_v2 ko-1 (2026-06-21, A5 Ascend950PR_9579 NPU6, P141 device-event). precision 33/33 preserved + det 33/33; perf median 0.45→0.67-0.90×, geomean 0.97→0.98-1.20× (2 runs, shared-host variance); small-m WINS (m=8 0.36→1.42×, m=64 0.25→1.21×). Vendor-grounded: A3-TBE is also small-op overhead-bound but at a 3-5× lower launch floor → real-but-partly-closeable gap (NOT a flat ceiling); mid-m 256-1024 residual = vendor-grounded floor.

**Other instances (predicted)**: modulate, upsample_{bilinear2d,bicubic2d,nearest}, roi_align_rotated — any port_a3 CV op whose `op_api/aclnn_*.cpp` does `l0op::Transpose`/`Contiguous` input marshaling. Cross-op KB asset for the CV cohort.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PORTA3-DEINTERLEAVE-THRESHOLDED，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
