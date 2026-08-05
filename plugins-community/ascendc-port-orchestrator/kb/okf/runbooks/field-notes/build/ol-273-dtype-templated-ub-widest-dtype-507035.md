---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Size a dtype-templated kernel's UB for the WIDEST dtype and run the full multi-dtype verify (507035)"
description: "A template<T> kernel sizing UB from an element count overflows at fp32 (double the fp16/bf16 footprint) → 507035 on the first fp32 case; a narrow-dtype-only capture falsely passes."
phenomenon: build_failure
signal:
  - "Device error 507035 on the FIRST fp32 case of a dtype-templated (fp32/fp16/bf16) elementwise kernel, with a clean build and a healthy device; bf16/fp16 cases PASS, masking the fp32-only UB overflow."
confidence: single_run
original_id: OL-273
classified_by: llm-assisted
timestamp_inferred: true
tags: [ub-budget, build, ol-273, 507035, multi-dtype, elementwise]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发

When a kernel templates over dtype (`template<typename T>` with fp32 / fp16 / bf16 instantiations) and sizes its UB queues/TBufs from an element **count** (`TILE_ELEMS × sizeof(T)`), the physical UB footprint **doubles** from the 2-byte dtypes (fp16/bf16) to the 4-byte dtype (fp32). A tile count that fits in UB at `sizeof(T)=2` can overflow UB at `sizeof(T)=4`. The overflow surfaces as **device error 507035** on the **first wider-dtype (fp32) case**, with a clean build and a healthy device (see [[OL-274]] for the read-only vendor-op probe that proves it is a kernel fault, not infra).

**Coverage blind spot (the compounding trap):** a capture that ran only the narrow dtype (bf16/fp16) PASSES and reads as a false-complete, because the fp32-only footprint was never exercised. This is the concrete UB-budget instance of the "simple op → run the FULL multi-dtype verify" rule: a multi-dtype elementwise kernel MUST be validated on **every** instantiated dtype, and the buffer budget MUST be checked against the **widest** one.

Verified on Ascend950PR / CANN 9.0.0. **Unverified on Ascend910_V220 (A3)**: A3 UB is 192 KB (not 256 KB), so the fp32/bf16 fit boundary shifts — the principle holds but the exact tile that fits differs.

## 根因 / 教训

Size the UB budget for the widest instantiated dtype, then verify all dtypes.

For a double-buffered elementwise unary with two fp32 scratch TBufs, the fp32-worst-case footprint is:

```
UB = (inQ + outQ) 2·BUFFER_NUM·TILE·sizeof(T)  +  2 fp32 TBuf 2·TILE·4
   fp32 @ BUFFER_NUM=2, TILE=4096:  2·2·4096·4 + 2·4096·4 =  96 KB  (fits, margin under 256KB AIV UB)
   fp32 @ BUFFER_NUM=4, TILE=8192:  2·4·8192·4 + 2·8192·4 = 320 KB  → 507035
```

**Iron-rule sizing:** keep `(2·depth + n_fp32_tbuf) · TILE · sizeof(widest_dtype) ≤ UB/2` (headroom for stack + tiling scratch). Fix an overflow by lowering `BUFFER_NUM` and/or `TILE_ELEMS` — **not** by dropping the wide dtype.

### Evidence

- gelu tanh-approx port (2026-07-05, port_a3_to_a5 V220→arch35, A5 Ascend950PR / CANN 9.0.0): initial `TILE_ELEMS=8192, BUFFER_NUM=4` → fp32 320 KB > 256 KB AIV UB → 507035 on the first fp32 case, while bf16/fp16 (192 KB) PASSED — a prior "9/9 bf16" capture had been a false-complete that masked it. Fix `BUFFER_NUM 4→2, TILE_ELEMS 8192→4096` → fp32 96 KB, 87/87 PASS across all 3 dtypes (bf16 29/29, fp16 29/29, fp32 29/29). Kernel comment `gelu_kernel.h:26` documents the fp32-worst-case model.

### Other instances (predicted)

Any dtype-templated kernel that sizes UB from an element count and whose test matrix includes fp32: elementwise unary/binary, activation, cast-heavy pipelines, per-tile reductions.
