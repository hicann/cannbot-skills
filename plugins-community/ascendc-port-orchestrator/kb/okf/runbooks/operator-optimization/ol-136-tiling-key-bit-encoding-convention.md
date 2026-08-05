---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "TILING_KEY bit-encoding convention: pack orthogonal flags into one uint64, extend never bit-steal"
description: "Pack orthogonal small-cardinality flags into one uint64 TILING_KEY with one bit-position per axis; when porting V220→A5 add new flag axes on unused bit-positions and never re-use existing key values (silent ambiguity bug)."
original_id: OL-136
confidence: single_run
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-136, multi-variant-dispatch, tiling-key, port_a3_to_a5]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

When a kernel has multiple orthogonal flags affecting algorithm choice (e.g. `is_slice`,
`has_beta`, `is_bf16`, `is_fp16`, `apply_quant_offset`, `use_regbase_path`, ...), pack them
into a single `uint64` TILING_KEY where each bit-position corresponds to one orthogonal axis.
The dispatcher then uses `TILING_KEY_IS(N)` with N being the full integer value (sum of all
set bit-positions), not individual bit-checks.

When porting an op from V220 to A5 and adding new flag axes (typical case: A5 enables a
`regbase_path` bit that V220 didn't have), **extend the encoding orthogonally** — pick a new
bit-position not previously used. Do NOT re-use existing key values with different meanings on
A5 (silent ambiguity bug).

**Applies to** `soc=all; cann=9.0.0; bisheng=15.0.5; op_class=multi-variant-dispatch`. This
is a CODE-CONVENTION rule, not hardware-tied. Verified on Ascend950PR (rms_norm_quant.cpp
dispatcher) and Ascend910_V220 (same convention in the V220 master).

### Concrete anchor (from `rms_norm_quant.cpp` dispatcher)

```cpp
// Encoding convention (bit-position → flag):
//   bit 0 (1):   is_slice          (slice tiling vs whole-row)
//   bit 1 (2):   reserved (unused)
//   bit 2 (4):   reserved
//   bit 3 (8):   reserved
//   bit 4 (16):  reserved
//   bit 5 (32):  reserved
//   bit 6 (64):  has_beta          (rms_norm has bias term)
//   bit 7 (128): reserved (was apply_quant in earlier version)
//   bit 8 (256): is_bf16           (bfloat16 input dtype)
//   ...

if (TILING_KEY_IS(1))   { /* slice + no_beta + fp32 */ ... }
else if (TILING_KEY_IS(65))  { /* slice + has_beta + fp32 */ ... }
else if (TILING_KEY_IS(257)) { /* slice + no_beta + bf16 */ ... }
else if (TILING_KEY_IS(321)) { /* slice + has_beta + bf16 */ ... }
else if (TILING_KEY_IS(27))  { /* slice + ... + reserved-bit combo ... */ ... }
```

### Why this matters for A5 ports

1. **Encoding extension is safe**: adding a new bit (e.g. bit 9 for `regbase_path`) doesn't
   disturb existing TILING_KEY values used by V220 — the V220 dispatcher only checks the old
   N values; the A5 dispatcher checks the new combinations.
2. **Encoding bit-stealing is NOT safe**: never re-use an existing bit (e.g. flipping bit 7's
   semantics from "apply_quant" → "is_arch35") because the host-side tiling code that emits
   TILING_KEY may still set it under the old meaning.
3. **Host + kernel must agree**: the tiling-data emission code on the host side
   (`<op>_tiling.cpp`) and the dispatcher in `<op>.cpp` (or `<op>_apt.cpp` for A5) must use
   the same bit-position table.

Source: cann-learner CAND-A3A5-12, promoted 2026-05-12 (Mode 5 batch 2). C36 lift kept
op_class=normalization-quant conceptually but the convention generalizes to any op with
multiple orthogonal binary/small-cardinality flags.
