---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "port_a3 V220-pure — emit ALL host-dispatched algorithm variants and replicate the dispatch rule; single-variant ports leak shape-dependent sub-LSB diffs that mimic hw-floor"
description: "V220-pure port_a3: when upstream host-tiling selects one of N sibling kernel headers, emit ALL variants and replicate the dispatch host-side; a single-variant port leaks 1-LSB shape-dependent diffs."
phenomenon: precision_issue
signal:
  - "a port_a3 op passes most shapes bit-exact but a small fraction show 1-LSB off-by-one diffs at single elements that look like a hardware floor"
confidence: single_run
original_id: OL-187
classified_by: llm-assisted
timestamp_inferred: true
tags: [precision, port-a3, ol-187, tiling-dispatch, anti-cheat]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发
A V220-pure plain-entry port_a3 op whose first iteration emitted only the "obvious" variant (e.g. NORMAL) under all tiling keys. Most shapes are bit-exact, but a small fraction show 1-LSB off-by-one diffs at single elements — easily mistaken for a hardware floor.

**Trigger classifier (load this lesson when):**
- The op is a V220-pure plain-entry port (`.upstream_prestaged.json` mode=`upstream_v220_entry`, NOT arch35 variant).
- Upstream `op_kernel/<op>.cpp` `extern "C" __global__ __aicore__` body — OR an `op_host/<op>_tiling.cpp` `RunBigKernel`-style dispatcher — selects among sibling headers (`<op>.h` / `<op>_single_n.h` / `<op>_split_d.h` / ...) via `TILING_KEY_IS(...)`, `blockFactor==1`, `numCol>threshold`, or similar host-knowable conditions.
- The port ships only ONE variant under all tiling-key values.

## 根因 / 教训
When the A3 truth reference is `torch_npu.<op>(...)`, the A3 wrapper transparently dispatches across the V220 algorithm variants per upstream's host-side rule. **Each variant computes a numerically-different intermediate chain** even though all are "the same algorithm" by spec — typical divergences: bf16 cast-trips in one variant but not another; fp32-accumulator-held vs cast-back-to-bf16 between row-reduce and elementwise stages; in-place vs ping-pong buffer ordering with different rounding order. These are sub-LSB on most shapes but cross int8 quant-cell boundaries on a small fraction of inputs and surface as **1-LSB off-by-one diffs at single elements**.

**The A5 port must therefore:**
1. **Emit ALL kernel headers** the upstream dispatcher can pick, not just the "main" one.
2. **Replicate the dispatch decision host-side** (pybind11 wrapper or `kernels.cpp` top level) using the same input→tiling-key derivation upstream uses.
3. Build extern-C entry points for the full `{variant × dtype}` cross-product so the host can launch any selection.

**Why specific to migration**: a backward CPU oracle commonly makes one explicit algorithmic choice, while a selected source-arch entry may silently dispatch a different variant by shape. Migration must preserve that source dispatch matrix.

Verified on soc=Ascend950PR_9579 (V351), cann=9.0.0, bisheng=15.0.5: `add_rms_norm_quant` (2026-05-24 kw-1) — NORMAL + SingleN dispatch by `blockFactor==1`; 7/7 T1 bit-exact bf16+fp16 once both variants emitted, 1.66× perf vs A3 baseline. (A3 is the truth reference and trivially passes by definition; the lesson is about the port direction.)
