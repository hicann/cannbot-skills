---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "GM workspace contract for fused ops — public outputs stay separate; opaque scratch is one aligned byte workspace sliced by host offsets"
description: "applies_to: any soc with __gm__ pointer arithmetic; cann=9.0.0+; op_class=fused_with_aux_output derived-from: cann-source (FA-class workspace layout convention, 2026-05-09) unverified_on: a5_ops appli"
confidence: single_run
original_id: P-P89
timestamp_inferred: true
tags: [patterns-index, optimization, lse, mask, logsumexp, p-p89, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

`applies_to: any soc with __gm__ pointer arithmetic; cann=9.0.0+; op_class=fused_with_aux_output`
`derived-from: cann-source (FA-class workspace layout convention, 2026-05-09)`
`unverified_on: a5_ops`

`applies_to: any soc with __gm__ pointer arithmetic; cann=9.0.0+; op_class=fused_with_aux_output | multi_stage_fused`
`derived-from: cann-source (FA-class workspace layout convention, 2026-05-09)`
`a5_ops_anchor: 3_FusionAttention emits auxiliary public outputs (softmax_max/softmax_sum) as separate tensors; packed single-workspace scratch remains convention-level, not yet fully shipped in a5_ops`

**Trigger**: Fused op has (a) multiple cross-stage GM scratch tensors, such as matmul scratch, post-activation probabilities, partial accumulators, layout-conversion temporaries, or (b) auxiliary public outputs needed by later passes/backward restore, such as FA `lse`/softmax stats, fused-dropout `mask`, softmax+CE `logsumexp`.

**Recommendation**: Use a two-tier memory contract.

1. **Public outputs** are caller-visible tensors. Allocate and return them as normal torch/CANN outputs, pass each output pointer separately, and bind each to its own `AscendC::GlobalTensor<T>::SetGlobalBuffer(...)`. Public outputs must not be hidden inside opaque workspace, because tests, callers, autograd restore, and shape contracts need torch-visible tensors.

2. **Workspace scratch** is kernel-internal and opaque to the caller. Allocate one `params.workspace` byte buffer. Host tiling computes a packed layout and passes byte offsets in tilingdata. Every typed scratch offset must be alignment-padded:

```cpp
off_qk      = 0;
off_probs   = AlignUp(off_qk + qkBytes, 512);
off_partial = AlignUp(off_probs + probsBytes, 512);
totalBytes  = AlignUp(off_partial + partialBytes, 512);
```

Kernel entry slices the workspace via `__gm__ uint8_t*` pointer arithmetic:

```cpp
auto wsBase = reinterpret_cast<__gm__ uint8_t*>(params.workspace);

gQKScratch.SetGlobalBuffer(
    reinterpret_cast<__gm__ float*>(wsBase + tiling.offQk));
gProbsScratch.SetGlobalBuffer(
    reinterpret_cast<__gm__ half*>(wsBase + tiling.offProbs));
gPartialOut.SetGlobalBuffer(
    reinterpret_cast<__gm__ float*>(wsBase + tiling.offPartial));

// Public outputs stay separate:
gO.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(params.o));
gLse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.lse));
```

3. **Pybind contract**: pybind queries host tiling / `<op>_workspace_size(...)`, allocates `torch::empty({totalBytes}, kInt8, kNPU)`, and passes `workspace.data_ptr()` plus each public output’s `data_ptr()` to the launch. The workspace tensor must remain alive until the launched work using it has completed on the relevant stream. Do not assume “pybind return” equals kernel completion unless the launch path/allocator gives stream-ordered lifetime guarantees; otherwise synchronize or retain ownership through the stream work.

**Why it matters**:
- Reduces N scratch allocations to one device allocation.
- Keeps deterministic, shape-derived workspace sizing in host tiling.
- Preserves public-output visibility while avoiding scratch leakage into the Python API.
- Gives a uniform convention for multi-stage fused kernels and aux-output fused kernels.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 全文小节（P-P89，convert_patterns_to_okf.py）。confidence 未升格。 -->
