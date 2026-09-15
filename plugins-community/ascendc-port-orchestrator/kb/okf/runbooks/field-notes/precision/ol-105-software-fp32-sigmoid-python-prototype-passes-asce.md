---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Software fp32 sigmoid Python prototype passes; AscendC SIMD port has bit-manipulation lowering bugs — debug in standalone test kernel before integrating"
description: "paradigm: ascendc"
phenomenon: precision_issue
signal:
  - "implementing OL-103 software fp32 transcendental in a real op kernel"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-105
timestamp_inferred: true
tags: [ascendc, ol-105]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — derived from op#2 SwiGLU sw-sigmoid integration attempt 2026-04-29.
- **Category**: precision / process
- **Loaded by**: developer attempting to port a Tier-1 software algorithm (sigmoid, exp, log, etc.) from Python prototype to AscendC SIMD
- **Trigger**: implementing OL-103 software fp32 transcendental in a real op kernel

### Lesson

The Tier-1 software fp32 sigmoid algorithm (per OL-103) was prototyped in Python (`/tmp/sw_sigmoid_python.py`) and verified to give **max rel err 1.25e-7** vs `torch.sigmoid` on 37 test points. Mathematically correct, passes fp32 MERE 2⁻¹³ comfortably.

Direct port to AscendC SIMD as `sw_sigmoid::SoftwareSigmoidFp32` integrated into op#2 SwiGLU's FusedSwiGLUF16 path **REGRESSED from 46/50 to 38/50** with MERE=0.999 on most fp16 cases — outputs were essentially garbage for affected cases. This indicates an implementation bug, NOT an algorithmic flaw.

### Likely failure modes (debugging hypotheses, not yet bisected)

1. **`And` / `Or` on `LocalTensor<int32_t>`**: API may require specific masks layout / count alignment that the prototype didn't verify.
2. **`ReinterpretCast` from fp32 LocalTensor to int32 LocalTensor**: behavior may differ from C++ `*reinterpret_cast<int32_t*>(&fp32_var)`.
3. **`ShiftLeft` / `ShiftRight` on `LocalTensor<int32_t>`**: may zero high bits / not be available / require different operand layout.
4. **Buffer aliasing**: using same physical buffer (`scratchA`, `scratchB`) for fp32 view AND int32 ReinterpretCast view — AscendC may not guarantee correctness for in-place dtype reinterpret across same buffer (per `dequant_kernel_patterns.md` §7.1: "AscendC 不保证支持 src/dst 是同一物理 buffer 但不同元素类型 的 Cast").
5. **PipeBarrier placement**: insufficient barriers between bit ops and arithmetic ops in different pipes.

### Action for next attempt

DO NOT integrate sw_sigmoid into a production kernel until verified standalone. Required workflow:

1. **Build a test kernel** that takes one input tensor `[N]` of fp32 and outputs `sigmoid(x)`.
2. **Compare element-wise** against PyTorch CPU on a small tile (32–256 elements).
3. **Bisect by step**: clamp → exp(neg_x) split into sub-steps → 1/d split into sub-steps. At each split, compare intermediate against Python prototype.
4. **Identify which step diverges** in AscendC vs Python.
5. Fix that step in `software_sigmoid_fp32.h`.
6. **Only then** integrate into FusedSwiGLUF16 / similar production kernels.

This methodology is the same as the aog-precision-probe workflow — just applied to a standalone helper instead of a full op.

### Anti-pattern (what I did)

Wrote ~150 lines of complex bit-manipulation, integrated directly into production kernel, hoped it works. When it didn't, no diagnostic feedback. Wasted ~30 min build/test cycle and ended up with same 46/50 baseline + a broken header.

### Self-critic

1. ✅ Empirical foundation: regression measured (46→38). Honest about failure.
2. ✅ Achievable bar: Python prototype proves algorithm is correct. The bug is in the SIMD port, not the algorithm.
3. ✅ Reward-hack scan: NOT claiming success. Explicit "not yet debugged".
4. ✅ Sealed reproducibility: header preserved at `2_SwiGLU/software_sigmoid_fp32.h`, regression numbers recorded.
5. ✅ Generalization scope: lesson applies broadly — any complex algo from Python proto needs intermediate-step verification before kernel integration.

### Cross-reference

- OL-103 (the sw fp32 sigmoid spec) — algorithm correct
- This OL-105 — the SIMD lowering needs debugging before reuse
- `software_sigmoid_fp32.h` is preserved for future debugging session

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-105（category=precision / process，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
