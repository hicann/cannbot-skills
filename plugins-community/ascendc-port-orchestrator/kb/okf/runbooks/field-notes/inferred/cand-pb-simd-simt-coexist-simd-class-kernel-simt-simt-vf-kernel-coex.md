---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "SIMD class kernel + SIMT `__simt_vf__` kernel coexistence in same `.so` corrupts SIMT path's precision (PB candidate, 1-op evidence)"
description: "Severity: HIGH (would block any Kind-2 SIMD rewrite of multi-path SIMT operators) Source: 26_AvgPool3d kw-2 (2026-05-03 Ascend950PR_9579, CANN 9.0.0) — 8 SIMD UB-tile rewrite iterations + 1 revert. Va"
phenomenon: build_failure
signal:
  - "when a kernel module emits BOTH a SIMD class kernel (Init/Process pattern with TPipe + TQue + TBuf) AND a SIMT __simt_vf__ kernel in the same .so, the SIMT path"
confidence: inferred
status: stub
original_id: CAND-PB-SIMD-SIMT-COEXIST
timestamp_inferred: true
tags: [candidate, inferred, __simt_vf__, _op_simd.so, _op_simt.so, cand-pb-simd-simt-coexist]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**Severity**: HIGH (would block any Kind-2 SIMD rewrite of multi-path SIMT operators)
**Source**: 26_AvgPool3d kw-2 (2026-05-03 Ascend950PR_9579, CANN 9.0.0) — 8 SIMD UB-tile rewrite iterations + 1 revert.
**Validation status**: 1 op evidence (SIMT regression observed); hypothesis plausible but unverified; no minimal repro on a 2nd op yet.

**Symptom**: when a kernel module emits BOTH a SIMD class kernel (Init/Process pattern with `TPipe + TQue + TBuf`) AND a SIMT `__simt_vf__` kernel in the same `.so`, the SIMT path may produce subtly wrong results — including for SIMT cases that worked correctly when the SIMD class was absent.

**Specific evidence (op#26 AvgPool3d kw-2)**:

The SIMT generic path was a verbatim copy of kw-1's `avgpool3d_vf<T>` (renamed to `avgpool3d_simt_fallback_vf<T>` for kw-2's mixed build). In kw-1 (SIMT-only build) it passed 72/72 cases; in kw-2 (SIMD class + SIMT in the same `.so`) the same SIMT code FAILED on 4-7 cases per dtype:

| Dtype | kw-1 SIMT-only | kw-2 SIMD+SIMT |
|---|---|---|
| fp32 | 52/52 PASS | 42-44/52 PASS |
| fp16 | 10/10 PASS | 5-6/10 PASS |
| bf16 | 10/10 PASS | 5-6/10 PASS |

Failing cases mix BOTH fast-path-routed (SIMD class) AND generic-routed (SIMT) — the SIMT regression cannot be explained by a SIMD class bug alone.

**Plausible mechanism (not yet confirmed)**:
- `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` set in both SIMD class entry-points AND SIMT entry-points
- SIMD class instantiates `TPipe + TQue + TBuf` state that may persist across kernel boundaries
- SIMT path's grid-stride loop with `__gm__ T*` scalar reads may pick up stale event/queue state from prior SIMD invocation
- OR mixed `__global__ __aicore__` kernels with `Simt::VF_CALL<...>` may force compiler to a different code-gen mode for the SIMT TU

**Mitigation strategies (proposed, not validated)**:
1. **One-pattern-per-.so rule**: either all-SIMT or all-SIMD class. For multi-path kernels (fast + generic), implement BOTH paths via the same pattern (generic path also uses Init/Process class with conditional logic in `Process()`).
2. **Separate .so per path**: build SIMD class in `_op_simd.so`, SIMT in `_op_simt.so`, dispatch at Python pybind level.
3. **Debug-instrument the SIMT regression**: write canary GM values at start/end of SIMT kernel to confirm whether SIMT is being called at all or its state is corrupted by prior SIMD invocation.

**Workaround applied (op 26)**: reverted to kw-1 SIMT-only baseline (72/72 PASS, 0.14× perf). Structural-ceiling exit at 0.14× — ko-1's 3-axis ablation already confirmed HBM scalar-load latency is the SIMT bottleneck; the architectural shift to SIMD class is blocked by this coexistence issue.

**Promotion criteria** (CAND → PB-N):
1. Reproduce on a second op where mixed SIMD class + SIMT `__simt_vf__` build exhibits SIMT regression vs SIMT-only build
2. Minimal repro: 2-kernel `.so` with one trivial SIMD class kernel and one SIMT scalar kernel reading from independent GM buffers — does the SIMT kernel produce different output when SIMD class is linked?
3. Identify exact compiler/linker code-gen difference between mixed and SIMT-only builds (`bisheng -S` or equivalent)
4. Confirm with CANN team if this is intentional ABI vs unintended pipeline-state leak

**Related**:
- PB-9 (DataCopy localDst/localSrc silent corruption — same family: pipeline-state correctness gotchas)
- OL-63 (TQue VECIN depth — same family: pipeline configuration affects code-gen)
- A3 op#13_Cat finding 2026-04-25 (KERNEL_TASK_TYPE_DEFAULT macro affects code-gen path — sibling concern)
- aog-self-critic C5 (no premature platform-blame — must reproduce on 2nd op before promoting to PB-N)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PB-SIMD-SIMT-COEXIST，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
