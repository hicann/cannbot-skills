---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "[ARCHIVED/DEPRECATED 2026-05-22] `DataCopy` GM->UB with non-zero `srcStride` for BSH->BNSD layout canonicalize on V220 produces wrong output [V220, layout-canonicalize, DataCopy-stride]"
description: "ARCHIVED 2026-05-22 — SUPERSEDED by CAND-FA-CANON-FREE. The original entry framed this as a V220 hardware/DataCopy bug + recommended a Python-side torch.reshape().permute().contiguous() workaround as"
phenomenon: build_failure
signal:
  - "Kernel function with internal AIV layout canonicalize (BSH -> BNSD via DataCopy with row-gather srcStride). Build clean. Runs to completion, no fault. Output ha"
confidence: single_run
original_id: PB-36
timestamp_inferred: true
tags: [datacopy, srcstride, aclnnflashattentionscorev2, ascendc, pb-36]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

**ARCHIVED 2026-05-22 — SUPERSEDED by CAND-FA-CANON-FREE.** The original entry framed this as a V220 hardware/DataCopy bug + recommended a Python-side `torch.reshape().permute().contiguous()` workaround as the "Preferred pattern." Both framings were wrong:

1. **CANN's own `aclnnFlashAttentionScoreV2`** runs on the same V220 / CANN 9.0.0 hardware and produces correct output for BSH/SBH/BSND/BNSD without any such workaround — verified directly on A3 (exit=0, math correct, output 256.0). If this were a hardware bug, CANN's FA would fail too. It doesn't, so the bug was in our kernel's design choice (using an AIV canon stage at all), not in the underlying hardware.
2. **The recommended "Python-side reshape" workaround was a No-Delegation rule violation**: `torch.reshape().permute().contiguous()` on a `.npu()` tensor dispatches to torch_npu → CANN `aclnnPermute` / `aclnnContiguous`. The 0.45× CANN perf claim it produced was a misleading hybrid pipeline benchmark, not a kernel-vs-kernel comparison.
3. **The structural fix** (CAND-FA-CANON-FREE) eliminates the AIV canon stage entirely: mm1/mm2 read strided source GM directly via `MatmulImpl::SetOrgShape` 5-arg variant with `orgN/orgKa/orgKb = sS` (physical row stride), matching CANN's own implementation. Hits 0.603× CANN on the same shape, pure AscendC, no delegation.

**Audit trail (per safety rule 5: keep deprecated entries for history; do not delete)**: the original body is preserved below for context. Do NOT cite PB-36 as a current pattern; cite CAND-FA-CANON-FREE instead. Cross-ref: `docs/design/FA_CLASS_DESIGN_NOTES.md#fa-canon-removal-structural-rewrite` for the structural rewrite design + verification numbers.

**Original (now-archived) body:**

`applies_to: soc=Ascend910_9382 (V220 A2/A3); cann=9.0.0; op_class=fp16 layout canonicalize via DataCopy with non-zero srcStride for strided GM gather (BSH/SBH/BSND -> BNSD)`
`verified_on: a5_ops:3_FusionAttention kw-5 (2026-05-22, A3 npu-a3-test) — FaCanonicalizeQKVKernel::CopyOne with DataCopyParams{blockCount=S, blockLen=D*sizeof(T)/32, srcStride=(H-D)*sizeof(T)/32, dstStride=0} produces wrong output on BSH path (max_abs=0.055 vs CANN aclnn ground-truth) while BNSD path (srcStride=0) is correct (max_abs=3e-5). Falsification chain rules out infrastructure causes: (1) CANN's own aclnnFlashAttentionScoreV2 exit=0 on A3 (proves CANN runtime + driver OK); (2) torch_npu.npu_fusion_attention PASS on A3 (proves Python-aclnn bridge OK); (3) 1_BatchMatmul AscendC build PASS max_abs=1.5e-5 on A3 with default config + isTransposeB=true + 64x64x64 sub-tile (proves NPUKernelBench build pipeline + cube path infrastructure OK); (4) FA BNSD path with canon-skipped passes; (5) FA BSH path with canon-active fails. Bug is therefore confined to this kernel's strided-DataCopy path.`
`unverified_on: V351 (Ascend950PR / A5) — peer A5 agent reports same Python-side BSH<->BNSD workaround on case_3 [4,64,512] head=8 fp16 gives max_abs=1.54e-2 (above T1=1e-2); separate cross-arch precision divergence (500x worse than A3 with same workaround), root cause TBD. Open question: does the canon stride bug exist on V351 too, OR is canon correct on V351 but the V351 cube path itself has a different precision issue?`

- **Severity**: HIGH (silent wrong output; verification can miss without ground-truth comparison since output is plausibly-scaled noise; canon AIV stage fires on EVERY layout != BNSD, which is the common case for FA users — `BSH`, `SBH`, `BSND` all transit through this code path)
- **Symptom**: Kernel function with internal AIV layout canonicalize (BSH -> BNSD via `DataCopy` with row-gather `srcStride`). Build clean. Runs to completion, no fault. Output has structurally correct shape, but values diverge from ground truth at ~5e-2 level (full output scale, above T1=1e-2). Direct probe of canon output: head 0 produces all-rows-identical output (canon reads same source row repeatedly); head 1 has zero rows at scattered `s` positions.
- **Root cause hypothesis** (unconfirmed; KB carries the EMPIRICAL fact + workaround): MTE2 engine on V220 CANN 9.0.0 mishandles `srcStride` field semantic for certain GM->UB `(blockCount, blockLen, srcStride)` tuples — either ignored (-> same row read N times) or misinterpreted (-> skipped rows). Possibly related to PB-22 (V220 MTE2 DataCopy 32B transfer limit per destination TBuf) or PB-9 (V220 UB->UB DataCopy nuances). Pinning the exact failure condition (which `srcStride` values trip the bug at which `blockCount`/`blockLen` combos) is a follow-up probe job — not a blocker for the workaround.
- **Fix / workaround**: Move layout canonicalization to the Python side (BSH -> BNSD via `torch.reshape().permute(0,2,1,3).contiguous()`), pass already-BNSD tensors to the kernel, kernel skips canon. After kernel returns, transpose output back. Costs one extra alloc + copy on Python side. Measured at 0.45x CANN on S=64, D=64, N=2 BSH fp16 (181 us vs CANN 82 us) — 10x improvement over kw-1 VEC fallback baseline (0.046x CANN), still below 0.6x target but a real, mergeable improvement. PERMANENT in-kernel fix (future-kw scope) options: (a) replace strided `DataCopy` with row-by-row scalar loop, (b) try `DataCopyExtParams` (V220 extended variant) — needs probe, (c) move canon to AIC side with `Nd2NzParams` (different DMA engine path).
- **Anti-pattern (DO NOT EMIT on V220)**:
  ```cpp
  // BAD on V220: GM->UB DataCopy with non-zero srcStride from BSH layout
  DataCopyParams p;
  p.blockCount = S;                              // gather S rows
  p.blockLen   = (D * sizeof(T)) / 32;           // each row D fp16 elements
  p.srcStride  = ((H - D) * sizeof(T)) / 32;     // non-zero (=4 for H=128,D=64,fp16) -> wrong output
  p.dstStride  = 0;
  DataCopy(ubBuf, gmBSH[off], p);                // silently broken on V220
  ```
- **Preferred pattern** (Python-side layout transpose + kernel sees only BNSD):
  ```python
  # Python (model_new_ascendc.py forward path):
  if layout == "BSH":
      B, S, H = query.shape
      D = H // head_num
      N = head_num
      q_bnsd = query.reshape(B, S, N, D).permute(0, 2, 1, 3).contiguous()
      # k, v same; pass q_bnsd / k_bnsd / v_bnsd to kernel with kern_layout="BNSD"
      attn_out_bnsd = _ext.run_fusion_attention(q_bnsd, k_bnsd, v_bnsd, N, "BNSD", scale)
      attention_out = attn_out_bnsd.permute(0, 2, 1, 3).contiguous().reshape(B, S, H)
  # Mirror handling for SBH / BSND. Kernel always sees BNSD; canon AIV stage is no-op.
  ```
  ```cpp
  // GOOD: contiguous BNSD DataCopy in kernel (no strided gather needed)
  DataCopyParams p;
  p.blockCount = 1;
  p.blockLen   = (S * D * sizeof(T)) / 32;
  p.srcStride  = 0;
  p.dstStride  = 0;
  DataCopy(ubBuf, gmBNSD[off], p);
  ```
- **Evidence**:
  - 3_FusionAttention kw-5 cycle 6 (2026-05-22T03:50Z, A3 npu-a3-test): Direct probe (`fa_canon_dbg.py`) on BSH input revealed head0 all-rows-identical output, head1 zero rows at s in {8,16,24,32,40} — the canon stage was reading source row 0 for every destination row of head 0, and skipping source rows entirely for head 1. BNSD-direct path on same numerical input (no canon) showed max_abs=3e-5 (T1 PASS).
  - Falsification chain (lets us localize the bug to *this* kernel's `srcStride` use and not infrastructure): `/tmp/test_fa.cpp` = CANN's own `test_aclnn_flash_attention_score.cpp` compiled on A3, ran exit=0 with correct math (output value 256.0). `/tmp/bmm_test/` = 1_BatchMatmul AscendC archive copied to A3 + rebuilt, PASS max_abs=1.5e-5 with default config; PASS max_abs=0.0002 with isTransposeB=true; PASS max_abs=0.0001 with 64x64x64 sub-tile. Both proved CANN runtime + NPUKernelBench infra are NOT the problem.
  - Owner mid-session correction (2026-05-22): pushed back on initial "V220 MIX dispatch broken" / "V351->V220 cross-arch divergence" / "CANN version bug" framings — drove the CANN-aclnn + minimal-pybind11 probes that produced the falsification chain above. All three prior framings empirically falsified.
  - After workaround applied: FA cube path PASS_T1 with max_abs=3.05e-5, MARE=2e-4. Pass B 9/9 PASS preserved. Perf 0.45x CANN on B=1, S=64, N=2, D=64 BSH fp16 (181 us vs CANN 82 us) — 10x over kw-1 VEC fallback baseline (0.046x CANN), still below 0.6x target.
  - Peer cross-arch datapoint (A5 V351, 2026-05-22 from agent-main): same Python-side reshape workaround on case_3 [4,64,512] head=8 fp16 gives max_abs=1.54e-2 (above T1=1e-2) at ~1 ms. Cross-arch divergence at 500x max_abs — root cause separate from canon bug. Open hypothesis: V351 cube path itself has precision regression (possibly fp16 accumulator behavior, or unverified cube_eligible code path that canon was previously hiding).
- **Cross-reference**: PB-9 (V220 UB->UB DataCopy -> `Adds` identity workaround; same V220 MTE-engine family of nuances but different direction); PB-22 (V220 MTE2 DataCopy 32B transfer limit per destination TBuf); related in-tree code: `workspace/3_FusionAttention/kernel/fusion_attention_kernel.h::FaCanonicalizeQKVKernel::CopyOne` (line ~747); `workspace/3_FusionAttention/kernel/fusion_attention_kernel.h::FaUncanonicalizeKernel` (mirror operation — `DataCopy` with `dstStride` to write BNSD output back as BSH; likely same bug, currently also bypassed by Python output-reshape) — follow-up CAND needed to confirm.

<!-- 迁移自 porter kb/target/ascendc/（PB-36，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
