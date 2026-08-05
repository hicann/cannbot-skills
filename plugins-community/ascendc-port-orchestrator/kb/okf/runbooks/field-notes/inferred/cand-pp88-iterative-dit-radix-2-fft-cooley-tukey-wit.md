---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Iterative DIT radix-2 FFT (Cooley-Tukey) with hybrid VEC/scalar butterfly path"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5+2026-04-13; op_class=spectral-transform (FFT/IFFT/DCT family) verified_on: soc=Ascend950PR; cann=9.0.0 unverified_on: soc=Ascend910_9382 (V220 f"
phenomenon: build_failure
signal:
  - "reference op is torch.fft.rfft / torch.fft.fft with PoT length, OR direct DFT O(N²) hits perf ceiling on large N, AND fp32 precision must match torch.fft (catas"
confidence: inferred
status: stub
original_id: CAND-PP88
timestamp_inferred: true
tags: [candidate, inferred, torch.fft.rfft, torch.fft.fft, tmp_a, tb_im, log2n, cand-pp88]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5+2026-04-13; op_class=spectral-transform (FFT/IFFT/DCT family)`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_9382 (V220 family — twiddle build via Sin/Cos<RADIAN_REDUCTION> primitive precision tier may differ; needs A3 probe)`

**Trigger**: reference op is `torch.fft.rfft` / `torch.fft.fft` with PoT length, OR direct DFT O(N²) hits perf ceiling on large N, AND fp32 precision must match torch.fft (catastrophic-cancellation cases).

**Pattern (per-row, single-AIV-owns-row determinism)**:
1. Zero-pad: `real_buf[0..seqlen) ← x[row]`, `real_buf[seqlen..N) = 0`, `imag_buf[0..N) = 0`.
2. Bit-reversal permutation in-place (scalar SetValue/GetValue with V→S/S→V sync).
3. For stage `s` in `[1..log2N]`: `m = 1 << s; half = m >> 1`; build per-stage twiddle via `ArithProgression + Muls + Sin<RADIAN_REDUCTION> + Cos<RADIAN_REDUCTION>`; dispatch:
   - `half < 8` → scalar butterfly path (avoids EC-26 unaligned-VEC violation since `m=2,4,8` violates 32 B alignment for fp32).
   - `half ≥ 8` → VEC butterfly path (Mul/Sub/Add; `tmp_a` aliased as `tb_im` to save one scratch buffer).
4. Per-row normalize: `Muls(real_buf, real_buf, 1/N, N); Muls(imag_buf, imag_buf, 1/N, N)`.
5. `DataCopy(GM, real_buf, AlignUp8(kcount))` with V→MTE3 sync; same for imag.

**Per-stage twiddle rebuild (NOT recurrence)**: each stage's twiddle is independently computed from `theta = ArithProgression × (-2π/m)` → no cross-stage error propagation. Per-stage cost: `half × Sin + half × Cos`. Total across `log2N` stages: ~N twiddle ops, dwarfed by `6N log2N` butterfly ops.

**UB layout (worst case N=8192, seqlen=4096, fp32)**: input TQue (16 KB) + real/imag bufs (32+32 KB) + twiddle_cos/sin (16+16 KB) + tmp_a/tmp_b/tb_re scratch (16×3 KB) + theta/idx (16+16 KB) = **176 KB** (fits 192 KB UB).

**Hybrid dispatch for non-PoT seqlen**: cheap runtime check `is_pot = (seqlen > 0) && ((seqlen & (seqlen-1)) == 0)` — PoT path → radix-2 FFT, non-PoT path → direct DFT fallback. Dispatch overhead negligible.

**Determinism**: by-construction satisfied (1 row → 1 core, fixed-order stages, deterministic bit-reversal, deterministic twiddle compute, no atomicAdd, no shared GM writes).

**Performance**: op#23 23_HyenaFftSizePaddingRfft kw-3 (2026-04-26): Pass A 49/49, det 49/49, perf ratio_median 1.17× (vs kw-2 direct-DFT 0.596× — ~2× improvement on PoT-dominated benchmark). Pass B 12/14: 2 large_mag fp32-ULP-limit cases (residual ~ N · ULP · |max_input|; classified as an fp32 algorithmic limit, not a kernel bug).

**Anti-pattern avoided**: case-specific predicates / ε nudges to mask adversarial-magnitude failures (OL-85 violation). Adversarial-magnitude residual is fp32 unit-ULP × |input|_max; classified as an fp32 algorithmic limit, not patched.

**Promote when**: a second spectral-transform op (FFT-derived: convolution-via-FFT, IFFT, real-FFT-of-2N-trick) confirms the same Cooley-Tukey + scalar-stage-cutoff template. Likely to live in a future `patterns/domains/spectral.md` (KB has zero spectral content today; gap noted as DEBT-052).

**Source**: op#23 23_HyenaFftSizePaddingRfft kw-3 (2026-04-26). 1-op evidence; KB had no spectral pattern at start of session — directive itself encoded the missing knowledge.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP88，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
