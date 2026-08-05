# PROBE REPORT — Q2 UB bank count / width / conflict behavior

## Verdict
INCONCLUSIVE_PARTIAL

**One-line summary**: The stride-sweep methodology successfully revealed a
**dense-sequential fast path** (stride=1 at 12.26 cyc/repeat) vs a **strided
slow zone** (14.7–15.5 cyc/repeat, +20–27%). But the observed spread is too
compressed to nail down bank count K or per-bank width W directly. No
catastrophic K× cliff was observed at any stride up to 32, which **rules out
a naïve K=8 banks × 32B interleave** (that would predict ~8× penalty at S=8)
but does not rule out a more forgiving bank network with ≥ 2 R-ports per bank
group (consistent with the 351x public doc claim of "2R+2W per bank group").

## Environment
- date: 2026-04-21 (UTC)
- host: 198.51.100.35 → container `npu_dev3`
- NPU: `/dev/davinci1` (NPU 1, Ascend950PR, idle pre+post, AICore 0%, no
  foreign process; NPU 0 busy with op#11 pytest, avoided)
- CANN: `Ascend-cann-toolkit 9.0.0 V100R001C10SPC001B218`
- bisheng: `clang 15.0.5 (clang-5c68a1cb1231 ...)` build stamp `2026-03-21T17:07:34+08:00`
- SOC target at build: `-DSOC_VERSION=Ascend950PR_9589`
- AIV frequency assumed for cycle derivation: **1.8 GHz** (same assumption as Q5)
- Python env: torch + torch_npu (requires `setenv.bash` to load libhccl.so)

## Question
How many banks does the 256KB UB have, what's the per-bank width, and what
stride/alignment avoids bank conflicts?

## Hypothesis vs Observation
- **Hypothesis**: common options are 8 or 16 banks × 32B or 64B, interleaved
  at native-load granularity (likely 32B × 8 banks on a 256-byte VEC register
  chip).
- **Observation**: Stride-sweep of `Mul<float>(mask=64, repeat=8, src0BlkStride=S)`:
  - S=1 (dense): **12.26 cyc/repeat** — baseline / fast path
  - S=2,3,5,6,7: **14.7–15.0 cyc/repeat** (+20–22%)
  - S=4,8,16: **15.5 cyc/repeat** (+27%) — mild upper plateau
  - S=32: 15.3 cyc/repeat — no further degradation
  - **No catastrophic cliff** anywhere. The dynamic range between best and
    worst non-unit stride is only ~6%, which is too compressed to cleanly
    back-solve bank count.

## Evidence

### Results (REPEAT_N = 4096 Mul calls, INNER_REPEATS = 8 per call, 1.8 GHz)

| src0BlkStride | median µs | min µs | stdev | cyc/call | cyc/repeat |
|---:|---:|---:|---:|---:|---:|
| 1  | 223.25 | 222.85 | 0.49 | 98.1  | **12.26** |
| 2  | 272.90 | 272.24 | 0.41 | 119.9 | 14.99 |
| 3  | 268.26 | 267.49 | 0.91 | 117.9 | 14.74 |
| 4  | 282.89 | 282.46 | 0.38 | 124.3 | **15.54** |
| 5  | 268.03 | 267.44 | 0.37 | 117.8 | 14.72 |
| 6  | 272.81 | 272.06 | 0.40 | 119.9 | 14.99 |
| 7  | 268.15 | 267.48 | 0.33 | 117.8 | 14.73 |
| 8  | 282.84 | 282.36 | 0.41 | 124.3 | **15.54** |
| 16 | 282.75 | 282.41 | 0.25 | 124.3 | 15.53 |
| 32 | 278.08 | 277.37 | 0.38 | 122.2 | 15.28 |

### Interpretation / Inference attempts

**Clear signal #1 — dense is 22% faster than any non-unit stride.**
S=1 is the only stride where the 8 × 32B blocks of a mask-64 repeat land in
strictly consecutive UB addresses. The ~22% penalty for any S≥2 says the vector
unit *can* burst-read 256B contiguous in a single cycle but must serialize /
arbitrate on any scatter. Practical guidance: **contiguous UB layout is a
first-order performance invariant**, bigger than any secondary bank effect.

**Clear signal #2 — mod-4 parity in the even strides.**
S ∈ {4, 8, 16} all sit at 15.53 cyc. S ∈ {2, 6} at 14.99 cyc. S=32 slightly
lower (15.28). Odd strides all at ~14.73. This is consistent with some
bank-group alignment where access patterns that are multiples of 4 blocks
(128B) align all 8 accesses to 2 bank groups, while non-4-multiples spread
across 4–8 bank groups, slightly reducing the per-group contention. The
**"conflict" is only ~6% between the aligned-bad (S=4,8,16) and misaligned-
better (S=3,5,7)** cases, far below a naïve "K-way conflict = K× slowdown".

**What we can rule out**:
- **K=8 banks × 32B interleave, single-port** — would predict ~8× slowdown
  at S=8. Observed: 1.27×. Rejected.
- **K=16 banks × 32B interleave, single-port** — same style reasoning. Rejected.
- **Any single-port bank layout** where stride-S collision triggers K-way
  serialization is inconsistent with the 6% spread.

**What is consistent with observation**:
- Multi-ported bank groups (the public doc's "2R+2W per group" statement).
  With 2 read ports per group, a 2-way conflict is absorbed for free; only
  a 3-way or worse conflict costs cycles, and even then the port structure
  throttles rather than stalls. That matches a small ~20% plateau rather
  than a cliff.
- The fast path (S=1) exploits a separate "full-width burst" mode of the
  vector register-file that bypasses bank arbitration entirely.

**What we CANNOT determine from this probe**:
- Exact bank count K. Our stride grid (1,2,3,4,5,6,7,8,16,32) does not
  produce a distinguishable fingerprint for K ∈ {8, 16, 32}.
- Per-bank width W (32B vs 64B). Our smallest access granularity was 32B
  (one block); distinguishing 32B vs 64B banks would require sub-block
  strided tests that AscendC BinaryRepeatParams doesn't expose.
- Read-vs-write conflict behavior. We only probed RR (two src reads into
  one dst write); didn't isolate WW or RW conflicts.

### Build
Clean build succeeded (after an initial env setup hurdle: `libhccl.so`
requires `source /usr/local/Ascend/cann-9.0.0/x86_64-linux/bin/setenv.bash`
before invoking the cmake-based build, otherwise TORCH_PATH resolves to
empty string and `<torch/extension.h>` cannot be found). One cosmetic warning:

```
[WARNING]: Multiple kernel functions are detected. It is recommended to define
only one kernel function per file.
```
(10 stride-variant `__global__` kernels share `probe_kernels.cpp` — does not
affect correctness.)

### Runtime
Single run produced stable timings across all 10 stride variants (stdev <
1 µs on medians of 223–283 µs, i.e. < 0.4% relative). Discard-first-5
methodology from Q5 carried over cleanly.

### Raw files
- Kernel: `workspace/probe_ub_bank_count/kernel/{probe_kernels.cpp,probe_kernel.h,probe_tiling.h,pybind11.cpp}`
- Build log: `workspace/probe_ub_bank_count/build.log`
- Run log: `workspace/probe_ub_bank_count/run.log`
- Timings: `workspace/probe_ub_bank_count/timings.csv`

## Practical recommendations for orchestrator / future ops

- **Rule of thumb for UB access**: keep the primary access dense-sequential
  (src0BlkStride=1 / src1BlkStride=1). Any strided access will cost ~22%
  regardless of stride value — there is **no magic "safe" stride** that
  matches the dense-sequential speed.
- **Don't worry about specific stride-K choices for bank avoidance.** The
  worst observed penalty (S=4,8,16) is +27% over baseline, only +6% over
  the best non-unit stride. Fine-tuning stride for bank structure yields
  diminishing returns.
- **If a kernel already uses stride≥2**, pick odd strides (3, 5, 7) if free
  — they measure ~5% faster than even-power strides (4, 8, 16). Not worth
  reshaping the kernel over, but worth knowing.
- **For aggressive optimization**, prefer restructuring to make UB access
  fully contiguous (e.g. transpose-via-DataCopy before VEC, rather than
  VEC-with-stride). The 22% dense→strided cliff is the real prize.

## Caveats
- The stride grid is sparse — {9,10,11,12,14,24} untested; a full power-of-2
  + nearby-offset sweep (e.g. 9,17,33) might reveal finer structure.
- Methodology measures one Mul with src0 strided and src1+dst dense. A
  symmetric two-sided stride probe (both src0 and src1 strided) might produce
  cleaner conflict signatures.
- Single NPU, single bisheng version; result may shift with compiler
  updates that change VEC scheduling.
- Frequency assumption (1.8 GHz) affects absolute cycle numbers but not
  relative cycle ratios (which are what matters for conflict analysis).
- **This verdict is indirect inference only.** Without access to an msprof
  `aiv_ub_bank_conflict` counter (checked via `msprof --help` — not available
  in CANN 9.0.0 user-facing interface), the true bank count cannot be
  definitively stated from timing data alone.

## Suggested next-step probes (if budget permits)

1. **Write-conflict probe**: two concurrent VEC stores with stride, to
   isolate WW conflicts from RR conflicts.
2. **Sub-block probe**: use `Mul` with `dstBlkStride` and `src0BlkStride`
   differing at sub-block level via element-offset tricks — may expose
   32B vs 64B per-bank width.
3. **msprof counter enumeration**: run `msprof --info counters` on 950PR to
   see if `bank_conflict` or similar hardware counters are exposed; if so,
   re-run the existing stride sweep while counting — that would produce a
   definitive K,W answer.
