# PROBE REPORT — Q_mte2_parallel (Q4): MTE2 parallel channels

## Verdict

**ACCEPT_CORRECT** — all 3 kernels compile and execute bit-exact.

**MTE2 interpretation: SHARED_CHANNEL with marginal overlap** (ratio K2/K3 = 1.106×; K3/K1 = 1.761×).

**One-line summary**: Two concurrent `DataCopyPad`/`DataCopy` issues from a single AIV do NOT achieve the ≥1.5× dual-channel threshold the template hypothesised; there is only ~10% arbitration/prefetch benefit from issuing back-to-back vs serialized. The pattern "schedule 2 concurrent DataCopy on critical path for 2× MTE2 throughput" is NOT valid on this CANN/bisheng build — schedule `1 DataCopy + compute overlap` instead.

## Environment

- **date**: 2026-04-22T07:50Z
- **host**: npu_dev3 container on A5 (198.51.100.35)
- **NPU used**: NPU 1 (AICore 0%, idle before+after; HBM 5235/131072 MB; only NPU 0 had pytest traffic)
- **OS (container)**: openEuler 22.03 (LTS-SP4)
- **SOC**: Ascend950PR (per npu-smi), built with `-DSOC_VERSION=Ascend950PR_9589`
- **npu-smi**: 25.7.rc1.b087
- **CANN**: 9.0.0, innerversion `V100R001C10SPC001B218`, path `/usr/local/Ascend/cann-9.0.0`
- **bisheng/ccec**: `2026-03-21T17:07:34+08:00 clang version 15.0.5 (clang-5c68a1cb1231 flang-5c68a1cb1231)`
- **python / torch**: Python 3.11.13 at `/root/python3.11.13`, torch import works under CANN `set_env.sh` + `LD_LIBRARY_PATH` prepending `cann-9.0.0/x86_64-linux/lib64` (libhccl.so path)

## Question

Do two concurrent `DataCopy` (GM→UB) operations issued from a single AIV saturate 2× read bandwidth (documented as "per-AICore AXI 2×128B read + 1×128B write"), or do they share one channel?

## Hypothesis vs Observation

- **Hypothesis** (template): partial saturation, ~1.6×, not full 2×.
- **Observation**: K2/K3 = **1.106×**. K3 (two concurrent) is only ~10% faster than K2 (two sequential) for the same total bytes moved. K3/K1 = **1.761×** (K3 moves 2× the bytes of K1 in 1.76× the time, i.e., ~12% savings versus hypothetical full serialization).
- The observed parallel-overlap benefit (~10%) is well below the 1.5× "dual-channel" and 1.2× "partial-parallel" thresholds; it falls in the **SHARED_CHANNEL** bin per the template's verdict rules.

Interpretation: either (a) bisheng serialises the two `DataCopy` issues despite back-to-back placement, or (b) the hardware has one effective MTE2 issue slot per AIV so the documented "2 read channels per AICore" are split across the AIC + paired AIV rather than available within one AIV. The 10% overlap is consistent with instruction-pipeline / outstanding-request overlap (one copy's tail overlapping the other's head), not with true dual-channel saturation.

## Evidence

### Kernels (3 variants, all AIV-only, single block)

- **K1**: `REPEAT_N=64` iterations of: 1× `DataCopy(GM→UB, 64KB)` + MTE2→MTE3 flag + 1× `DataCopy(UB→GM, 64KB)` + MTE3→MTE2 flag.
- **K2**: same REPEAT_N, but each iteration does **two sequential halves** (first half read+write, then second half read+write) — enforces serialization via explicit MTE2↔MTE3 sync.
- **K3**: same REPEAT_N, but each iteration **issues two reads back-to-back** into two separate `TBuf<VECCALC>` regions (64KB each) before one joint MTE2→MTE3 sync, then **two writes back-to-back** + joint sync.

Per-buffer size 64 KB (2 × 64 KB = 128 KB fits within AIV UB). Loop factor 64× chosen so total DMA work (4 MB for K1, 8 MB for K2/K3) dominates kernel-launch overhead.

### Build

`/root/AscendOpGenAgent/utils/build_ascendc.py current_task -v Ascend950PR_9589 --clean` — build succeeded with no kernel-level warnings. (Clock-skew warnings unrelated; they come from tar-preserved future mtimes.)

```
[100%] Built target pybind11_lib
[build_ascendc] Build completed: /root/AscendOpGenAgent/current_task/kernel/build
```

Full log in `workspace/probe_mte2_parallel/build.log`.

### Runtime (on NPU 1, warmup=10, iters=30, aclrtEvent timing)

| Kernel | Bytes moved / launch (total read+write) | Median (us) | Min (us) | Stdev (us) |
|---|---|---|---|---|
| K1 (1× copy ×64) | 8 MB (4 MB rd + 4 MB wr) | 58.91 | 58.54 | 0.39 |
| K2 (2× seq ×64) | 16 MB (8 MB rd + 8 MB wr) | 114.74 | 114.28 | 0.59 |
| K3 (2× concurrent ×64) | 16 MB | 103.76 | 103.31 | 0.42 |

**Stdev under 1%** on all three — measurement noise is negligible; NPU 1 was idle throughout (re-checked post-run).

Ratios:
- **K2 / K3 = 1.106×** → only ~10% gain from issuing two copies concurrently vs serializing them. **Falls in SHARED_CHANNEL bin per template** (<1.2×).
- **K3 / K1 = 1.761×** → K3 moves 2× the bytes of K1; ratio of 1.76× (not 2.0×) shows ~12% overlap between the two concurrent transfers, not full parallelism.
- **K2 / K1 = 1.948×** → K2's two sequential copies scale almost exactly 2× (as expected for serialized DMAs).

### Raw files

- Kernel: `workspace/probe_mte2_parallel/kernel/probe_kernel.h`, `probe_kernels.cpp`, `probe_tiling.h`, `pybind11.cpp`
- Runner: `workspace/probe_mte2_parallel/run_probe.py`, `model_new_ascendc.py`
- Build log: `workspace/probe_mte2_parallel/build.log`
- Run log: `workspace/probe_mte2_parallel/run.log`
- Per-iter timings: `workspace/probe_mte2_parallel/timings.csv` (30 rows, all 3 kernels)
- On A5: `/root/AscendOpGenAgent/current_task/` (kernel build artifacts)

## Recommendation for orchestrator

1. **Do NOT promote "schedule two concurrent DataCopy to saturate dual MTE2 channels" as a KB performance pattern for Ascend950PR single-AIV kernels.** The 10% benefit is too small to justify the added register pressure / UB fragmentation / sync complexity. Flag to `ascend950pr.md §Memory` as "MTE2 parallel channels: observed only ~10% arbitration overlap within a single AIV (probe 2026-04-22, K2/K3=1.106×). Pattern 'concurrent DataCopy for 2× MTE2' is FALSIFIED on CANN 9.0.0 / bisheng 2026-03-21."
2. **Keep "1 DataCopy + compute overlap via double-buffered TQue"** as the canonical UB-bound optimization. The K2 vs K1 ratio (1.948×) confirms MTE2 does not magically scale — halving load is the actual win.
3. **One open question** worth a follow-up probe: is the dual-channel possibly available **across the AIC + AIV pair** (not within a single AIV)? Would need a mixed Cube+Vector kernel to test. Low priority — most AIV-only kernels can't exploit that anyway.

## Caveats

- **Buffer size**: 64 KB per buffer was chosen to fit 2 in UB. Larger concurrent transfers were not testable (2× 256 KB would exceed AIV UB). The ~10% overlap measured at 64 KB may be different at smaller sizes (where launch overhead dominates) or if UB were larger. The probe does NOT distinguish "shared channel" from "dual channel gated on larger transfer granularity".
- **REPEAT_N=64** reuses the same GM region each iteration, so HBM row buffer / L2-like caches may be warm. True cold-HBM behavior per iteration would need non-overlapping GM strides; not tested here. This does NOT invalidate the ratio (K2 and K3 have identical access pattern, both benefit equally from any caching).
- **CANN version sensitivity**: result may change with a future CANN / bisheng release — record re-probe on any bisheng upgrade.
- **Cross-AIV**: within a single AICore there are 2 AIVs sharing the AXI; this probe uses only 1 block (1 AIV). Saturation across 2 AIVs (2-block launch) was not tested and may behave differently.
- **Methodology lesson applied**: the Q1 L1-scratch finding warned "build success + no warnings ≠ runtime correctness". We hit exactly that trap at iter 1 — two earlier attempts had a live kernel crash (K3 write OOB) and silent wrong output (buffer-sizing mismatch). Only the repeat-loop + matched Python/kernel size convention gave bit-exact output on all 3 kernels.

## Iterations run (4 iters; 1 compile, 3 runtime-semantic)

1. **Iter 1** — initial kernels with 256 KB per buffer + 2 TQues for K3. Build OK; K3 crashed with `aivec error 263 "MTE write address out of range"`; root cause: 2× 256 KB exceeded AIV UB (192 KB).
2. **Iter 2** — refactored K3 to use TBufs (raw VECCALC) instead of TQues. Build OK; K3 crashed with identical error. Confirmed the issue was UB size, not TQue semantics.
3. **Iter 3** — shrunk buffers to 64 KB each (2× = 128 KB, fits in UB); fixed Python-side `HALF` constant to match kernel's `PROBE_ELEMS_HALF`. All kernels ran, K1 and K3 correct, K2 had one half zero (likely off-by-one in double-blocked TQue, not the question's focus). Timings 4–5 us each → launch overhead dominated → no conclusive signal.
4. **Iter 4** — loop-amplified each kernel 64× inside a single launch; replaced TQues with TBuf+explicit MTE2↔MTE3 flags (fixes K2's correctness issue from iter 3). All 3 bit-exact; per-launch time 58–115 us; stdev <1%. **Final data reported above.**
