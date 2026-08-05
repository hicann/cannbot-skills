# Roofline Model for Ascend950PR (A5)

Use this model to determine if a kernel is compute-bound or memory-bound, and to set realistic performance targets.

## Hardware Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| AI Vector Cores | 56 | hardware/target/ascend950pr.md |
| VEC Peak (FP32) | ~28 TFLOPS | 56 cores × 512 FLOPS/cycle × 1GHz |
| VEC Peak (FP16) | ~56 TFLOPS | 2x FP32 (half-precision packing) |
| **CUBE Peak (FP16)** | **~373 TFLOPS** | **Measured 2026-06-12, .171 NPU1 957b, torch.matmul 8192³** |
| **CUBE Peak (BF16)** | **~368 TFLOPS** | **Measured (same run)** |
| **CUBE Peak (FP32)** | **~24 TFLOPS** | **Measured — cube is fp16/bf16-optimized, fp32 not favored** |
| HBM Bandwidth | 1.5 TB/s | Measured (theoretical ~2 TB/s) |
| UB per Core | **248 KB usable** (256 KB phys) | `GetCoreMemSize(UB)`=253952 (262144 − 8KB framework reserve); cannbot npu-arch §2.5 canonical. SIMT-effective ≈208KB (−40KB DCache). Never hardcode — see hardware/target/ascend950pr.md |
| L2 Cache | 48 MB shared | Read-only cache |

> ⚠ **CUBE vs VEC peak — do not conflate.** matmul / FlashAttention / any
> op dominated by `Matmul`/`Mmad` runs on the **CUBE unit** (peak ~373 TFLOPS
> fp16), NOT the vector unit (~56 TFLOPS fp16). Using the VEC peak as the
> ceiling for a cube-bound op understates it ~6.7× → an op at 24% of its real
> cube ceiling looks like ~80%+ "efficient" and the perf gate wrongly stops
> optimizing. Elementwise / reduction / softmax are vector-unit ops → use the
> VEC peak. `roofline_eval.py` encodes this split via `peak_cube_*` fields +
> `_peak_tflops(op_type)`; ridge is computed per-unit (see below).

## Operational Intensity Calculation

```
OI = FLOPs / Bytes_transferred (FLOP/byte)

For SG Forward (per token, top_k experts):
  FLOPs = 2 * top_k * hdim (multiply + add per element per expert)
  Bytes = top_k * hdim * sizeof(T) (load expert) + hdim * sizeof(T) (store output)
  OI = 2 * top_k * hdim / ((top_k + 1) * hdim * sizeof(T))
     = 2 * top_k / ((top_k + 1) * sizeof(T))

  fp32, K=4: OI = 8 / (5*4) = 0.4 FLOP/byte → memory-bound
  fp16, K=4: OI = 8 / (5*2) = 0.8 FLOP/byte → memory-bound
```

## Ridge Point

```
Ridge OI = Peak_FLOPS / Peak_BW

VEC-bound ops (elementwise / reduction / softmax):
  fp32: 28 TFLOPS / 1.5 TB/s ≈ 18.7 FLOP/byte
  fp16: 56 TFLOPS / 1.5 TB/s ≈ 37.3 FLOP/byte

CUBE-bound ops (matmul / attention):
  fp16: 373 TFLOPS / 1.5 TB/s ≈ 248.7 FLOP/byte
  fp32:  24 TFLOPS / 1.5 TB/s ≈  16.0 FLOP/byte
```

The ridge for a cube-bound op is ~6.7× higher than the vec ridge — many
matmul/FA shapes that look "compute-bound" against the vec ridge are still
well below the cube ridge (i.e. memory-bound on the cube unit).

SG kernels (OI < 1) are **deeply memory-bound**. Optimizations should focus on:
1. Reducing GM reads (data reuse, loop reorder)
2. Maximizing MTE2/VEC overlap (TQue prefetch)
3. NOT on compute optimizations (VEC is not the bottleneck)

## Theoretical Performance Bounds

### SG Forward (Memory-Bound)

```
min_time = total_bytes / bandwidth
total_bytes = T * K * hdim * sizeof(T)  # expert reads (dominant)
            + T * hdim * sizeof(T)       # output writes

Example: prod_a (T=8192, K=4, H=256, fp32):
  read_bytes = 8192 * 4 * 256 * 4 = 32 MB
  write_bytes = 8192 * 256 * 4 = 8 MB
  total = 40 MB
  min_time = 40 MB / 1.5 TB/s = 0.027 ms

  With expert reuse (expert-major, 64 experts):
  read_bytes = 64 * 256 * 4 = 64 KB (experts) + 8192 * 4 * 4 = 128 KB (indices)
  total ≈ 8.2 MB (output + indices + expert data)
  min_time = 8.2 MB / 1.5 TB/s = 0.005 ms
```

## Using the Model

### Before Exploration
1. Calculate OI for the kernel
2. Determine if compute-bound or memory-bound
3. Calculate theoretical min_time
4. Compare with actual time → efficiency = theoretical / actual

### Setting Targets
- If efficiency < 30%: significant optimization opportunity exists
- If efficiency 30-60%: moderate improvements possible
- If efficiency > 60%: close to hardware limits, diminishing returns
- If efficiency > 80%: near optimal, explore algorithmic changes only

### Early Stopping
If after an optimization:
- The kernel is >60% of theoretical peak → consider stopping
- Further improvement requires >2x reduction in data movement → may need algorithm redesign (escalate to human)
