# AscendC Kernel Optimization Strategy Index

Source: production-grade operators from ops-nn and ops-transformer. Select all applicable strategies for your operator, then **Read each referenced detail file** before writing AscendC code.

## 多数据类型支持策略 (Multi-Dtype Support)

| ID | Strategy | When to Apply | Sources | Detail File |
|----|----------|---------------|---------|-------------|
| D1 | Mixed precision architecture | Any op with FP16/BF16 input | 28 ops | strategies/dtype_01_mixed_precision.md |
| D2 | Template kernel type dispatch | Compile-time multi-type kernel | 18 ops | strategies/dtype_02_template_kernel.md |
| D3 | TilingKey-driven type dispatch | Host-side multi-type selection | 33 ops | strategies/dtype_03_tilingkey_dispatch.md |
| D4 | FP8/INT4 quantization conversion | Quantization output operators | 7 ops | strategies/dtype_04_fp8_int4_conversion.md |
| D5 | BF16-specific handling | BF16 output with rounding control or multi-platform | 18 ops | strategies/dtype_05_bf16_special_handling.md |

## 性能优化策略 (Performance Optimization)

| ID | Strategy | When to Apply | Sources | Detail File |
|----|----------|---------------|---------|-------------|
| P1 | Double buffering (BUFFER_NUM=2) | Any memory-bound kernel | 27 ops | strategies/perf_01_double_buffer.md |
| P2 | Adaptive tiling (Split-N vs Split-D) | Variable row/column dimensions | 20 ops | strategies/perf_02_adaptive_tiling.md |
| P3 | Small-D multi-row merging | Hidden size ≤ 640 or small D | 5 ops | strategies/perf_03_small_d_optimization.md |
| P4 | Multi-core load balancing | Uneven data distributions | 20 ops | strategies/perf_04_load_balance.md |
| P5 | Pipeline sync (PipeBarrier/events) | All double-buffered kernels | 17 ops | strategies/perf_05_pipeline_sync.md |
| P6 | Multi-algorithm adaptive selection | Normalization, reduction ops | 11 ops | strategies/perf_06_multi_algo_selection.md |
| P7 | 32B alignment + DataCopyPad | Non-aligned input shapes | 27 ops | strategies/perf_07_data_alignment.md |
| P8 | UB memory partitioning | Kernels with multiple UB tensors | 9 ops | strategies/perf_08_ub_memory_mgmt.md |
| P9 | Deterministic output (workspace) | Training ops needing reproducibility | 5 ops | strategies/perf_09_deterministic_output.md |
| P10 | Vectorized data copy | CopyIn/CopyOut optimization | 19 ops | strategies/perf_10_vectorized_copy.md |
| P11 | Tail block handling (GatherMask) | Pooling/gather with uneven splits | 9 ops | strategies/perf_11_tail_block_handling.md |
| P12 | Broadcast & mask operations | Operators with mask inputs, broadcasting dimensions, or conditional selection | 4 ops | strategies/perf_12_broadcast_mask.md |
| P13 | Special algorithms & high-level AscendC APIs | Complex control flow, irregular access, or domain-specific high-level APIs | 13 ops | strategies/perf_13_special_algorithms.md |

## 精度优化策略 (Precision Optimization)

| ID | Strategy | When to Apply | Sources | Detail File |
|----|----------|---------------|---------|-------------|
| A1 | FP32 intermediate computation | BF16/FP16 with precision requirement | 9 ops | strategies/acc_01_fp32_intermediate.md |
| A2 | Welford numerically stable mean/var | LayerNorm, BatchNorm, RMSNorm | 5 ops | strategies/acc_02_welford_algorithm.md |
| A3 | Rounding mode control (CAST_*) | Any Cast to lower precision | 16 ops | strategies/acc_03_rounding_mode.md |
| A4 | SetFlag/WaitFlag event sync | Data dependencies across pipes | 3 ops | strategies/acc_04_pipeline_barrier.md |
| A5 | Softmax numerical stability | Softmax, attention score ops, any op with NaN/Inf risk | 20 ops | strategies/acc_05_softmax_stability.md |
| A6 | High-precision rsqrt (Newton-Raphson) | Normalization ops needing rsqrt | 3 ops | strategies/acc_06_high_precision_rsqrt.md |
| A7 | Index & boundary safety | Any operator with index inputs, gather/scatter, or user-controlled indices | 6 ops | strategies/acc_07_index_boundary.md |
| A8 | Quantization-specific precision | Quantization output operators, custom float formats, dequantization with bias | 5 ops | strategies/acc_08_quant_precision.md |

## How to Use

1. Identify your operator's characteristics (dtype, shape, op type)
2. Select ALL applicable strategy IDs from the tables above
3. Read each referenced `strategies/XXX.md` detail file
4. Apply the selected patterns in `tiling_pass`, `init_pass`, `process_pass`, `process_nonaligned_pass`

### Quick Reference by Operator Type

| Operator Type | Recommended Strategies |
|---------------|------------------------|
| LayerNorm / RMSNorm | D1, D2, P1, P2, P3, A1, A2, A6 |
| Quantization ops | D1, D4, D3, P2, P4, A3, A8 |
| Element-wise (foreach) | D1, D2, P1, P2, A1, A4 |
| Softmax / Attention | D1, P1, P5, A3, A4, A5 |
| Pooling / Gather | D1, P1, P2, P11, A7 |
| Optimizer ops | D1, P1, P2, P9, A1 |
| Broadcast / Mask ops | D1, P1, P12, A5 |
| Index / Scatter ops | D1, P1, P7, A7 |
| Special / Complex ops | P13, D2, D3 |
