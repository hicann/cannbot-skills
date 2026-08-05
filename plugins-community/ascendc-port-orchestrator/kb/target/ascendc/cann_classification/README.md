# CANN op-family → architecture classification

**Source**: batch-grep of `~/workspace/cann/{ops-nn,ops-transformer}` 2026-05-25 by main agent per owner directive 02:16Z.

**Purpose**: prevent expedient-pure-VEC hack-class output when CANN reference uses cube+vec fused architecture. Same anti-cheat class as CPU-fallback.

**Cited by**: OL-188 (architecture-requirement gate) + finalize_pipeline hook `_check_architecture_class`.

## Grep methodology

Markers detected (any of):
- `matmul::Matmul<` (matmul library cube instantiation)
- `MatmulImpl<` (lower-level cube primitive)
- `REGIST_MATMUL_OBJ` (KFC client cube registration)
- `KERNEL_TYPE_MIX_AIC_1_2` / `KERNEL_TYPE_MIX_AIC_2_2` (MIX kernel task type)
- `cube_block` (V351 cube tile-block abstraction)

Run from CANN repo root: `find ops-nn ops-transformer -type d -name op_kernel`, then `grep -rE <markers>` per dir.

## Results (2026-05-25 grep)

| Total ops scanned | Cube-required | Vec-only (default) |
|---|---|---|
| 998 | 85 | 913 |

### Cube-required op families (85 ops total)

| Family | Op count | Notes |
|---|---|---|
| `ops-transformer/attention` | 30 | FA / MLA / sparse / quant / GQA variants |
| `ops-nn/matmul` | 14 | batch_mat_mul_v3 / quant_batch_matmul_v3 / fused_quant_mat_mul / weight_quant_batch_matmul_v2 etc. |
| `ops-transformer/experimental` | 12 | experimental fused variants |
| `ops-nn/conv` | 8 | conv2d_v2 / conv3d_v2 + backprop / deformable_conv2d |
| `ops-transformer/mhc` | 5 | mhc_pre / mhc_post + sinkhorn variants |
| `ops-transformer/gmm` | 5 | grouped_matmul + swiglu_quant + finalize_routing |
| `ops-transformer/ffn` | 4 | feed-forward fused variants |
| `ops-nn/rnn` | 3 | dynamic_rnn / dynamic_rnnv2 / single_layer_lstm_grad |
| `ops-nn/experimental` | 2 | matmul_fp32 + weight_quant_batch_matmul_experiment |
| `ops-transformer/posembedding` | 1 | rotary_position_embedding (uses cube for one variant) |
| `ops-nn/quant` | 1 | flat_quant (cube + vec mixed) |

Full op list: `cube_required_ops.txt` in this dir.

### Vec-only families (default)

Examples: `ops-nn/activation`, `ops-nn/foreach`, `ops-nn/index` (gather/scatter), `ops-nn/loss`, `ops-nn/norm`, `ops-nn/optim`, `ops-nn/pooling`, `ops-nn/hash`, `ops-transformer/moe` (routing/expand ops — distinct from gmm), `ops-transformer/common`.

These ops are CORRECT to implement as VEC-only — pure-VEC matches CANN's reference.

## Classification rule (canonical)

**An op REQUIRES cube architecture if**:
1. Its V220 / V351 CANN reference `op_kernel/*.{h,cpp}` contains ANY of the cube markers above
2. OR its dispatcher (`<op>.cpp`) registers `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_*)`
3. OR its host tiling computes TCubeTiling / matmul-related tile sizes

**An op is VEC-only OK if**:
1. None of the cube markers present in CANN reference
2. AND the algorithm doesn't have GEMM / convolution / attention-scoring as core compute

**Boundary cases**:
- Reduction ops (e.g. norm with built-in reduce) — usually VEC OK
- Element-wise quant ops without GEMM — VEC OK
- `flat_quant` family (uses cube for one path, vec for another) — see OL-185 for calibration

## Safety-net detection rule (OL-188 + finalize hook)

A generated kernel is **architectural-hack** (REJECT pre-ship) IF:
- Op-family is in the cube-required list above, AND
- Generated `<op>_kernels.cpp` / kernel.h contains NONE of: `matmul::Matmul<`, `MatmulImpl<`, `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_*)`, `REGIST_MATMUL_OBJ`

Implementation: `src/scripts/orchestrator/checks/architecture_class_check.py` (to be written).

## Customer applicability

This classification is part of the harness KB — customer fresh-installing the harness gets this file + the OL entry + the safety-net hook. When customer generates an op:
- Worker reads op-family classification from KB before authoring kernel
- Finalize gate rejects vec-only output for cube-required op-class
- Customer either fixes their kernel OR explicitly waives with documented justification

This is the "vec-only-when-CANN-is-fused = HACK" rule operationalized in harness layer, not just prose.
