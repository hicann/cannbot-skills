---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Multi-instance partition-dispatch when output rows partition by a host-knowable boundary"
description: "When output rows partition cleanly by a host-computable boundary (group offsets, batch/expert/segment id), set blockDim = number_of_partitions and let each core own one partition, applying the single-instance accelerator pattern to its slice — deterministic by construction."
confidence: single_run
original_id: OL-93
classified_by: llm-assisted
timestamp_inferred: true
tags: [platform-compat, optimization, ol-93, partition-dispatch, blockdim, moe]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**When it applies** — the op's output is a concatenation of independent sub-computations along one axis and the partition boundaries are known on the host before launch. Source shapes that match:
- A loop of independent ops over a list of sub-shapes (`for g in range(G): out_g = f(A_g, B[g])`, result `concat(out_0, …, out_{G-1})`).
- A single op with a "groups" / "experts" / "segments" / "buckets" parameter where each group's compute is independent (MoE expert dispatch, GroupedMatmul, GroupedConv, segmented-batch attention, segmented sparse-gather, multi-batch RNN unroll).
- An op whose output is a stack/concat of per-batch results AND batch-size is bounded by AIC count.

### Principle — partition-as-AIC-grant

When per-output-row work partitions cleanly by a boundary the host can compute (group offsets, batch index, expert ID, segment start):
- **Set `blockDim = number_of_partitions`** and have each accelerator core own exactly one partition's work.
- Each core applies the **single-instance accelerator pattern** (typed-config + on-stack runtime tiling, per OL-91) to ITS partition. The accelerator config is identical per core; only the shape-runtime fields (slice rows, slice base offset) vary.

### Two host-side pre-computes (CPU vectors → small NPU tensors)

- **Cumulative output offsets:** `cum_out[P+1]` where `cum_out[p+1] = cum_out[p] + rows_in_partition_p`. Each core reads `cum_out[bid]` for its output base — no per-core scan.
- **Sentinel-extended boundary array:** when input partition boundaries come from an offset list `offsets[P]` with the "last partition runs to end" convention, **append a sentinel**: `offsets[P] = total_rows`. Every core then uses `end = offsets[bid+1]` uniformly, with no last-group special case. (Mirror this in pybind: allocate `offsets` of size `P+1` with the sentinel set.)

### Per-core kernel decode (three lines)

```cpp
const int32_t bid            = GetBlockIdx();
const int32_t row_off        = uniform_partition ? bid * uniform_size : offsets[bid];
const int32_t partition_size = uniform_partition ? uniform_size : (offsets[bid + 1] - row_off);
```

The `uniform_partition` flag is a host-passed scalar (`int32_t`). Branching on it costs one register compare per core — negligible vs the matmul/op body.

### Determinism (by construction)

Each output row is owned by exactly one core; no atomic write; per-core operation order is fixed by the typed-config's constant tiling. `DET_POLICY=required` is satisfied without contortion regardless of partition count or partition-size variability.

### Perf shape

Matches the reference well when the per-core partition is small/medium and the reference is also doing `P` independent calls (each one accelerator-call). (Source text truncated here; the full perf-mismatch case was not captured in the batch excerpt.)
