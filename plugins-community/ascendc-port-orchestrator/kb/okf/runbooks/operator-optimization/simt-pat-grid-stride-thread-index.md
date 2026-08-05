---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "SIMT kernel 线程索引与步进式（grid-stride）遍历"
description: "block_index 作为参数传入（不用 blockIdx），线程 ID = block_index*blockDim.x + threadIdx.x；用 kv_idx += total_thread_num 的步进循环覆盖 n > 总线程数的情况。"
confidence: single_run
original_id: SIMT_PATTERNS.md#3-线程索引模式
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, simt, optimization, grid-stride, thread-index]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
AscendC SIMT kernel 中，block 索引通过参数 `block_index` 传入（在 `.cpp` dispatcher 内用 `GetBlockIdx()` 取得），不能像 source 那样直接使用 `blockIdx`。

基础线程 ID：
```cpp
uint64_t kv_idx = (uint64_t)block_index * blockDim.x + threadIdx.x;
```

当处理的 key 数 `n` 可能大于总线程数时，使用步进式（grid-stride）主循环，每线程处理一个 key、步进 `total_thread_num`：
```cpp
for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x;
     kv_idx < n; kv_idx += total_thread_num) {
    // 每线程处理一个 key
}
```

对比 source（在 AscendC 中禁止照搬）：`int tid = blockIdx.x*blockDim.x + threadIdx.x; int stride = gridDim.x*blockDim.x;`。移植时把 `blockIdx.x` 替换为参数 `block_index`，把 `gridDim.x*blockDim.x` 步长替换为传入的 `total_thread_num`。
