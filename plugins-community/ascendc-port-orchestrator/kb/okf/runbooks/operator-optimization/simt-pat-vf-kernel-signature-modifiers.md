---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "AscendC SIMT VF kernel 函数签名与关键修饰符"
description: "SIMT VF kernel 的 _vf 函数须用 __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline 修饰，参数按 GM_ADDR 指针 + 标量的约定顺序排列。"
confidence: single_run
original_id: SIMT_PATTERNS.md#1-函数签名规范
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, simt, optimization, kernel-signature, vector-function]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
生成 AscendC SIMT VF（Vector Function）kernel 的核心计算函数（命名为 `{kernel_name}_vf`）时，函数头必须固定按以下修饰符组合声明：

```cpp
template <typename K = uint64_t, typename V = float, typename S = uint64_t>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void {kernel_name}_vf( ... )
```

修饰符含义：
- `__simt_vf__`：标识该函数为 AscendC SIMT VF kernel。
- `__aicore__`：声明运行在 AI Core 上，对应 source 的 `__global__` / `__device__`。
- `LAUNCH_BOUND(N)`：声明 kernel 最大线程数，N 为编译时常量 `THREAD_NUM`（通常 512 或 1024）。
- `inline`：必须声明为内联。

参数排列约定（与 source 原型对齐，便于逐一映射）：先是全局内存指针（一律 `GM_ADDR` 类型，如 `buckets_addr_gm`、`keys_addr_gm`），再是标量配置（`capacity`、`bucket_capacity`、`dim`）；随后是本次操作规模与调度参数：`n`（本次 key 数）、`thread_all`（总线程数 gridDim×blockDim）、`global_epoch`、`block_index`（对应 source blockIdx.x）、`max_bucket_shift`（log2(bucket_capacity)）、`capacity_divisor_magic` / `capacity_divisor_shift`（快速取模魔数与位移）。
