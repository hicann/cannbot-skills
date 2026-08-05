---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "SIMT 原子操作：Simt 命名空间 API 与锁定/解锁模式"
description: "用 Simt::AtomicCas 锁 key（返回旧值，==key 即锁定成功）、Simt::AtomicExch 解锁、atomicAdd 递增计数；对应 source 的 compare_exchange_strong / store(release) / atomicAdd。"
confidence: single_run
original_id: SIMT_PATTERNS.md#4-原子操作
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, simt, optimization, atomics, compare-and-swap]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
AscendC SIMT 原子操作由 `Simt` 命名空间 API 提供，通过 `kernel_operator.h` 引入。移植 source 原子逻辑时按下表对应替换。

关键 API（`Simt::AtomicCas` / `AtomicExch` 均返回操作前的旧值）：
- `T Simt::AtomicCas(__gm__ T* ptr, T expected, T desired)`：把 `*ptr` 从 expected 原子替换为 desired，返回旧值。
- `T Simt::AtomicExch(__gm__ T* ptr, T val)`：原子把 `*ptr` 设为 val，返回旧值。
- `int32_t Simt::Ffs(int32_t val)`：返回最低位 1 的位置（1-based）。
- `void Simt::ThreadBarrier()`：block 级线程屏障同步。
- `T atomicAdd(__gm__ T* ptr, T val)`：全局原子加，注意不在 `Simt` 命名空间内。

锁定/解锁模式（对应 source `compare_exchange_strong` + `store(release)`）：
```cpp
// 锁定 key：CAS 把当前 key 换成 LOCKED_KEY
K try_key = Simt::AtomicCas(current_key_ptr, key, static_cast<K>(LOCKED_KEY));
if (try_key == key) {
    // 成功锁定（旧值等于期望的 key）
}
// 解锁：原子写回 key，丢弃返回值
(void)Simt::AtomicExch(current_key_ptr, key);
```

计数用 `atomicAdd`：`atomicAdd(bucket_size, 1)` 递增桶大小；`uint64_t evicted_idx = atomicAdd(d_evicted_counter, 1UL)` 获取淘汰位置。
