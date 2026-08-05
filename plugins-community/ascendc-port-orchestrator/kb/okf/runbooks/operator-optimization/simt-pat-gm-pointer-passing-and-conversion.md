---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "全局内存指针的 GM_ADDR 参数声明与 __gm__ 内部转换"
description: "主机分配的全局内存指针一律以 GM_ADDR 入参，禁止用具体指针类型；函数体内 reinterpret_cast 为带 __gm__ __restrict__ 限定符的指针，禁止省略 __gm__。"
confidence: single_run
original_id: SIMT_PATTERNS.md#2-全局内存参数-GM_ADDR
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, simt, optimization, gm-addr, memory-pointer]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
所有来自主机分配（`aclrtMalloc`）的全局内存指针，跨 kernel 边界传递时必须遵守两条规则：

1. 参数声明：一律声明为 `GM_ADDR` 类型，禁止直接使用具体指针类型。
   - 正确：`GM_ADDR buckets_gm, GM_ADDR keys_gm, GM_ADDR values_gm`
   - 错误：`Bucket<K,V,S>* buckets, K* keys`（不能直接用指针类型）

2. 内部转换：在函数体内用 `reinterpret_cast` 把 `GM_ADDR` 转换为带 `__gm__` 限定符（并加 `__restrict__`）的指针，按用途分型：

```cpp
// 读写型全局指针
__gm__ Bucket<K,V,S>* __restrict__ buckets = reinterpret_cast<__gm__ Bucket<K,V,S>*>(buckets_gm);
// 只读型（对应 source 的 const T*）
__gm__ const K* __restrict__ keys = reinterpret_cast<__gm__ const K*>(keys_gm);
// 标量输出
__gm__ bool* __restrict__ founds = reinterpret_cast<__gm__ bool*>(founds_gm);
// 指针数组（如 V** value_ptrs）
__gm__ V* __gm__* __restrict__ value_ptrs = reinterpret_cast<__gm__ V* __gm__*>(value_ptrs_gm);
// 整型指针
__gm__ int32_t* __restrict__ buckets_size = reinterpret_cast<__gm__ int32_t*>(buckets_size_addr_gm);
```

核心规则：函数体内所有全局内存指针赋值必须带 `__gm__` 限定符，禁止省略。指针数组需在两级都标 `__gm__`（`__gm__ V* __gm__*`）。
