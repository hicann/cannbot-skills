<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式原子操作落地

> AscendC SIMT 原子操作在散射累积场景的实测用法。冲突判定与决策树见 [../../generic/access-model.md](../../generic/access-model.md) §2/§3。

## 1 分 dtype 分发模式（AtomicAddDispatch）

按目标 dtype 重载分发，隔离 API 差异，kernel 主逻辑只调统一入口：

```cpp
__simt_callee__ inline void AtomicAddDispatch(__gm__ float* addr, float val)
{ asc_atomic_add(addr, val); }                      // 返回值可靠

__simt_callee__ inline void AtomicAddDispatch(__gm__ half* addr, half val)
{ asc_atomic_add(addr, val); }                      // 支持但返回值不可靠：只用写语义

__simt_callee__ inline void AtomicAddDispatch(__gm__ bfloat16_t* addr, bfloat16_t val)
{ asc_atomic_add(addr, val); }                      // 同上

// int64 走专用接口 asc_atomic_add_int64（sparse_bincount 口径）

__simt_callee__ inline void AtomicAddDispatch(__gm__ double* addr, double val)
{ /* CAS 模拟，见 §2 */ }
```

## 2 double 的 CAS 模拟（无原生 GM 原子加）

`asc_atomic_add` 不支持 `__gm__ double*`。用 `asc_atomic_cas` 循环 + 位级加法模拟：

```cpp
__simt_callee__ inline void AtomicAddDispatch(__gm__ double* addr, double val)
{
    __gm__ uint64_t* addrAsUll = (__gm__ uint64_t*)addr;
    uint64_t valBits = *(uint64_t*)((void*)&val);
    uint64_t old = *addrAsUll, assumed;
    do {
        assumed = old;
        uint64_t newBits = DoubleAddBits(assumed, valBits);   // 位级 IEEE 754 加法
        old = asc_atomic_cas(addrAsUll, assumed, newBits);
    } while (assumed != old);
}
```

- `DoubleAddBits` 用纯整数位操作实现 IEEE 754 double 加法（指数对齐、规格化、NaN/Inf 传播），因为核上禁 double 算术
- uint64 CAS 支持为 ascend950 实测口径，迁移平台须复核（capability-mapping 待验证项）
- CAS 循环在高竞争下性能退化明显：double + 重复索引场景优先评估分桶预归约路线

## 3 幂等直写替代原子

输出语义为幂等值时**消除原子**：

```cpp
if (binaryOutput) {
    output[outIdx] = static_cast<TW>(1);            // 直写，无竞争问题
} else {
    AtomicAddDispatch(&output[outIdx], w);
}
```

sparse_bincount 的 binary_output 实测采用；命中即省全部原子开销。

## 4 使用要点

| 要点 | 说明 |
|------|------|
| 返回值 | 只有 float 可依赖返回值；half/bfloat16_t 返回值不可靠（API 文档明确），链式逻辑禁止使用 |
| 竞争代价 | 同地址高频竞争时吞吐下降；索引均匀分布时影响小，聚簇分布需评估分桶 |
| 确定性 | 原子顺序非确定 → 浮点结果与 GPU atomicAdd 同口径（容差验证），整型精确 |
| 验证 | 重复索引用例必须覆盖（验证和守恒）；double 路径单独验证 CAS 位级加法正确性 |
