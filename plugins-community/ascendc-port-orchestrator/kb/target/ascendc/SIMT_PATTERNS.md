# AscendC SIMT Kernel 编程模式参考

本文档总结了 HierarchicalKV-ascend 项目中 AscendC SIMT kernel 的编程规范和常用模式，
基于已完成的参考实现（`clear_kernel`, `find_and_update_kernel`, `init_table_kernel`, `insert_and_evict_kernel`）归纳而来。

---

## 1. 函数签名规范

### 标准签名模板

```cpp
template <typename K = uint64_t, typename V = float, typename S = uint64_t>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void {kernel_name}_vf(
    GM_ADDR buckets_addr_gm,          // 哈希桶数组（全局内存）
    GM_ADDR buckets_size_addr_gm,     // 桶大小数组（如需）
    uint64_t capacity,                // 哈希表总容量（桶数 × 桶大小）
    uint32_t bucket_capacity,         // 单桶最大 key 数
    uint32_t dim,                     // value 向量维度
    GM_ADDR keys_addr_gm,             // 输入 key 数组
    // ... 其他 GM_ADDR 参数 ...
    uint64_t n,                       // 本次操作的 key 数量
    uint64_t thread_all,              // 总线程数（gridDim × blockDim）
    uint64_t global_epoch,            // 当前 epoch（用于 score 计算）
    uint32_t block_index,             // 当前 block 索引（对应 source blockIdx.x）
    uint32_t max_bucket_shift,        // 桶索引位移量（log2(bucket_capacity)）
    uint64_t capacity_divisor_magic,  // 快速取模魔数
    uint64_t capacity_divisor_shift   // 快速取模位移
)
```

### 关键修饰符说明

| 修饰符 | 含义 |
|--------|------|
| `__simt_vf__` | 标识函数为 AscendC SIMT VF（Vector Function）kernel |
| `__aicore__` | 声明运行在 AI Core 上（对应 source 的 `__global__` 或 `__device__`） |
| `LAUNCH_BOUND(N)` | 声明 kernel 最大线程数，N 为 `THREAD_NUM`（通常 512 或 1024） |
| `inline` | 必须声明为内联 |

### THREAD_NUM 选择规则

AscendC 使用编译时常量 `THREAD_NUM`，根据算子复杂度选择：

| 算子类型 | THREAD_NUM | 说明 |
|----------|------------|------|
| 查找类（find/find_and_update） | 512 | 平衡并行度和资源使用 |
| 插入更新类（insert/upsert） | 512 | 需要原子操作， moderate 并行度 |
| 批量初始化/清空（clear/init） | 1024 | 简单操作，高并行度 |
| 导出类（dump） | 2048 | 高并行度，协作组处理 |

**注意**：AscendC `THREAD_NUM` 是编译时常量，与 source 运行时动态决定 block size 不同。

---

## 2. 全局内存参数（GM_ADDR）

### 参数声明规则

所有来自主机分配（aclrtMalloc）的全局内存指针，在参数中必须声明为 `GM_ADDR` 类型：

```cpp
// ✅ 正确
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void my_kernel_vf(
    GM_ADDR buckets_gm,
    GM_ADDR keys_gm,
    GM_ADDR values_gm,
    uint64_t n,
    ...
)

// ❌ 错误
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void my_kernel_vf(
    Bucket<K,V,S>* buckets,  // 不能直接用指针类型
    K* keys,
    ...
)
```

### 内部指针转换规则

在函数体内，将 `GM_ADDR` 转换为带 `__gm__` 限定符的指针：

```cpp
// 读写型全局指针
__gm__ Bucket<K, V, S>* __restrict__ buckets =
    reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_gm);

// 只读型全局指针（对应 source 的 const T*）
__gm__ const K* __restrict__ keys =
    reinterpret_cast<__gm__ const K*>(keys_gm);

// 标量输出
__gm__ bool* __restrict__ founds =
    reinterpret_cast<__gm__ bool*>(founds_gm);

// 指针数组（如 V** value_ptrs）
__gm__ V* __gm__* __restrict__ value_ptrs =
    reinterpret_cast<__gm__ V* __gm__*>(value_ptrs_gm);

// 整型指针
__gm__ int32_t* __restrict__ buckets_size =
    reinterpret_cast<__gm__ int32_t*>(buckets_size_addr_gm);
```

> **核心规则**：函数体内所有全局内存指针赋值必须带 `__gm__` 限定符，禁止省略。

---

## 3. 线程索引模式

### 基础线程 ID 计算

```cpp
// AscendC（block_index 作为参数传入，不使用 blockIdx）
uint64_t kv_idx = (uint64_t)block_index * blockDim.x + threadIdx.x;

// 步进式遍历（处理 n > total_thread_num 的情况）
for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x;
     kv_idx < n; kv_idx += total_thread_num) {
    // 每线程处理一个 key
}
```

### 对比 source

```cpp
// source（不允许在 AscendC 中使用）
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
```

---

## 4. 原子操作

### 锁定/解锁模式对比

```cpp
// source: 锁定 key
K expected_key = key;
bool result = current_key->compare_exchange_strong(
    expected_key, static_cast<K>(LOCKED_KEY),
    source::std::memory_order_acquire, source::std::memory_order_relaxed);

// AscendC: 锁定 key
K try_key = Simt::AtomicCas(current_key_ptr, key, static_cast<K>(LOCKED_KEY));
if (try_key == key) {
    // 成功锁定
}

// source: 解锁
key_address->store(key, source::std::memory_order_release);

// AscendC: 解锁
(void)Simt::AtomicExch(current_key_ptr, key);
```

### Add

```cpp
// 递增桶大小
atomicAdd(bucket_size, 1);

// 获取淘汰位置
uint64_t evicted_idx = atomicAdd(d_evicted_counter, 1UL);
```

### Simt 命名空间 API

定义在 AscendC 运行时，通过 `kernel_operator.h` 引入：

```cpp
namespace Simt {

// Compare-And-Swap：将 *ptr 从 expected 原子替换为 desired
// 返回 *ptr 在操作前的值
template <typename T>
T AtomicCas(__gm__ T* ptr, T expected, T desired);

// Exchange：原子地将 *ptr 设为 val，返回旧值
template <typename T>
T AtomicExch(__gm__ T* ptr, T val);

// Find First Set：返回最低位 1 的位置（1-based）
int32_t Ffs(int32_t val);

// 线程屏障（block 级同步）
void ThreadBarrier();

}  // namespace Simt

// 全局原子加（不在 Simt 命名空间内）
template <typename T>
T atomicAdd(__gm__ T* ptr, T val);
```

---

## 5. 空指针保护

在 kernel 函数开头，必须对关键指针参数做空值保护：

```cpp
// 示例：find_and_update_kernel_vf
if (buckets_gm == nullptr) return;
if (keys_gm == nullptr) return;
if (value_ptrs_gm == nullptr) return;
```

---

## 6. 完整的函数头文件模板

```cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_{KERNEL_NAME_UPPER}_KERNEL_H_
#define ASCENDC_{KERNEL_NAME_UPPER}_KERNEL_H_

#include <kernel_operator.h>
#include <cstdint>
#include "../../../include/types.h"
#include "../../../include/utils.h"
// 仅在使用 ScoreFunctor 时包含：
// #include "../../../include/score_functor.h"

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;

template <typename K = uint64_t, typename V = float, typename S = uint64_t>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void {kernel_name}_vf(
    // 参数列表
) {
  // 空指针检查
  // ...

  // GM_ADDR → __gm__ 指针转换
  // ...

  // 主循环（步进式线程遍历）
  for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x;
       kv_idx < n; kv_idx += total_thread_num) {
    // 每线程处理逻辑
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_{KERNEL_NAME_UPPER}_KERNEL_H_
```

---

## 7. 已完成参考 kernel 一览

| Kernel | 文件路径 | 核心操作 |
|--------|----------|----------|
| `clear_kernel_vf` | `hkv_hashtable/clear_kernel/v35/clear_kernel.h` | 遍历所有槽位，写入 EMPTY_KEY/EMPTY_SCORE，重置桶大小 |
| `find_and_update_kernel_vf` | `hkv_hashtable/find_and_update_kernel/v35/find_and_update_kernel.h` | 哈希定位 → 线性探测 → 返回 value 指针和 found 标志 |
| `init_table_kernel` (多函数) | `hkv_hashtable/init_table_kernel/v35/init_table_kernel.h` | 初始化桶指针、填充 EMPTY_KEY/EMPTY_SCORE |
| `insert_and_evict_kernel_vf` | `hkv_hashtable/insert_and_evict_kernel/v35/insert_and_evict_kernel.h` | 插入 key，满时按 score 淘汰最旧 key |
| `dump_kernel_vf` | `hkv_hashtable/dump_kernel/v35/dump_kernel.h` | 导出 key-value-score，使用 GROUP_SIZE 协作 |
| `rehash_kernel_vf` | `hkv_hashtable/rehash_kernel/v35/rehash_kernel.h` | 表容量翻倍后重新哈希，碎片整理 |

在生成新 kernel 时，优先参考**功能最相似**的已有实现作为代码风格模板。

---

## 8. `.cpp` Dispatcher 文件模板

每个算子目录除 `v35/<kernel>.h` 外，还需在顶层生成 `<kernel>.cpp`，作为 AscendC 内核入口（对应 source 的 `__global__` 启动点）。

### 8.1 标准模板（含淘汰策略）

```cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "./v35/{kernel_name}_kernel.h"
#include <cstdint>
#include "../../include/simt_vf_dispatcher.h"
#include "kernel_operator.h"

using namespace npu::hkv;

extern "C" __global__ __aicore__ void {kernel_name}(
    GM_ADDR buckets,
    // ... 与 _vf 函数相同的 GM_ADDR 和标量参数 ...
    uint64_t n,
    uint64_t global_epoch,
    int32_t evict_strategy,           // 有淘汰策略时包含
    uint32_t value_size,              // 必须：用于类型分发
    uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  uint64_t system_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
  const uint64_t total_thread_num = THREAD_NUM * GetBlockNum();

  DISPATCH_VALUE_SIZE(
      value_size,
      DISPATCH_EVICT_STRATEGY(
          evict_strategy,
          (Simt::VF_CALL<{kernel_name}_vf<uint64_t, DTYPE, uint64_t, STRATEGY>>(
              Simt::Dim3{static_cast<uint32_t>(THREAD_NUM)},
              buckets,
              // ... 其他 GM_ADDR 参数 ...
              n,
              update_score,
              global_epoch,
              total_thread_num,
              system_cycle,
              GetBlockIdx(),
              max_bucket_shift,
              capacity_divisor_magic,
              capacity_divisor_shift))));
}
```

### 8.2 简单模板（无淘汰策略，如 clear_kernel）

```cpp
#include "./v35/{kernel_name}_kernel.h"
#include "../../include/simt_vf_dispatcher.h"

using namespace npu::hkv;

extern "C" __global__ __aicore__ void {kernel_name}(
    GM_ADDR buckets,
    GM_ADDR buckets_size,
    size_t bucket_max_size,
    size_t table_capacity,
    uint32_t value_size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  const uint64_t thread_all = THREAD_NUM * GetBlockNum();

  DISPATCH_VALUE_SIZE(
      value_size,
      (Simt::VF_CALL<{kernel_name}_vf<uint64_t, DTYPE, uint64_t>>(
          Simt::Dim3{THREAD_NUM},
          buckets, buckets_size, bucket_max_size,
          table_capacity, thread_all, GetBlockIdx())));
}
```

### 8.3 关键规则

| 规则 | 说明 |
|------|------|
| **入口函数名** | `{kernel_name}`（不带 `_vf` 后缀） |
| **修饰符** | `extern "C" __global__ __aicore__`（固定写法） |
| **block_index** | 在 `.cpp` 内用 `GetBlockIdx()` 获取，传入 `_vf` |
| **total_thread_num** | `THREAD_NUM * GetBlockNum()`，传入 `_vf` 的 `thread_all` 参数 |
| **DISPATCH_VALUE_SIZE** | 始终需要，将 `value_size` 映射为 `DTYPE` 模板参数 |
| **DISPATCH_EVICT_STRATEGY** | 仅当 kernel 使用 ScoreFunctor / 淘汰策略时包含 |
| **DISPATCH_GROUP_SIZE** | 仅当 kernel 有协作组（insert_and_evict/dump 类）时包含，见下方说明 |
| **Simt::VF_CALL** | 固定包装，第一参数为 `Simt::Dim3{THREAD_NUM}` |
| **include 路径** | `./v35/{kernel_name}_kernel.h`、`../../include/simt_vf_dispatcher.h` |
| **system_cycle** | 仅当 _vf 函数需要时：`static_cast<uint64_t>(AscendC::GetSystemCycle())` |

#### DISPATCH_GROUP_SIZE 使用说明

**适用算子**：`insert_and_evict_kernel`, `dump_kernel`, `rehash_kernel`

**功能**：将运行时的 `group_size`（2/4/8/16/32）映射为编译时 `GROUP_SIZE` 模板参数，用于协作组 shuffle 操作。

**嵌套顺序**（从内到外）：
1. 最内层：`Simt::VF_CALL<...>` 调用 kernel
2. `DISPATCH_EVICT_STRATEGY`（如需要）
3. `DISPATCH_VALUE_SIZE`
4. 最外层：`DISPATCH_GROUP_SIZE`（如需要）

**示例**（insert_and_evict）：
```cpp
DISPATCH_GROUP_SIZE(
    group_size,
    DISPATCH_VALUE_SIZE(
        value_size,
        DISPATCH_EVICT_STRATEGY(
            evict_strategy,
            (Simt::VF_CALL<
                insert_and_evict_kernel_vf<uint64_t, DTYPE, uint64_t, STRATEGY, GROUP_SIZE>
            >(...)))));
```

**注意**：使用 `DISPATCH_GROUP_SIZE` 的算子，其 `_vf` 函数需要额外添加 `GROUP_SIZE` 模板参数。
