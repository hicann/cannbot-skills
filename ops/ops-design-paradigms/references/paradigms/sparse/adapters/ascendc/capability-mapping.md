<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式 AscendC 能力映射

> 通用层 [capability-contract.md](../../generic/capability-contract.md) 谓词在 AscendC SIMT（ascend950 / arch35 实测口径）下的真值。其他 SoC 须重新核实，禁止外推。

## 1 真值表

| 谓词 | 真值（ascend950 / arch35） | 依据 |
|------|---------------------------|------|
| `CAP_SIMT_GM_DIRECT` | ✅ `__gm__ T*` 指针在 `__simt_vf__` 函数内直接读写离散地址 | sparse_bincount / sparse_segment_sum_grad / sparse_apply_adadelta 实测 |
| `CAP_SIMT_ATOMIC_ADD_DTYPE_SET` | `{float, half, bfloat16_t, int32, int64}`；**double 不支持 GM 原子加** | sparse_segment_sum_grad（源自 asc_atomic_add API 文档）；int64 用 `asc_atomic_add_int64` |
| `CAP_SIMT_ATOMIC_ADD_RETURN_RELIABLE_SET` | `{float}`（half / bfloat16_t 的返回值**不可靠**，API 文档明确说明） | sparse_segment_sum_grad：只用其原子写入语义，不依赖返回值 |
| `CAP_SIMT_ATOMIC_CAS_WIDTH` | `{uint32_t, uint64_t}`；uint64_t CAS 用于 double 模拟（实测采用，迁移其他平台时**待验证**） | sparse_segment_sum_grad `asc_atomic_cas((__gm__ uint64_t*)addr, assumed, newBits)` |
| `CAP_SYNC_ALL` | ✅ `AscendC::SyncAll()`，头文件 `simt_api/device_sync_functions.h` | sparse_segment_sum_grad 两阶段骨架 |
| `CAP_SYNC_SCHEDULE_MODE` | ✅ tiling 侧 `context->SetScheduleMode(1)` 启用同步调度；**缺失时 SyncAll 行为不保证** | sparse_segment_sum_grad tiling |
| `CAP_SIMT_THREAD_MAX` | 2048（`dim3` 三维乘积上限） | SIMT API 文档（common_functions.h） |
| `CAP_STACK_PER_THREAD` | 64 B/线程 | sparse_bincount 实测口径 |
| `CAP_VF_PARAM_MAX` | 28（`asc_vf_call` 入参个数上限；只支持裸指针与基本类型，**不支持结构体/数组**） | sparse_apply_adadelta（11 参 < 28） |
| `CAP_UB_HYBRID_SIZE` | SIMD/SIMT 混合场景 216 KB（非混合 248 KB），经 `SetLocalMemorySize`/TBuf 使用 | SIMT 混合编程辅助函数文档 |

## 2 派生事实

- **线程分档**：计算类（原子/分支）512、纯搬运 2048、行级状态类中低档——均须 ≤ 2048 且为编译期常量
- **索引 dtype 分档线程数**：sparse_segment_sum_grad 按 `sizeof(IDX_T)` 取 1024（int32）/ 512（int64），索引读宽与并行度权衡
- **元数据传递**：stride/shape 表 ≤ `CAP_VF_PARAM_MAX` 个标量可直接入参；超限或数组形态经 UB 共享内存传递（`__ubuf__` 指针），体积 ≤ `CAP_UB_HYBRID_SIZE`

## 3 待验证项

| 项 | 验证义务 |
|----|---------|
| `asc_atomic_cas` 对 `__gm__ uint64_t*` 的支持 | 首次使用 double 模拟路径前，以官方 API 文档 + 单元测试复核 |
| int64 原子加在不同 SoC 的可用性 | 迁移平台时复核 `asc_atomic_add_int64` 声明 |
| half/bfloat16_t 原子加精度口径 | 与竞品 GPU atomicAdd(half) 对齐验证 |
