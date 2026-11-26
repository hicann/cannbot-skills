<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# SlidingWindow 范式 AscendC 能力映射

> 通用层 [capability-contract.md](../../generic/capability-contract.md) 谓词在 AscendC SIMT（ascend950 / arch35 实测口径）下的真值。其他 SoC 须重新核实，禁止外推。

## 1 真值表

| 谓词 | 真值（ascend950 / arch35） | 依据 |
|------|---------------------------|------|
| `CAP_SIMT_GM_DIRECT` | ✅ `__gm__ T*` 指针在 `__simt_vf__` 函数内直接读写离散地址 | max_pool_with_argmax_v3 / max_pool_grad_with_argmax_v3 / avg_pool_v2_grad 窗口读写实测 |
| `CAP_FP_NEG_INF` | ✅ `F32_NEG_INF = 0xff800000` 位模式 + `reinterpret_cast`（或 `fmaxf` 域 −Inf 常量） | max_pool_with_argmax_v3 max 初始化实测 |
| `CAP_FP_MAX_NAN` | ✅ `(val > acc) \|\| isnan(val)` 手写标量更新（CycleUpdate）实现 NaN 优先传播；`isnan` 标量原语可用 | max_pool_with_argmax_v3_simt.h CycleUpdate |
| `CAP_INT_DIV_COST` | 双档实测：正向窗口聚合直除 `%` / `/` 可接受（THREAD_DIM 256，max_pool_with_argmax_v3）；坐标分解密集场景须 UintDiv 快除（反向 THREAD_DIM 1024 + 8 参数表，max_pool_grad_with_argmax_v3；avg_pool 正向 6 参数表） | 上述算子分档实测；参数求法见 [../../../common/uintdiv-rules.md](../../../common/uintdiv-rules.md) |
| `CAP_SIMT_ATOMIC_ADD_DTYPE_SET` | `{float, half, bfloat16_t, int32, int64}`；double 不支持 GM 原子加 | Sparse 范式同平台登记（[../../../sparse/adapters/ascendc/capability-mapping.md](../../../sparse/adapters/ascendc/capability-mapping.md)）；本范式分支 C 沿用 |
| `CAP_SYNC_ALL` | ✅ `AscendC::SyncAll()`，头文件 `simt_api/device_sync_functions.h` | 分支 C 两段骨架（同 Sparse 范式登记） |
| `CAP_SYNC_SCHEDULE_MODE` | ✅ tiling 侧 `context->SetScheduleMode(1)`；缺失时 SyncAll 行为不保证 | 同 Sparse 范式登记；max_pool_v3_grad 生成记录 design_peer HIGH（漏配实测事故） |
| `CAP_SIMT_THREAD_MAX` | 2048（`dim3` 三维乘积上限） | SIMT API 文档（common_functions.h）；同 Sparse 范式登记 |
| `CAP_STACK_PER_THREAD` | 64 B/线程 | 同 Sparse 范式登记 |
| `CAP_VF_PARAM_MAX` | 28（`asc_vf_call` 入参个数上限；只支持裸指针与基本类型） | max_pool_with_argmax_v3 SIMT 入参约 14 标量 + 指针 < 28 |
| `CAP_UB_HYBRID_SIZE` | SIMD/SIMT 混合场景 216 KB（非混合 248 KB） | max_pool_grad_with_argmax_v3 divParams（8 元素）经 `TBuf` 分配；同 Sparse 范式登记 |

## 2 派生事实

- **线程分档（正反向不同档是预期形态）**：正向窗口聚合 256（max 的 −Inf + CycleUpdate 分支 + mask 写占寄存器，max_pool_with_argmax_v3 实测）；反向反窗口枚举 1024（梯度累加器轻、无效扫描占比高，max_pool_grad_with_argmax_v3 / avg_pool_v2_grad 实测）；分支 C 原子散射 512（同 Interpolation/Sparse 档）
- **核数传递形态**：正向 kernel 的 `blockIdxParam` / `blockNum` 由宿主类经 `GetBlockIdx()` / `GetBlockNum()` 计算后作标量入参传入 `__simt_vf__` 函数（blockIdx/blockDim 内建量与自定义循环起点并存，max_pool_with_argmax_v3 形态）
- **快除参数表布局**：每维 2 项（magic/shift），反向 4 维 8 参数（HW/W/strideH/strideW），经 `LocalTensor` 分配 + `DataSyncBarrier<MemDsbT::UB>()` 写后同步
- **NCHW / NHWC 双布局**：同一 SIMT kernel 模板按 `FORMAT_TYPE` 双实例（分解顺序 c 在内 / c 在外各自维护）；由 TilingKey 分派
- **max 混合精度**：fp16/bf16 输入升 fp32 比较（`static_cast<float>` 后与 fp32 −Inf 哨兵比较），winner 索引不受 dtype 影响

## 3 待验证项

| 项 | 验证义务 |
|----|---------|
| 直除在 950 之外 SoC（如 910B）的代价档位 | 迁移平台时以正向热点算子实测重核 `CAP_INT_DIV_COST` |
| 反向 1024 线程在含原子分支 C 的收益 | 分支 C 首例实测（当前 1024 档仅 A/B 分支实测） |
| fp16 winner 索引 dtype 分档（int32/int64）对反窗口命中比较的开销 | 大 shape int64 场景实测 |
