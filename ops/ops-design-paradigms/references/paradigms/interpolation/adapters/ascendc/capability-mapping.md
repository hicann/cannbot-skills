<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式 AscendC 能力映射

> 通用层 [capability-contract.md](../../generic/capability-contract.md) 谓词在 AscendC SIMT（ascend950 / arch35 实测口径）下的真值。其他 SoC 须重新核实，禁止外推。

## 1 真值表

| 谓词 | 真值（ascend950 / arch35） | 依据 |
|------|---------------------------|------|
| `CAP_SIMT_GM_DIRECT` | ✅ `__gm__ T*` 指针在 `__simt_vf__` 函数内直接读写离散地址 | three_interpolate / sync_resize_bilinear_v2 邻居 gather 实测 |
| `CAP_FP64_AICORE` | ❌ **AICore 禁止 fp64 运算** | sync_resize_bilinear_v2 精度修复链路：golden fp64 语义须以 df32 补偿模拟 |
| `CAP_FMA_CONTROL` | ✅ `volatile` 中间变量强制乘积先舍入落盘，阻断 ccec 的 FMA 收缩 | three_interpolate issue #35 实测（偏差恒 2^-18，volatile 修复后逐位一致）；sync_resize MapCoord 同法 |
| `CAP_FMA_ERROR_TERM` | ✅ `fmaf(h, q, -h*q)` 提取精确误差项 | sync_resize_bilinear_v2 TwoProdFMA 分解（TermDf） |
| `CAP_FP_FLOOR` | ✅ `floorf` 向负无穷取整；配套 `fminf` / `fmaxf` / `isfinite` | sync_resize_bilinear_v2 MapCoord 实测 |
| `CAP_SIMT_ATOMIC_ADD_DTYPE_SET` | `{float, half, bfloat16_t, int32, int64}`；double 不支持 GM 原子加 | Sparse 范式同平台登记（[../../../sparse/adapters/ascendc/capability-mapping.md](../../../sparse/adapters/ascendc/capability-mapping.md)）；本范式 grad 散射（three_interpolate_backward / upsample_bilinear2d_grad）沿用 |
| `CAP_SYNC_ALL` | ✅ `AscendC::SyncAll()`，头文件 `simt_api/device_sync_functions.h` | grad 两段骨架（同 Sparse 范式登记） |
| `CAP_SYNC_SCHEDULE_MODE` | ✅ tiling 侧 `context->SetScheduleMode(1)`；缺失时 SyncAll 行为不保证 | 同 Sparse 范式登记 |
| `CAP_SIMT_THREAD_MAX` | 2048（`dim3` 三维乘积上限） | SIMT API 文档（common_functions.h）；同 Sparse 范式登记 |
| `CAP_STACK_PER_THREAD` | 64 B/线程 | 同 Sparse 范式登记 |
| `CAP_VF_PARAM_MAX` | 28（`asc_vf_call` 入参个数上限；只支持裸指针与基本类型，不支持结构体/数组） | three_interpolate 9 参 < 28 |
| `CAP_UB_HYBRID_SIZE` | SIMD/SIMT 混合场景 216 KB（非混合 248 KB），经 `SetLocalMemorySize`/TBuf 使用 | three_interpolate divParams（8 元素）经 `LocalMemAllocator<Hardware::UB>` 分配 |

## 2 派生事实

- **线程分档**：坐标映射 + 加权采样类 512（实测：1024 线程下 ~20+ 活跃寄存器值有 spill 风险，sync_resize R006 论证保持 512）；最近邻/极简采样可上探 2048；补偿累加路线中低档
- **32/64 位索引双路径**：`totalElements ≤ INT32_MAX` 时 `INDEX_SIZE_T = uint32_t`，否则 `uint64_t`——32 位路径地址偏移仍以 `uint64_t` 承载（`bs × ms × cs` 可超 UINT32_MAX，three_interpolate 修复实测）
- **快除参数经 UB 传递**：`GetUintDivMagicAndShift` 结果以 `LocalTensor<uint32_t>` 小表广播（8 元素 32B 对齐），写后 `DataSyncBarrier<MemDsbT::UB>()` 再入 kernel
- **索引读入防御**：idx 读值先 `static_cast<uint64_t>` 再 clamp 到 `[0, extent-1]`（int64 负值/大值防 OOB，实测）

## 3 待验证项

| 项 | 验证义务 |
|----|---------|
| `volatile` 阻断在新版本 ccec 的持续有效性 | 升级编译器版本后以 issue #35 类用例（恒定 ULP 偏差检测）回归 |
| 950 之外 SoC（如 910B）`CAP_FMA_ERROR_TERM` 行为 | 首次迁移前以 `fmaf` 误差项单测复核 |
| half/bf16 输入升 fp32 累加的末位舍回口径 | 与竞品 GPU 同 dtype 对拍 |

本表每行须在算子验证方案（Validation 产物）中登记回归义务与触发时机（[../../patterns.md](../../patterns.md) 消费规则 3）——缺登记即验证方案不完整（实测评审一般项）。验证方案中的「待补 / 待标定」占位行（如 tilingKey、性能目标值）同样适用本义务：逐行登记回归义务与触发时机；上游产物（如 BranchCatalog）已具备填入条件后仍留占位即缺陷（实测插值 r2 一般项）。
