<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式 Kernel 入口与 Tiling 组织

> AscendC SIMT 的入口模式、TilingData/TilingKey 组织与 host 侧要点。以 sparse_bincount / sparse_segment_sum_grad 实测结构为基准。

## 1 入口结构（三段式）

```cpp
template <uint32_t schMode>
__global__ __aicore__ void op_name(GM_ADDR x, GM_ADDR y, GM_ADDR output,
                                   GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(OpTilingData);
    GET_TILING_DATA_WITH_STRUCT(OpTilingData, tilingData, tiling);   // 1. 解 tiling

    __gm__ T* xGm = (__gm__ T*)x;                                    // 2. GM_ADDR 直转裸指针
    // 纯 SIMT 算子不经 GlobalTensor/GetPhyAddr 中转

    if constexpr (schMode == MODE_XXX) {                             // 3. 按 TilingKey 分派
        asc_vf_call<OpSimtKernel<TA, TB>>(
            dim3(THREAD_NUM),                                        //    编译期线程常量
            tilingData.field1, tilingData.field2,                    //    标量 + 裸指针入参
            xGm, yGm, outGm);
    }
}
```

- `__simt_vf__` 内核函数不能是类的成员函数；参数只支持裸指针与基本类型
- 一个 kernel 内允许多次 `asc_vf_call`（如 sparse_segment_sum_grad 的零初始化段 + 散射段），段间全核依赖须 `SyncAll()`（见 [../../generic/sync-model.md](../../generic/sync-model.md) §3）

## 2 TilingData 设计规则

| 规则 | 说明 |
|------|------|
| 字段 = host 可算的标量 | 切分参数（N、perCore、needCoreNum）、值域参数（size、rank）、模式开关（binaryOutput、hasWeights） |
| **不含线程数** | 线程数是编译期常量，随 TilingKey 分档 |
| **不含 kernel 可自算的量** | 循环边界类派生量在 kernel 内由基础字段推出 |
| 值依赖预计算结果入 tiling | sparse_reshape 的 stride、-1 维推断结果、y_shape 均在 tiling 侧算好传入 |

host tiling 侧骨架：

```cpp
int64_t coreNum = ascendcPlatform.GetCoreNumAiv();          // AIV 核数
int64_t perCore = (N + coreNum - 1) / coreNum;              // Step 1 均分
if (perCore < PER_CORE_MIN && N > 0) perCore = PER_CORE_MIN; // 下限保护（对齐 32）
int64_t needCoreNum = (N + perCore - 1) / perCore;           // Step 2 反推
context->SetBlockDim(needCoreNum);
context->SetScheduleMode(1);   // 仅同步骨架需要（SyncAll 前提）
// workspace：无跨核交换时仅申请系统 workspace（GetLibApiWorkSpaceSize）
```

## 3 TilingKey 编码规则

- **只编码场景模式，不编码 dtype**：dtype 由 `asc_vf_call` 的模板参数 + DTYPE_ 宏在编译期多实例化（sparse_bincount：单模式 MODE；sparse_segment_sum_grad：MODE_SCATTER 单模式多 dtype）
- 场景维度举例：rank 分档（1D/2D 输出）、伴随张量有无（weights）、幂等模式（binary_output）、索引 dtype 档
- 例外：线程数随索引 dtype 分档时（1024/512），该维度可显式进 TilingKey

## 4 dtype 多实例化

```cpp
// 按 (值dtype, 累加dtype) 组合展开模板实例
asc_vf_call<OpSimtKernel<int32_t, float>>(dim3(THREAD_NUM), ...);   // 组合#1
asc_vf_call<OpSimtKernel<int64_t, double>>(dim3(THREAD_NUM), ...);  // 组合#7
```

- 索引中间量统一 `int64_t` 承载：`int64_t val = static_cast<int64_t>(values[idx])`，防 int32 大 shape 乘加溢出
- 索引读宽分档：`THREAD_NUM = (sizeof(IDX_T) == 4) ? 1024 : 512`（sparse_segment_sum_grad）

## 5 infershape / op 定义侧

- 值依赖输入（shape/new_shape 类）三处声明缺一不可：`Input(...).ValueDepend(OPTIONAL)`、infershape 注册块 `InputsDataDependency`、tiling 注册块 `TilingInputsDataDependency`（详见 [../../../common/value-depend-process.md](../../../common/value-depend-process.md)）
- 输出 shape 由值决定时（IndexRemap），推导逻辑读输入值完成；空张量不在 infershape 端短路
- 越界报错类校验放 tiling 侧（`OP_CHECK_IF`），不放 kernel
