<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# SlidingWindow 范式 Kernel 入口与 Tiling 组织

> AscendC SIMT 的入口模式、TilingData/TilingKey 组织与 host 侧要点。以 max_pool_with_argmax_v3 / max_pool_grad_with_argmax_v3 / avg_pool 实测结构为基准。

## 1 入口结构

```cpp
extern "C" __global__ __aicore__ void max_pool_with_argmax_v3(
    GM_ADDR x, GM_ADDR y, GM_ADDR argmax, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipeBase;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);        // 或 KERNEL_TYPE_AIV_ONLY（grad）
    if (TILING_KEY_IS(SIMT_NCHW)) {                            // 多路线包按 TilingKey 分派
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3SimtTilingData, tilingData, tiling);
        MaxPoolWithArgmaxV3SimtNamespace::MaxPoolWithArgmaxV3<DTYPE_X, DTYPE_ARGMAX, format> op(tilingData);
        op.Init(x, y, argmax);
        op.Process();                                          // 宿主类 → asc_vf_call → __simt_vf__ 内核
    } else if (TILING_KEY_IS(...SIMD 路线...)) { /* 本范式不消费 */ }
}
```

- 宿主类形态：`Init(...) → Process()`，Process 内做 TilingData 搬 UB、快除参数预计算、`DataSyncBarrier` 后 `asc_vf_call`（[simt-path.md](simt-path.md) §1）
- `__simt_vf__` 内核不能是类成员函数：宿主类与内核函数同文件分离（类持 tiling/GlobalTensor，内核持裸指针）
- 正向多输出（y + argmax）：两个 GM 指针分别入参

## 2 TilingData 设计规则

| 规则 | 说明 |
|------|------|
| 字段 = host 可算标量 | 窗口参数（k/s/pad/dilation）、输入/输出维、count（outSize 或 inSize）、needCoreNum |
| **不含线程数** | 编译期常量，与 `__launch_bounds__` 共用（正向 256 / 反向 1024 分档属设计契约） |
| 快除参数不入 TilingData | host/宿主类侧算好经 UB 广播（每次 1~2 表） |
| 属性归一化在 host | padding_mode（SAME/VALID→数值 pad）、ceil_mode 在 tiling 侧解析为数值，kernel 只见数值 |
| 维度上限校验 | tiling 侧 `OP_CHECK_IF`（越界/非法属性前置报错），kernel 内分支只做空窗短路 |

host tiling 侧骨架（两步法同通用层 [../../generic/splitting.md](../../generic/splitting.md) §3；`SetBlockDim(needCoreNum)` + 空输入下限 1 核；分支 C 才配 `SetScheduleMode(1)`）。

## 3 TilingKey 编码规则

- **多路线包按 TilingKey 段分派路线**（SIMD 小核/大核/NHWC 聚合/SIMT 各占号段），本范式知识只覆盖 SIMT 段（[simt-path.md](simt-path.md) §4）；新设计优先采用 `ASCENDC_TPL_ARGS_DECL` 模板编程（avg_pool_v2_grad 先例），成熟包存量 `TILING_KEY_IS` 分派保留现状
- SIMT 段内编码维度：布局（NCHW/NHWC）、索引 dtype（int32/int64）、pad 有无、countIncludePad / divisorOverride（grad）
- dtype 不编码：`DTYPE_X` / `DTYPE_ARGMAX` 宏 + 模板参数编译期多实例
- 场景选段规则在 tiling 侧：通道数 / 窗口规模 / dtype 组合超出 SIMD 路线约束时落 SIMT 段——选段判据写入 tiling 设计文档

## 4 宿主类与快除参数

```cpp
// Process 内（max_pool_grad_with_argmax_v3 形态）
pipe_->InitBuffer(paramBuf_, PARAM_NUM * sizeof(DIV_T));          // 快除参数表进 UB
GetUintDivMagicAndShift<DIV_T>(magicHW, shiftHW, hW);              // host 侧预计算（每维 2 项）
...
DataSyncBarrier<MemDsbT::UB>();                                    // 写后同步
asc_vf_call<MaxPoolGradWithArgmaxNchw<...>>(
    dim3(THREAD_DIM), (__ubuf__ DIV_T*)paramBuf_.GetPhyAddr(), gradData, ...);
```

- 参数表布局按分解轴排列（HW/W/strideH/strideW），`__ubuf__` 指针入参广播给全线程
- 窗口标量（≤ `CAP_VF_PARAM_MAX`）直接入参，不进表

## 5 infershape / op 定义侧

- 正向输出 shape 由窗口参数推导（`out = floor((in + padL + padR − span)/s) + 1`，ceil_mode 差异在此归一）；反向输出 shape = orig_input shape（读 shape 推导，无值依赖）
- **argmax mask 是正向注册输出**（非 workspace/临时量）——反向算子以独立输入消费它；正反向 mask 索引口径（线性坐标）与 dtype（int32/int64）契约写进两算子规格
- 属性注册完整性：ksize/strides/pads/padding_mode/data_format/ceil_mode/divisorOverride 全量登记（**漏注册 ATTR 是 avg_pool_update 生成记录 dev_peer FAIL 实测事故**）
- **多语义 dtype 输出的 OpDef 全档注册**：输出带语义 dtype 档位（如 argmax 的 int32/int64）时，OpDef 该输出的 DataType 数组须按组合槽位覆盖全部档位，并与 InferDataType 档位、TilingKey 档位三方闭合——**漏档即默认路径接口契约不可满足**（max_pool_with_argmax_v3 生成记录评审阻塞项 D1：dtype 属性默认 int32 而 OpDef argmax 仅注册 int64）
- 空输入不在 infershape 端短路；tiling 侧下限 1 核、kernel 侧 count=0 短路
