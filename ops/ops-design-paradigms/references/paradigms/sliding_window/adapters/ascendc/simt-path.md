<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# SlidingWindow 范式 SIMT 实现路线

> AscendC SIMT 下正向窗口聚合与反向路由的落地。以 max_pool_with_argmax_v3（正向）/ max_pool_grad_with_argmax_v3（反向）/ avg_pool（正向均匀聚合）实测结构为基准；算法模型见 [../../generic/window-math.md](../../generic/window-math.md) 与 [../../generic/grad-routing.md](../../generic/grad-routing.md)。

## 1 正向：输出驱动窗口聚合

```cpp
// max_pool_with_argmax_v3（THREAD_DIM 256，直除分解）
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void MaxPoolForwardNchw(
    const int64_t count, const __gm__ scalar_t* bottomData, ..., __gm__ scalar_t* topData,
    __gm__ idx_scalar_t* topMask, int blockIdxParam, int blockNum)
{
    for (idx_accscalar_t index = blockIdxParam * blockDim.x + threadIdx.x; index < count;
         index = index + blockNum * blockDim.x) {
        // 坐标分解（直除形态）
        idx_accscalar_t pw = index % outputWidth;
        idx_accscalar_t ph = (index / outputWidth) % outputHeight;
        idx_accscalar_t nxc = index / outputWidth / outputHeight;
        // 窗口边界（window-math §2）
        idx_accscalar_t hstart = ph * strideH - padH;
        idx_accscalar_t hend = (hstart + (kernelH - 1) * dilationH + 1) < height
                                    ? (hstart + (kernelH - 1) * dilationH + 1) : height;
        while (hstart < 0) { hstart += dilationH; }   // 负起点按 dilation 推进（非 clamp）
        // 聚合（−Inf 哨兵 + NaN 优先）
        float maxval = *reinterpret_cast<const float*>(&F32_NEG_INF);
        idx_accscalar_t maxidx = hstart * width + wstart;
        for (h = hstart; h < hend; h += dilationH)
            for (w = wstart; w < wend; w += dilationW)
                CycleUpdate(val, h * width + w, &maxval, &maxidx);   // (val>maxval)||isnan(val) 才更新
        topData[index] = static_cast<scalar_t>(maxval);              // 双输出：值 + winner 线性索引
        topMask[index] = static_cast<idx_scalar_t>(maxidx);
    }
}
```

- **avg 族差异**：acc=0 累加 → `sum / divisorFactor`（divisorOverride / countIncludePad / 有效面积三分档，收尾除法一次）；无 mask 输出
- 坐标分解的快除形态（avg_pool / 反向）：`Simt::UintDiv<DIV_T>(index, magic, shift)` 逐轴剥离，参数表经 UB 广播（[kernel-entrance.md](kernel-entrance.md) §4）
- 核数经 `blockIdxParam` / `blockNum` 标量入参（宿主类 `GetBlockIdx()` / `GetBlockNum()` 计算），与 `__simt_vf__` 内建 blockIdx 并存

## 2 反向分支 A/B：输入驱动反窗口

```cpp
// max_pool_grad_with_argmax_v3（THREAD_DIM 1024，UintDiv 8 参数表）——分支 B argmax 路由
for (index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x) {
    // 输入坐标分解（快除）: index → (nc, h, w)
    // 反窗口区间（window-math §4）
    phstart = PStart(h, padH, kernelH, dilationH, magicStrideH, shiftStrideH);
    phend   = PEnd(h, padH, hOutDim, magicStrideH, shiftStrideH);     // PEnd 内含 poolSize 封顶
    // argmax 路由检查 + 贡献累加（零原子）
    float gradient = 0;
    for (ph = phstart; ph < phend; ++ph)
        for (pw = pwstart; pw < pwend; ++pw) {
            idx = ph * wOutDim + pw + offset;
            if (topMask[idx] == h * wDim + w)                         // winner 命中检查
                gradient += static_cast<float>(bottomData[idx]);
        }
    topData[index] = static_cast<VALUE_T>(gradient);                  // 逐位直写
}
```

```cpp
// PStart/PEnd：反窗口端点（整数域快除）
FORMAT_TYPE PStart(size, pad, kernel, dilation, magicStride, shiftStride) {
    if (size + pad < ((kernel - 1) * dilation + 1)) return 0;          // 无覆盖窗口
    return UintDiv(size + pad - ((kernel - 1) * dilation + 1), ...) + 1;
}
FORMAT_TYPE PEnd(size, pad, poolSize, magicStride, shiftStride) {
    pEnd = min(UintDiv(size + pad, ...) + 1, poolSize);
}
```

- **分支 A 差异**（avg_pool_v2_grad）：路由检查换成 `gradient += grad[窗口位] / 除子`（除子口径与正向同值）；无 mask 输入
- 反窗口枚举的无效扫描（mask 不命中）靠 1024 线程并行隐藏；命中检查是纯整数比较，无分支发散代价（发散可承受）
- argmax mask 的索引口径：批次内线性索引 `h·W + w`，int32/int64 按 TilingKey 分档——正反向必须同口径同 dtype

## 3 反向分支 C：原子散射

- 骨架与 Interpolation grad / Sparse 散射同构：零初始化段（幂等常量写）+ 散射段（`asc_atomic_add`）两段顺序 launch；同 stream 排序免栅栏，否则 `SyncAll()` + `SetScheduleMode(1)` 成对（**漏配 SetScheduleMode(1) 是 max_pool_v3_grad 生成记录 design_peer HIGH 实测事故**）
- 散射目标按正向窗口数学推（每 grad_output 元素向 k² 个窗口元素散射，或经 mask 只向 winner 散射）
- 重算型路由（orig_input/orig_output 重推 winner）在散射前完成，避免散射段二次读

## 4 SIMT 路线的包内定位与自含组织

成熟算子包是多路线的（SIMD 小核 / 大核 / NHWC 聚合 / SIMT），由 TilingKey 分派：

- **本范式只登记 SIMT 路线**（[scene.md](../../generic/scene.md) 前置约束）；既有包内非 SIMT 路线文件的设计知识不在本范式消费范围
- 既有包内 SIMT 路线的定位方式：TilingKey 段（max_pool_with_argmax_v3 的 `500001/500002/500011/500012` SIMT 段）+ schMode 模板参数（TPL_SIMT_KERNEL）；tiling 侧按场景（大 C / 小 C / SIMT 兜底）选段；SIMT 段的选型特征：通道数极端、dtype 组合冷门、窗口参数超出 SIMD 路线约束的兜底路线
- **新算子设计不复制该形态**：SIMT 单路线自含组织即可——TilingData 就地单文件定义、TilingKey 用 `ASCENDC_TPL_ARGS_DECL` 模板编程（[example/](example/) 模板包示范；多算子共享 common 库是成熟仓的工程演进形态，按需再抽，不预先建）

## 5 设计输出清单

- [ ] 正/反向类型判定与路由分支（A/B/C）
- [ ] 坐标分解形态（直除 vs UintDiv 参数表）与线程分档依据
- [ ] 窗口/反窗口公式与 [../../generic/window-math.md](../../generic/window-math.md) 同源核对
- [ ] 聚合语义（−Inf / NaN / tie-break / 除子三分档）声明
- [ ] mask 索引口径与 dtype 契约（正向伴随输出 ↔ 反向路由）
- [ ] 分支 C 的两段骨架与同步成对项
