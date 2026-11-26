<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# SlidingWindow 范式切分模型

> **层级**：通用知识层（generic/），语言中立。
>
> 坐标系：正向以 outSize（输出元素总数）为主坐标；反向以 inSize（输入元素总数）为主坐标——**方向决定坐标系**。

## 1 方向坐标系

| 方向 | 驱动坐标 | 遍历对象 | 写侧 |
|------|---------|---------|------|
| 正向 WindowAggregate | outSize | 输出位置 | 逐位直写 y[p]（无冲突） |
| 反向 GradRoute·A/B | inSize | 输入位置 | 逐位直写 y[index]（无冲突） |
| 反向 GradRoute·C | outSize（读侧） | grad_output 元素 | 散射累积（原子） |

- 分支 C 的读侧仍规则（grad_output 逐元素），写侧冲突不由切分消除——与 Interpolation grad 同构

> **A/B/C 指什么**（对照 [grad-routing.md](grad-routing.md) 反向路由三分支）：反向的 A/B/C 不是三种坐标系，而是反向梯度的三种路由分支，按「写侧是否冲突 / 是否有路由信息」选择：
> - **A 反窗口枚举**——每个输入位置独立枚举覆盖它的输出窗口（avg_pool_v2_grad 型），输入驱动、无冲突；
> - **B argmax 路由**——用正向 argmax mask 把梯度只送到 winner 位置（max_pool_grad_with_argmax_v3 型），输入驱动 + mask 检查、无冲突；
> - **C 原子散射**——无路由信息且不可反推时按 grad_output 元素散射累积，读侧输出驱动、写侧需原子。
> 上表中 A/B 共用 inSize 驱动坐标、C 用 outSize 读侧坐标，即由此而来。
- 空输出 / 空输入（size=0）仍按 1 核下限发起、kernel 侧短路；分支 C 的零初始化段不可省

## 2 坐标分解

输出/输入线性索引 → 多维坐标 `(n, c, ph, pw)`：

- **快除**：magic/shift 把逐元素除法消为乘移，参数经片上共享内存广播（每维 2 参数，依赖 `CAP_UB_HYBRID_SIZE`）；参数求法与限制见公共知识 [../../common/uintdiv-rules.md](../../common/uintdiv-rules.md)
- **直除取舍**：`%` / `/` 直写依赖 `CAP_INT_DIV_COST`——反向（THREAD_DIM 高、元素多）实测用快除（max_pool_grad_with_argmax_v3：8 参数表）；正向 THREAD_DIM 低、除法次数相对少时直除可接受（max_pool_with_argmax_v3 直除先例）。热点算子按分档实测后定，不在设计期拍脑袋
- **降维**：首维可整除时商直接就是外层坐标（`index / (H·W)` 一次快除消 n+c 两轴）；NCHW 与 NHWC 的分解顺序不同（c 在外 vs c 在内），布局模板分派各自维护
- **布局连续性与线程载荷**：NCHW 下 W 为最内连续维，按输出/输入位置切线程时单线程的窗口读沿 W 连续；NHWC 下 C 为最内连续维，窗口读跨空间位置时按 C 跨度步进、跨通道向量读连续——线程映射的最优连续载荷维度随布局不同（场景与路线背景见 [scene.md](scene.md) §2.1；NHWC 聚合占独立 TilingKey 号段，见 [../adapters/ascendc/kernel-entrance.md](../adapters/ascendc/kernel-entrance.md)）
- **待验证**：NHWC 载荷映射的搬运量与对齐影响未实测，设计期按上述机械性质论证并在产出中声明验证义务
- 中间量宽整型承载（大 shape 防 int32 溢出）

## 3 核间切分（两步法）

```
Step 1  perCore = ceil(total / 物理核数)
Step 1' perCore < PER_CORE_MIN 时抬升（对齐执行单元编组粒度）
Step 2  needCoreNum = ceil(total / perCore)     # 下限 1 核
```

- 职责区间内 grid-stride 步进，尾块天然覆盖；空载核直接返回
- 正反向坐标系不同但公式同构——同一算子族正反向 tiling 代码可复用骨架

## 4 线程数与资源预算

| 负载 | 线程数取向 | 依据 |
|-----|-----------|------|
| 正向窗口聚合（max：−Inf + CycleUpdate 分支 + mask 写） | 中低档（256） | 窗口循环 + 分支更新占寄存器；max_pool_with_argmax_v3 实测 256 |
| 反向反窗口枚举（分支 A/B） | 高档（1024） | 梯度累加器轻、无效扫描占比高，线程并行隐藏分支开销；max_pool_grad_with_argmax_v3 / avg_pool_v2_grad 实测 1024 |
| 分支 C 原子散射 | 中档（512） | 同 Interpolation/Sparse 散射档 |

线程数为编译期常量，与 `__launch_bounds__` 共用；正反向同算子不同档（256/1024）是**预期形态**，分档依据写入设计文档。

## 5 元数据传递

| 数据 | 形态 | 通道 |
|------|------|------|
| 窗口参数（k/s/pad/dilation，≤ 14 标量） | 标量 | 执行段入口直传（`CAP_VF_PARAM_MAX` 预算内） |
| 快除参数表（每维 2 项 × 分解维数） | 小数组 | 片上共享内存广播，写后 UB 同步再进 kernel |
| argmax mask / grad 张量 | 大张量 | GM 直读（值依赖，运行期） |
| 除子 / countIncludePad / divisorOverride | 标量 | 入参或 tiling 字段 |

## 6 切分输出清单

- [ ] 方向坐标系声明（正向 outSize / 反向 inSize）
- [ ] 坐标分解路径（快除表布局 vs 直除）与依据
- [ ] PER_CORE_MIN 取值与 needCoreNum 公式（含空输入下限 1 核）
- [ ] 线程分档（正反向不同档须注明依据）
- [ ] 分支 C 的两段 launch 与同步方式
- [ ] 尾块与空载核行为（grid-stride 覆盖证明）
