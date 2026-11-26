<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# SlidingWindow 范式场景模型

> **层级**：通用知识层（generic/），语言中立。
>
> 来源：max_pool_with_argmax_v3 / max_pool_grad_with_argmax_v3 / avg_pool / avg_pool_v2_grad 四个 AscendC SIMT 算子的实测蒸馏（ascend950 / arch35，ops-nn 仓），并参照 max_pool_v3_grad、dilation2_d、col2_im_v2、extract_image_patches 的生成与上库记录。

## 1 准入条件

本范式覆盖的算子需同时满足：

1. **整数规则窗口聚合**——窗口由规则参数（kernel / stride / pad / dilation，自适应池化由输出位置换算等效窗口）定义，窗口内元素地址由输出坐标**整数运算**推导；无浮点源坐标
2. **正向输出驱动 / 反向输入驱动**——正向每个输出位置独立枚举自己的窗口聚合，输出间无冲突；反向每个输入位置独立计算自己的梯度（反窗口枚举或路由散射）
3. **聚合函数是标量语义**——max / avg / lp-norm / sum / 取单元素（argmax 路由），窗口内逐元素聚合

## 2 术语与坐标系

| 术语 | 含义 |
|------|------|
| **kernel (kH, kW)** | 窗口尺寸 |
| **stride (sH, sW)** | 窗口滑动步长 |
| **pad (tPad/lPad/bPad/rPad)** | 输入四边填充（虚拟扩展，不实际搬运） |
| **dilation (dH, dW)** | 窗口内采样间隔（有效窗口跨度 `(k−1)·d+1`） |
| **窗口边界 [start, end)** | 输出位置 (ph, pw) 对应输入区间：`start = p·s − pad`，`end = min(start + (k−1)·d + 1, extent + pad)` 再双向裁剪 |
| **空窗** | 裁剪后 `start ≥ end` 的窗口（大 pad 配小输入时可出现） |
| **反窗口 [pStart, pEnd)** | 反向中覆盖某输入位置的输出位置集合（[window-math.md](window-math.md) §4） |
| **argmax mask** | 正向输出的 winner 线性索引张量，反向 argmax 路由的依据 |
| **聚合算子 ⊕** | 窗口聚合函数：max / avg / lp / sum / first |
| **outSize / inSize** | 输出 / 输入元素总数（正向 / 反向各自的切分主坐标） |

### 2.1 布局维度（NCHW / NHWC）

本范式覆盖 **NCHW 与 NHWC 两种 4D 布局**。布局是池化核的编码维度之一（[../adapters/ascendc/kernel-entrance.md](../adapters/ascendc/kernel-entrance.md)：SIMT 段内编码维度含布局（NCHW/NHWC）、索引 dtype 等）；仓内成熟池化包按双布局实现（[../adapters/ascendc/index.md](../adapters/ascendc/index.md)：avg_pool 一行载明「NCHW/NHWC 双布局」）。

两种布局下窗口枚举的访存形态（布局机械性质，可直接推导）：

| 布局 | 最内连续维 | 窗口枚举读形态 |
|------|-----------|---------------|
| NCHW | W | 同一通道内窗口沿 W 的元素地址连续，窗口读以 W 向跨度为主 |
| NHWC | C | 同一空间位置的 C 个通道地址连续；窗口相邻空间位置间隔 C 个元素，读按「空间步进 × C 跨度」进行；跨通道聚合的向量读天然连续（NHWC 聚合路线的访存优势） |

**路线关系**（仓内事实）：多路线成熟包中 NHWC 聚合占独立 TilingKey 号段，与 SIMD 小核 / 大核 / SIMT 并列分派（kernel-entrance.md）；本范式 SIMT 示例模板以 NCHW 为准（蒸馏来源为 NCHW 实测 kernel），NHWC 路线由 TilingKey 分派承接。切分与坐标分解对两种布局的适配见 [splitting.md](splitting.md) §2。

**待验证**：NHWC 布局下窗口枚举的搬运量与对齐量化影响未实测（当前无 NHWC 实测载体）；设计 NHWC 算子时须在产出中声明该假设并给出验证义务（口径同 [capability-contract.md](capability-contract.md)）。

## 3 算子类型与族划分

### 3.1 正向：WindowAggregate（输出驱动）

```
for 每个输出位置 p（并行）:
    窗口边界 = 窗口数学(p)               # window-math §2
    acc = ⊕ 初始化                       # max: −Inf；avg/sum: 0
    for h, w in 窗口（步进 dilation）:
        acc = 更新(acc, x[窗口元素])       # max: CycleUpdate；avg: sum +=
    y[p] = 收尾(acc)                      # avg: acc / 除子；max: acc（+ argmax mask）
```

聚合函数维度：

| ⊕ | 初始化 | 更新 | 收尾 | 伴随输出 | 实例 |
|---|--------|------|------|---------|------|
| max | −Inf | 严格大于才替换（NaN 优先传播） | 值 + winner 索引（argmax mask） | mask | max_pool_with_argmax_v3 |
| avg | 0 | 累加 | ÷ 除子（divisorOverride / countIncludePad / 有效数） | — | avg_pool |
| lp / sum | 0 / 1 | 幂和累加 | 开根 / 恒等 | — | global_lp_pool |

### 3.2 反向：GradRoute（输入驱动，三分支）

反向梯度回传按**路由信息**分三支（决策树见 [grad-routing.md](grad-routing.md)）：

| 分支 | 路由信息 | 冲突模型 | 实例 |
|------|---------|---------|------|
| **A 反窗口枚举** | 无需路由信息——每个输入位置枚举覆盖它的全部输出窗口 | 无冲突（输入驱动逐位直写） | avg_pool_v2_grad |
| **B argmax 路由** | 正向 argmax mask（winner 索引） | 无冲突（反窗口 + mask 检查，或 scatter 到唯一 winner） | max_pool_grad_with_argmax_v3 |
| **C 原子散射** | 无路由信息且不可反推（一般 stencil / 重算代价高） | 竞争写 → 原子累积 | max_pool_grad（重算型散射） |

**核心洞察**：池化反向不必走原子——换坐标系（输出驱动散射 → 输入驱动反窗口枚举）即可消解写冲突；分支 A/B 的选择依据是正向聚合是否携带路由信息（mask）。

## 4 数据流骨架

```
正向（WindowAggregate）:
  [段1: 输出坐标]     输出线性索引 → (n, c/ph, pw) 多维分解
  [段2: 窗口数学]     start/end 推导 + pad 双向裁剪 + 空窗短路
  [段3: 窗口枚举]     行间步进/行内步进 dilation；离散 GM 读
  [段4: 聚合]         标量 acc 更新（max 的 NaN 优先分支）
  [段5: 直写]         y[p] = 收尾(acc)；max 族同步写 argmax mask

反向（GradRoute·A/B）:
  [段1: 输入坐标]     输入线性索引 → (n, c, h, w)
  [段2: 反窗口]       pStart/pEnd 区间推导（窗口数学的逆运算）
  [段3: 路由检查]     B 支: topMask[输出位] == 本输入位 → 命中
  [段4: 贡献累加]     局部 gradient 累加（零原子）
  [段5: 直写]         y[index] = gradient
```

- 两方向均为**单段 kernel、逐位直写**：分支 A/B 无输出初始化、无栅栏
- 分支 C（原子散射）退化为 Interpolation grad / Sparse 散射同款骨架：零初始化 + 散射两段 launch + 原子（同步知识复用，不在本范式重复定义）

## 5 覆盖形态与口径说明

1. **全局池化**（global_max/avg_pool，窗口 = 全输入）是本范式 kH=H、kW=W 的退化形态，属本范式
2. **adaptive pool 的"按输出比例换算窗口边界"**仍是整数窗口（比率只进边界公式，不作邻居权重），属本范式覆盖口径

## 6 依赖谓词

| 谓词 | 用途 |
|------|------|
| `CAP_SIMT_GM_DIRECT` | 窗口元素离散读、反向梯度直写的前提 |
| `CAP_FP_NEG_INF` | max 聚合的 −Inf 哨兵初始化 |
| `CAP_FP_MAX_NAN` | 标量 max 的 NaN 传播语义（window-math §3） |
| `CAP_INT_DIV_COST` | 坐标分解直除 vs 快除的路线选择（splitting §2） |
| `CAP_SIMT_ATOMIC_ADD_DTYPE_SET` | 分支 C 原子散射的 dtype 域 |
| `CAP_SYNC_ALL` / `CAP_SYNC_SCHEDULE_MODE` | 分支 C 两段骨架同步 |
| `CAP_SIMT_THREAD_MAX` / `CAP_STACK_PER_THREAD` | 线程分档上界与栈预算 |
| `CAP_VF_PARAM_MAX` / `CAP_UB_HYBRID_SIZE` | 窗口参数标量入参 / 快除参数表体积 |

谓词定义见 [capability-contract.md](capability-contract.md)；各语言真值见对应适配层 capability-mapping。
