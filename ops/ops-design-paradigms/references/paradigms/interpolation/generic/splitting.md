<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式切分模型

> **层级**：通用知识层（generic/），语言中立。
>
> 坐标系：totalOut（输出元素总数）为正向主切分坐标；grad 反向以 grad_output 元素数为坐标、写侧按正向映射散射。

## 1 并行粒度选择

| 粒度 | 适用条件 | 实例 |
|------|---------|------|
| **输出元素级**（totalOut 全展开） | 输出逐位独立（本范式正向常态） | three_interpolate（B×N×C）、sync_resize_bilinear_v2、crop_and_resize（boxes×cropH×cropW×D） |
| **行/通道级**（1 执行单元 = 1 输出行） | 同行邻居可复用（坐标映射只做一次） | 输出 W 维为内循环、H 维为切分坐标；仅当 totalOut 小且坐标映射重时评估 |
| **ROI 级**（1 执行单元 = 1 框） | 框内顺序处理 | RoiSample 小框数场景；框间负载不均风险 |

判定法则：**默认输出元素级**（最大化并行，坐标映射为 O(1) 标量计算）；仅当 totalOut 过小或坐标映射含重量级值依赖（旋转/变形系数查表）时评估行级 / ROI 级。

## 2 坐标分解（线性索引 → 多维坐标）

输出元素级粒度下，执行单元持有的线性索引须分解为多维坐标 `(n, y, x, c)` 再做坐标映射：

- **快除**：`n = idx ÷ stride_n; r = idx mod stride_n; y = r ÷ stride_y; …`——整型除法/取模在执行单元上开销高，用 magic/shift 预计算参数把除法消为乘移（语言侧事实与参数布局见公共知识 [../../common/uintdiv-rules.md](../../common/uintdiv-rules.md)）
- **参数预计算**：magic/shift 由 host 侧按各维 stride 算好，经片上共享内存（`__ubuf`/`CAP_UB_HYBRID_SIZE` 内的小表）广播给全部线程；每维 2 参数
- **降维技巧**：相邻轴若对基址等价可合并消一次除法（如 (n×C+c) 与 index/(H×W) 等价时，N/C 轴不分解、只做 H/W 两次快除）
- **中间量类型**：索引与 stride 乘加统一宽整型（大 shape int64），防 int32 溢出

## 3 元数据传递

| 数据 | 形态 | 通道 |
|------|------|------|
| scale / extent / srcStart | 标量（≤ `CAP_VF_PARAM_MAX` 个） | 执行段入口直传 |
| 快除参数表 | 小数组（每维 2 项） | 片上共享内存广播 |
| 框坐标 / 索引 / 权重张量 | 大张量 | 值依赖读入（GM 直读，不预搬） |

标量超出入参上限或成表时走共享内存；体积核算 ≤ `CAP_UB_HYBRID_SIZE`。

## 4 线程数与资源预算

| 算子负载 | 线程数取向 | 依据 |
|---------|-----------|------|
| 坐标映射 + 加权采样（本范式常态） | 中档（512） | 坐标映射 + 快除 + 邻居索引占 ~20 活跃寄存器值，1024 线程有 spill 风险（实测保持 512） |
| 补偿累加路线 | 中低档 | df32 高低位字 + TwoSum 中间量进一步抬升寄存器压力 |
| 最近邻 / 极简采样 | 高档（平台上限） | 线程越多越能隐藏离散 gather 延迟 |

线程数为**编译期常量**，随 TilingKey 场景分档，不进 TilingData；与 `__launch_bounds__` 共用同一常量。

## 5 核间切分与 grad 散射

正向（两步法，与稀疏范式同构）：

```
Step 1  perCore = ceil(totalOut / 物理核数)
Step 1' perCore < PER_CORE_MIN 时抬升到 PER_CORE_MIN（对齐执行单元编组粒度）
Step 2  needCoreNum = ceil(totalOut / perCore)
```

- `totalOut = 0`（空 boxes / 空 N）仍启动 ≥1 核，kernel 侧短路返回；**正向无输出初始化需求**（逐位直写）
- 每核职责区间内执行单元编号 grid-stride 步进，天然覆盖尾块

grad 反向（散射累积骨架）：

- **两段顺序 launch**：输出（grad_input）零初始化 → 散射累积；同 stream 排序免核间栅栏（`CAP_SYNC_ALL` 备用：无法保证 stream 顺序时显式栅栏，`CAP_SYNC_SCHEDULE_MODE` 前提）
- 散射目标地址按正向坐标映射计算（每 grad_output 元素向 k ≤ 4 个邻居散射 `g × w`）；**贡献丢弃类越界**（coordinate-mapping §4）在散射前过滤
- 原子选择与确定性路径同 Sparse 范式散射累积族（浮点原子非确定、与竞品 GPU 对齐用容差；要求确定性时分桶预归约）
- 切分坐标仍是 grad_output 元素数（读侧规则），写侧冲突不由切分消除

## 6 切分输出清单

- [ ] 并行粒度判定（默认元素级；行/ROI 级须给负载核算）
- [ ] 坐标分解的轴清单与快除参数表布局
- [ ] PER_CORE_MIN 取值与编组对齐说明（ALIGN 给出具体取值与依据，禁止留占位注释）
- [ ] needCoreNum 公式（含 totalOut=0 下限 1 核）
- [ ] 线程数常量与寄存器压力核算（补偿路线单列）
- [ ] grad 路径的初始化/散射两段骨架与同步方式声明
