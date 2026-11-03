<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式切分模型

> **层级**：通用知识层（generic/），语言中立。
>
> 坐标系：N（索引条目数）为主切分坐标，innerSize（行/分组内元素数）为次坐标。

## 1 并行粒度选择

| 粒度 | 适用条件 | 实例 |
|------|---------|------|
| **元素级**（N × innerSize 全展开） | 条目间、行内元素间均独立 | sparse_apply_ftrl、sparse_bincount |
| **行级**（1 执行单元 = 1 索引条目，行内顺序处理） | 行内有数据依赖（状态更新）或行内复用一次索引查找 | sparse_apply_adadelta、sparse_segment_sum_grad |
| **条目级**（1 执行单元 = 1 索引，直接变换） | 纯索引变换，无伴随数据 | sparse_reshape |

判定法则：**行内存在跨元素依赖 → 行级；否则元素级**（最大化并行）。行级粒度下行内循环的空间局部性是收益来源（scatter-add 每条目只查一次索引，行内 inner 连续读）。

## 2 线程映射（grid-stride）

统一模式：执行单元以 `起始 + k×总执行单元数` 步进遍历职责区间，天然覆盖尾块：

```
for (idx = 区间起点 + 本单元编号; idx < 区间终点; idx += 总单元数)
```

- 每核先按核编号切出 `[coreStart, coreEnd)` 职责区间，区间内再按执行单元编号 grid-stride
- 职责区间起点 `coreStart >= N` 的核直接返回（空载核）
- 无 warp 级协作需求时执行单元完全独立，分支发散不影响正确性

## 3 核间切分（两步法）

```
Step 1  perCore = ceil(N / 物理核数)              # 物理核均分
Step 1' perCore < PER_CORE_MIN 时抬升到 PER_CORE_MIN # 下限保护，防小任务启动全核
Step 2  needCoreNum = ceil(N / perCore)            # 反推实际核数
```

- **PER_CORE_MIN 必须对齐到执行单元编组粒度**（如 32/warp）：否则尾编组空转；实测经验值 1024（sparse_bincount：误取 32 曾导致性能回退）
- 抬升后 needCoreNum 下降是**预期行为**：小 N 场景少核多干，避免调度开销
- `N = 0`（空稀疏结构）时仍须启动 ≥1 核：散射累积族的零初始化阶段不能省略

## 4 线程数与资源预算

| 算子负载 | 线程数取向 | 依据 |
|---------|-----------|------|
| 计算密集（含原子/分支） | 中档（512） | 平衡寄存器压力与并行度；减少同地址原子竞争 |
| 纯搬运/轻计算 | 高档（2048，平台上限） | 线程越多越能隐藏离散访存延迟 |
| 行级粒度 + 状态变量 | 中低档 | 行内状态占寄存器，线程数过高触发溢出 |

预算项：每执行单元栈空间（平台常量，如 64B/线程）× 线程数 ≤ 栈预算；局部状态变量（RowApply 的行内状态）× 线程数 ≤ 寄存器文件。线程数是**编译期常量**，随 TilingKey 场景分档，不进 TilingData。

## 5 切分输出清单

- [ ] 主坐标 N 的职责区间划分（每核 `[coreStart, coreEnd)`）
- [ ] PER_CORE_MIN 取值与编组对齐说明
- [ ] needCoreNum 公式（含 N=0 下限 1 核）
- [ ] 并行粒度判定（元素级/行级）及依据
- [ ] 线程数常量及寄存器/栈预算核算
- [ ] 尾块与空载核行为（grid-stride 覆盖证明）
