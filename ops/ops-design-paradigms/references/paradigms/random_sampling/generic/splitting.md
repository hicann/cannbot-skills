<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling 范式切分模型

> **层级**：通用知识层（generic/），语言中立。
>
> 坐标系：L（输出规模）为主坐标；切分同时承担**负载划分**与**流区间划分**双重职责。

## 1 输出空间均分（向量路线）

```
normalPer = ceil(L / 总单元数)，向上对齐到 a（单元对齐粒度，下限 L_min）
usedUnits = ceil(L / normalPer)，空输出时下限 1
tailPer   = L − normalPer × (usedUnits − 1)
```

- 对齐粒度 a 与下限 L_min 防小任务过度切分（实测基准 a=4 元素、L_min=256）
- 每单元的片上缓冲按存活节点数均分并按块大小对齐（缓冲预算 = 可用容量 − 共享暂存 − 混合编程扣减）

## 2 执行单元与流对齐（线程路线）

每输出消耗 c、并行单元数 B 时的两种布局：

```
连续布局（g = c·B）：  单元 p 处理 [p·g, (p+1·g)) 内的连续 c 元素组
    流位置(i) = ⌊i / g⌋，组内偏移 = i mod g 对应单元内的 c 连续槽位
离散布局（g = B）：    单元 p 处理 i ≡ p (mod B) 的散布元素
    流位置(i) = ⌊i / (c·B)⌋，组内取第 (i mod B) 路
```

| 布局 | 适用 | 特性 |
|------|------|------|
| 连续 | 每输出消耗多个随机数（c>1）且需整组消费 | 每单元一次产出覆盖连续输出段 |
| 离散 | c=1 或组内散布写 | 输出散布，与展开因子 U 组合隐藏延迟 |

- **展开因子 U**：每单元每轮处理 U 个散布元素（间隔 B），减少流推进次数与循环开销（实测基准 U=4）
- 除法 `i/B` 用预计算魔数+移位替代（除法慢于乘法一个量级）
- 布局选择错误不破坏不变量 I1/I2（见 [prng-model.md](prng-model.md) §3），只影响性能——但**布局决定流位置公式**，实现必须与推导一致

## 3 大输出空间切分（块协议）

触发条件：线性索引或最大字节偏移超出索引位宽 W（如 32bit 限 2^31−1）：

```
1. 合并连续维（去 1 维、合并 stride 连续维）得到紧凑 (shape, strides)
2. 递归二分：选字节跨度最大的维对半切，直至每块满足位宽约束（块数上限 M）
3. 每块独立推导 (元素数, GM 偏移, 并行单元数, 流起点 offset_b)
4. 块间 offset 累进（流不重叠），块内按 §2 布局
```

- 超过块数上限 M 仍不满足位宽 → 报错（不静默截断）
- 每块的并行单元数独立推导（块规模不同），块内 usedUnits 不做全局均衡——块协议的正确性优先

## 4 短路口径的 TilingData 填充语义（R18，范式裁决）

空 tensor / 防御路径（N=0 不下发 kernel 或图模式兜底）下 TilingData 各字段的填充口径**必须无歧义**（设计文档与 UT 预言机双源冲突会死锁实现流程）：

| 字段类 | 裁决 |
|--------|------|
| **计数字段**（totalNum / blockFactor / blockDim 等） | 按 N 语义清零 / 置 1 |
| **平台派生量**（ubFactor / buffer 尺寸等） | **无条件按公式填充**（与公式节的自然推导一致，UT 预言机据此生成） |

裁决理由：UT 预言机从公式节推导必然得到"无条件填充"；设计文档若另立"置 0"口径即产生双源歧义。范式默认即消除歧义，偏离须在设计文档与 UT 同步声明。

## 5 单次产出量上界（R10）

流生成接口的 count 形参承载位宽有限（实测 16bit）：**单次批量产出量必须 < 65535**（按元素数折算），超限**静默截断**（不报错）——切分时 ubFactor/单块产出量的上界约束必须显式核算。

## 6 切分输出清单

- [ ] 主坐标 L 的单元划分（normalPer/tailPer/usedUnits，含对齐与下限）
- [ ] 片上缓冲预算（存活节点数均分 + 对齐 + 混合编程扣减；RNG 暂存 K 份折算见适配层 common-template §2）
- [ ] 布局选择（连续/离散）与展开因子 U、流位置公式一致性
- [ ] numPerCore 组对齐（k 粒度必要条件，prng-model §3）
- [ ] 单次批量产出量 < 65535 核算（§5）
- [ ] 索引位宽 W 判定；超限时的块协议（上限 M、报错路径）
- [ ] 空输出（L=0）下限路径（至少 1 单元）+ 短路口径填充语义（§4）
- [ ] 每块流起点 offset_b 累进表
