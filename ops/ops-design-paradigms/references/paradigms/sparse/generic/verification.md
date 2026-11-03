<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式验证方法

> **层级**：通用知识层（generic/），语言中立。
>
> 覆盖 golden 设计、精度口径与边界用例；值依赖契约的验证义务同步列出。

## 1 golden 基准选择

| 输出语义 | golden 基准 | 理由 |
|---------|-----------|------|
| 浮点累积（原子） | 竞品 **GPU** 实现（atomicAdd） | 双方同为非确定实现，容差内可比；CPU 顺序实现舍入路径不同，逐位比对必然误报 |
| 整型累积 / 幂等直写 / 索引变换 | NumPy / CPU 参考实现 | 确定语义，可逐位比对 |
| 值依赖输出 shape（IndexRemap） | 参考实现同源推导（含 -1 维推断分支） | shape 与值都须比对 |

golden 脚本内输出缓冲以零初始化构造（与 kernel 零初始化契约对齐）。

## 2 精度标准（按 dtype 分档）

| dtype | 口径 |
|-------|------|
| int32 / int64 | 逐位一致 |
| float16 / bfloat16 | 最大相对误差 < 1e-3 |
| float32 | 最大相对误差 < 1e-6 |
| float64 | 最大相对误差 < 1e-12（若平台无原生 double 原子，CAS 模拟路径须单独验证） |

## 3 边界用例（必覆盖）

| 用例 | 验证点 | 对应算子族 |
|------|--------|-----------|
| **zero nnz**（N=0 / 空稀疏结构） | 输出全零（零初始化路径），不越界、不崩溃 | 全部 |
| **全密集**（nnz = numel，索引全空间覆盖） | 累积上界、原子高竞争路径 | Count / ScatterAdd |
| **重复索引**（同一目标多次累积） | 原子正确性：和守恒 | Count / ScatterAdd |
| **唯一索引**（无竞争路径） | 直写正确性 | RowApply |
| **越界索引** | 裁剪忽略 or host 报错，与竞品语义一致 | Count（忽略）/ ScatterAdd（报错） |
| **空伴随张量**（weights 为空 → 权重默认 1） | 伴随张量缺省分支 | Count |
| **幂等模式**（binary_output） | 直写替代原子，值恒 1 | Count |
| **-1 维推断**（new_shape 含一个 -1） | 推断公式、无 -1 与多 -1 报错 | IndexRemap |
| **大 innerSize × 小 N**（如 19994×50） | 行级粒度负载均衡、行内循环上限 | ScatterAdd / RowApply |
| **特殊值**（NaN / ±Inf / ±0.0） | IEEE 754 传播路径 | 累积族 |

## 4 性能验证要点

- **数据分布敏感**：稀疏算子性能与索引分布强相关（原子冲突率、行访问局部性）；性能用例须声明索引分布（均匀 / 重复聚簇 / 顺序）
- 对照点：小 N 抬升 PER_CORE_MIN 后的少核路径、大 N 全核路径分别测
- 原子冲突退化的诊断：对比"唯一索引数据集"与"重复索引数据集"的耗时差

## 5 值依赖契约验证

- 值依赖输入的声明（op 定义、infershape 注册、tiling 注册三处）逐处核对——缺一处即 host 侧拿不到数据
- 空张量不在 infershape 端短路：输出 shape 自然带 0 维，由 tiling/kernel 端处理
- 见公共知识 [../../common/value-depend-process.md](../../common/value-depend-process.md)

## 6 验证输出清单

- [ ] golden 口径（GPU / CPU）与精度分档已声明
- [ ] 边界用例表全覆盖（按命中的算子族）
- [ ] 索引分布声明齐备（均匀 / 聚簇 / 顺序）
- [ ] 值依赖三处声明核对通过
- [ ] "待验证"项（框架清零契约、CAS 位宽支持等）已列入验证义务
