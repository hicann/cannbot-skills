<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式 AscendC 语言适配层

> **层级**：AscendC 语言适配层（adapters/ascendc/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。
>
> 本适配层基于 **Ascend C SIMT 编程模型**（索引驱动离散访存场景的标准选择）。regbase（SIMD 向量）路线的通用开发方法见 [../../../general-methodology.md](../../../general-methodology.md)。

## 1 定位

本目录承载 Sparse 范式中**强耦合 AscendC SIMT** 的知识：入口与 tiling 组织、原子操作落地、索引数学实现、能力真值与陷阱。开发语言为 ascendc（由调用方明确提供）时，与通用知识层组合消费：

```
通用算法/模型（generic/）× AscendC SIMT 实现细节（本目录）→ SIMT 设计与代码产出
```

SIMT 基础 API（线程模型、访存函数、数学函数、调试接口）的官方文档位于 `$ASC_DEVKIT_DIR/docs/zh/api/SIMT-API/`；`ascendc-simt-best-practices` skill 提供 API 分类导航与实践总结，本适配层只展开 Sparse 范式特有的内容。

## 2 能力映射

本语言对通用层能力谓词的真值见 [capability-mapping.md](capability-mapping.md)。通用层算法中引用的谓词（原子 dtype 集合、线程上限、栈预算等）一律以该文件填值为准。

## 3 阅读路径

| 任务阶段 | 入口 | 内容 |
|------|------|------|
| Kernel 入口与 tiling | [kernel-entrance.md](kernel-entrance.md) | `__global__` 入口、TilingData/TilingKey 组织、`asc_vf_call` 实例化、host 侧 tiling 要点 |
| 原子操作落地 | [atomic-ops.md](atomic-ops.md) | 分 dtype 分发、double 的 CAS 模拟、返回值可靠性、幂等直写替代 |
| 索引数学 | [index-mapping.md](index-mapping.md) | 线性化/反线性化、-1 维推断、stride 表经 UB 传递、宽整型防护 |
| 陷阱与定位 | [pitfalls.md](pitfalls.md) | 零初始化竞争、原子 dtype 限制、PER_CORE_MIN 对齐、值依赖漏配等错题集 |

## 4 实测基准算子

本适配层结论蒸馏自以下 ascend950（arch35）实测算子，编码前可对照源码组织方式：

| 算子 | 算子族 | 关键特征 |
|------|--------|---------|
| sparse_bincount | Count | atomic 累积、值域裁剪、binary_output 幂等直写、DTYPE_ 宏模板 |
| sparse_segment_sum_grad | ScatterAdd | 零初始化 + SyncAll 两阶段、AtomicAddDispatch 分 dtype、double CAS 模拟 |
| sparse_apply_adadelta | RowApply | 1 线程 = 1 行、唯一索引无原子、标量参数走 VF 入参 |
| sparse_apply_ftrl | RowApply | 元素级粒度、powf 精度对齐 |
| sparse_reshape | IndexRemap | 值依赖 shape/new_shape、stride 预计算、-1 维推断 |

**内嵌样例包**：sparse_bincount（Count 族最小样例）已整包镜像至 [example/sparse_bincount_package/](example/sparse_bincount_package/)（op_graph/op_host/op_kernel/tests 完整结构；provenance 见 [example/README.md](example/README.md)）；其余基准算子按算子名在 CANN 组织开源算子仓（https://gitcode.com/cann，如 [ops-math](https://gitcode.com/cann/ops-math)、[ops-nn](https://gitcode.com/cann/ops-nn)）检索阅读。

## 5 公共知识依赖

| 主题 | 位置 |
|------|------|
| 值依赖处理 | [../../../common/value-depend-process.md](../../../common/value-depend-process.md) |
| 同步与一致性 | [../../../common/sync-and-consistency.md](../../../common/sync-and-consistency.md) |

## 6 消费规则

1. 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载其他语言适配层
2. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题
3. 能力谓词以 [capability-mapping.md](capability-mapping.md) 为真值；标注"待验证"的项作为假设输出并附验证义务
4. 编码前对照 §4 基准算子核实 API 用法；参考实现的具体参数须按当前算子规格重新核定
