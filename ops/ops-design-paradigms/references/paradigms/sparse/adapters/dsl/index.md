<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式 CANNBotDSL 语言适配层

> **层级**：CANNBotDSL 语言适配层（adapters/dsl/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。
>
> 本适配层基于 **CANNBotDSL SIMT 编程模型**（`vf(mode="simt")`，索引驱动离散访存的标准选择），调度密集场景叠加 **AI CPU metadata 模式**（`@aicpu_kernel` 分核计划外置）。权威依据：cannbot-dsl 仓（开源：https://gitcode.com/cann/cannbot-dsl ，docs/zh/guide/ 与 samples/，DSL 0.8.0，Ascend 950PR/950DT）。

## 1 定位

本目录承载 Sparse 范式中**强耦合 CANNBotDSL（下文简称 DSL）**的知识：入口与编译组织、原子操作落地、索引数学实现、能力真值与陷阱。开发语言为 `dsl` 时，与通用知识层组合消费：

```
通用算法/模型（generic/）× DSL 实现细节（本目录）→ DSL 设计与代码产出
```

DSL 与 AscendC 适配层的三个范式级差异（详见各叶子文档）：

| 差异 | DSL 形态 | 影响 |
|------|---------|------|
| 构图期 host **不能读运行期 GM 值** | 值依赖输入（shape/new_shape 类）无法在 host tiling 读取；改走 AI CPU（运行期可读 GM）或设备侧自算 | [kernel-entrance.md](kernel-entrance.md) §5 |
| 同 stream 顺序 launch 天然有序 | 零初始化 + 散射两阶段可拆两次 launch（stream 排序免 SyncAll）；单 kernel 内跨段依赖才用 `global_sync_all()` | [kernel-entrance.md](kernel-entrance.md) §4 |
| 无 TilingData/TilingKey 载体 | 元数据经 **params Tensor（GM int64 数组）+ 手工 ABI** 传递；路径分派 = host 侧 Python 选择 + 独立编译实例 | [kernel-entrance.md](kernel-entrance.md) §2/§3 |

## 2 能力映射

本语言对通用层能力谓词的真值见 [capability-mapping.md](capability-mapping.md)。通用层算法中引用的谓词（原子 dtype 集合、线程上限、UB 预算等）一律以该文件填值为准。

## 3 阅读路径

| 任务阶段 | 入口 | 内容 |
|------|------|------|
| Kernel 入口与编译 | [kernel-entrance.md](kernel-entrance.md) | `@kernel`/`@host` 结构、params Tensor ABI、`cb.compile`+`lru_cache` 编译实例、launch 几何、值依赖处理 |
| 原子操作落地 | [atomic-ops.md](atomic-ops.md) | `cb.simt.atomic_add` dtype 边界、`mem_copy(atomic_add=True)` 批量累积、CAS 模拟、幂等直写 |
| 索引数学 | [index-mapping.md](index-mapping.md) | int64 全程承载、stride/shape 表 ABI、host 预计算、越界防护 |
| 陷阱与定位 | [pitfalls.md](pitfalls.md) | UB 预算扣减、Hardware Loop 退化、launch 形态校验、ABI 契约漂移等错题集 |

## 4 基准样例（cannbot-dsl 仓实测）

本适配层结论蒸馏自以下 cannbot-dsl 样例（路径相对仓根 `samples/`），编码前可对照源码组织方式：

| 样例 | 算子族 | 关键特征 |
|------|--------|---------|
| scatter_elements ★ | Scatter/scatter-add | 纯 SIMT 数据面、params Tensor ABI（MAX_RANK stride 表）、多路径 host 分派、atomic_add float32、DMA 初始化 + SIMT 散射两段 launch |
| gather_v2 | Gather | 索引驱动离散读、host 分档 |
| sparse_flash_attention（+ metadata 系） | BlockSparse | 稀疏候选块调度经 AI CPU metadata（`@aicpu_kernel` 分核计划） |
| quant_block_sparse_attn / sparse_flash_mla | BlockSparse | 量化 + 稀疏块注意力，metadata 复用 |

★ = 新算子接入前必读对照（SIMT 数据面完整先例）。

**内嵌样例**：与 AscendC 侧同算子对齐的 DSL 结构草稿见 [example/sparse_bincount_simt_draft/](example/sparse_bincount_simt_draft/)（**DRAFT 未验证**——Count 算子族骨架：两段 launch、路径分派、原子/幂等直写；语义契约对照 [../ascendc/example/sparse_bincount_package/](../ascendc/example/sparse_bincount_package/)，实现载体按语言）。§4 表中 scatter_elements 等源仓样例仍在 cannbot-dsl 仓内阅读（编码前对照），不再内嵌镜像。

调度密集（分核依赖张量内容）场景的 AI CPU metadata 模式详见 cannbot-dsl `docs/zh/guide/aicpu.md` 与 `samples/flash_attn/flash_attn_metadata.py`。

## 5 公共知识依赖

| 主题 | 位置 |
|------|------|
| 值依赖处理 | [../../../common/value-depend-process.md](../../../common/value-depend-process.md)（DSL 差异见 kernel-entrance §5） |
| 同步与一致性 | [../../../common/sync-and-consistency.md](../../../common/sync-and-consistency.md)（DSL 差异见 kernel-entrance §4） |

## 6 消费规则

1. 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载 ascendc 适配层、不参考其 API 与代码模板
2. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题
3. 能力谓词以 [capability-mapping.md](capability-mapping.md) 为真值；标注"待验证"的项作为假设输出并附验证义务
4. 编码前对照 §4 基准样例核实 API 用法；样例的具体参数（tile 字节、线程数）须按当前算子规格重新核定
5. DSL 权威资料：cannbot-dsl 仓（开源：https://gitcode.com/cann/cannbot-dsl ）docs/zh/guide/（编程模型）与 docs/zh/api/（API 详情）；引用接口时以目标版本 API 文档为准
