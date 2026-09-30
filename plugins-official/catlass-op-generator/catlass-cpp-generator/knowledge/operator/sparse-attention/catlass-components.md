---
type: operator
title: block_sparse_attention Arch22 CATLASS 组件映射
description: 按稀疏映射、实际模板能力、资源和数值契约选择 QK/PV 与非 GEMM 组件。
tags:
- catlass-cpp
- sparse-attention
- block_sparse_attention
- BlockSparseAttention
- 块稀疏注意力
- arch22
operator_families:
- sparse-attention
architectures:
- DAV_2201
status: stable
generated: {"by": "process:sparse-attention-knowledge-port"}
verified: [{"by": "process:cross-source-check"}]
sources: [{"id": "sparse-attention-catlass-components-source", "resource": "audit:custody/sparse-attention#catlass-components", "title": "catlass-components 固定来源（提取侧独立保管）", "kind": "audit-record"}]
---

# 接口与概念

## 算子算法

QK 和 PV 是矩阵计算节点；mask 压缩、online softmax、rescale 和 LSE 是独立逻辑。
CATLASS 的 dense attention 组件可提供机制参考，不能直接代表块稀疏接口兼容。

## 分核策略与基本块切分

以 batch/head/X 块中的 Q 行 owner 为任务边界，将逻辑 KV 流拆成合法矩阵微块。
在线组扩宽之前，确认模板是否实现多子块处理、PV 累加及末次 Fixpipe。
单 MMAD 特化不会仅因 actualShape 变大就自动获得多块能力。

## 数据路径与存储层级

可用自定义 TileCopy 将非连续 KV 片段装入连续 L1，或对连续保留片段分别调用矩阵组件。
两种方案都需要验证 Y 块展开、真实 stride、末块有效长度以及 QK/PV 映射一致性。
中间 gathered K/V 若实际落盘，其搬运与 workspace 必须计入成本。

## 流水排布、同步关系与数值精度

按真实构造/析构检查 Resource、HardEvent 与 CrossCoreFlag 的所有权。
QK/PV 共享 L0 时联合证明释放条件；softmax/rescale 共享状态时联合检查布局和代际。
普通 no-mask softmax 的名称不保证满足指定的低精度及 LSE 契约。

# 用法

先读取通用 [BlockMmad](../../catlass/block-mmad.md) 和 [Epilogue](../../catlass/epilogue.md)，
再检查目标 CATLASS 版本的实际模板特化。下表是检索入口，不承诺所有版本具有相同签名。

| 需求 | 组件或职责 | 选择时核对 |
| --- | --- | --- |
| QK | `BlockMmad` / `TileMmad`；FAQK/FAIQK 系列策略 | TND 的 Q stride、稀疏 K 装载、S stride、Q 驻留与合法微块 |
| PV | FAPV/FAIPV 系列策略或支持累加的 BlockMmad | 输入 dtype 的 P、同一 KV 流、首片初始化/后片累加/末次写回 |
| KV 装载 | `TileCopy` 或索引装载封装 | paged 索引并非 sparse mask，需明确地址映射转换 |
| mask 压缩 | 自定义 Block/Tile | 有序 idx/count、无效位置、每次调用更新 |
| online softmax | `EpilogueAtlasA2OnlineSoftmax` 等机制参考 | 实际状态/slot 布局、真实列归约、P cast、精度分支 |
| 输出 rescale | `EpilogueAtlasA2RescaleO` 或自定义 Block/Tile | alpha 代际、O 累加顺序、最后归一化和 LSE |
| 行广播 | `TileBroadcastOneBlk` / `TileOneBlkColumnBroadcastMul` | 紧凑向量与整32B广播块契约、静态形状和有效行 |
| 调度/同步 | 自定义 scheduler 与实际资源/flag API | 稀疏负载、首尾排空、零行子核、事件计数 |

# 代码模式

```text
CATLASS BlockMmad / TileMmad: QK, PV
operator-owned Block / Tile: mask-to-index, sparse TileCopy,
                             online state, rescale, finalize / LSE
```

这是职责划分，不是可直接编译的实例化代码。QK/PV 使用 CATLASS 矩阵组件，
稀疏索引与 epilogue 由算子侧的 Block/Tile 组织。

# 约束

- 检查 dtype/layout/TileShape 对应的实际定义，只有 forward declaration 不代表支持。
- dense FA 的固定 workspace、PAGED 分支或布局转换需与稀疏输入的接口约定分别核对。
- P/V 的分块布局需使用完整父 layout 计算子片偏移，不能按连续行主序猜测 zN/nZ 地址。
- AB 双缓冲与 C 累加槽独立管理；同一事件的初始化、消费和释放只归一个 owner。
- 更宽的 L1 装载可能降低搬运次数，也会增加资源与控制开销；容量合法不等于性能更好。

# 失败表现

重复预置已由组件拥有的事件、单 MMAD 超容量、P/V 稀疏映射不同、把首 lane 有效的
32B cell 当作完整广播块，都可能导致卡死或错误结果。接口名相同不能排除这些问题。

# 验证方法

分别验证组件实例化、矩阵累加、稀疏映射和 epilogue 数值，再验证组间同步。
覆盖 FP16/BF16、D64/128、TND 多 head、Q/KV tail、非连续保留块、多微块累加和 LSE。
通用示例通过只能确认组件与构建环境，不能代替算子参考和完整调用测量。

[^sparse-attention-catlass-components-source]: 审计编号 sparse-attention-catlass-components-source；固定来源与版本记录由提取侧独立保管，不随生成侧资料分发。
