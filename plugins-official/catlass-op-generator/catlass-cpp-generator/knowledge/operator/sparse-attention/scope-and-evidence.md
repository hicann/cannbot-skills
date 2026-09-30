---
type: operator
title: block_sparse_attention Arch22 范围与使用边界
description: 确认块稀疏注意力的输入域、知识读取顺序和实现验证条件。
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
sources: [{"id": "sparse-attention-scope-and-evidence-source", "resource": "audit:custody/sparse-attention#scope-and-evidence", "title": "scope-and-evidence 固定来源（提取侧独立保管）", "kind": "audit-record"}]
---

# 接口与概念

## 算子算法

`block_sparse_attention` 在稀疏 mask 保留的 KV 块上计算显式 scale×QKᵀ、行 softmax 和 PV，
属于块稀疏 softmax attention，不属于线性状态递推。本目录的接口约定限定于 DAV_2201 的
非量化 FP16/BF16、TND/BNSD 输入；它不是所有稀疏注意力接口的统一定义。

## 分核策略与基本块切分

区分 Q 稀疏块 X、KV 稀疏块 Y、Cube 微块、在线更新组和在途组数。
同一输出行必须有明确的 owner；扩大组宽或改变分核后，重新验证索引映射、尾部和状态归属。
不能仅根据硬件核数或任务数量推导负载均衡。

## 数据路径与存储层级

设计需要描述 mask→索引/计数、S、P、局部 O 及 m/l/alpha 的生产者、消费者与存活区间。
CATLASS 组件名称相近不表示接口、布局、数值精度或同步协议兼容。

## 流水排布、同步关系与数值精度

按 C1(QK)→V1(online softmax)→C2(PV)→V2(rescale/finalize) 检查同组依赖。
跨组交错需要独立保存对应组的 alpha 和 shape；O 与 LSE 必须分别验证。
具体精度模式见[语义与分派](semantics-and-dispatch.md)。

# 用法

| 当前问题 | 读取条目 | 需要确认的内容 |
| --- | --- | --- |
| 输入与数学定义 | [语义与分派](semantics-and-dispatch.md) | 布局、长度、mask、精度组合、未定义输入 |
| 调度与同步 | [任务与流水](scheduling-and-pipeline.md) | 行 owner、索引、在线组、slot 生命周期 |
| 组件选择 | [CATLASS 映射](catlass-components.md) | 实际特化能力、资源和事件归属 |
| Vector 实现 | [多行 epilogue](vector-epilogue-design.md) | 状态布局、广播、归约、尾部与工作精度 |
| 正确性验收 | [Golden 与验证](golden-and-validation.md) | 实际输入、参考实现、比较器、O/LSE |
| 运行与计时 | [执行与测量](execution-and-measurement.md) | 实载版本、异步缓冲、完整调用范围 |
| 优化选择 | [性能假设](performance-hypotheses.md) | 成本模型、候选条件、反例与配对验证 |

可通过 family 名称 `sparse-attention` 或别名 `block_sparse_attention` 查询本目录。

# 代码模式

在插件根目录初始化工程知识后，先查询摘要，再读取相关条目：

```bash
python skills/catlass-cpp-knowledge/scripts/record_knowledge.py query \
  --project-root <workspace> --family block_sparse_attention --arch DAV_2201 --compact
python skills/catlass-cpp-knowledge/scripts/record_knowledge.py get \
  --project-root <workspace> --path operator/sparse-attention/semantics-and-dispatch.md
```

# 约束

- 本目录不覆盖 Ascend950、量化、BSND 或 `innerPrecise=4`。
- 实现与验证应使用一致的输入输出约定、目标库版本和参考实现。
- 已有实现中的组宽、资源预算、核数和计时方式不能直接作为新工程常量。
- 本目录提供技术机制与验证方法，不代表当前候选已通过编译、精度或性能验收。

# 失败表现

将名称当作算法分类、把布局相近的 dense attention 直接套到稀疏接口、将旧版本的能力扩大到
新架构，都会造成范围误用。只验证 O 或只验证常见 shape 也不能证明整个输入域正确。

# 验证方法

先核对目标平台和输入域，再冻结实际输入、参考及判据。按条目列出的边界验证新实现，
记录真实加载版本和完整调用计时。公开结论应写明适用版本、条件及限制；
未经运行验证的优化应标为设计假设，并说明验证方法。

[^sparse-attention-scope-and-evidence-source]: 审计编号 sparse-attention-scope-and-evidence-source；固定来源与版本记录由提取侧独立保管，不随生成侧资料分发。
