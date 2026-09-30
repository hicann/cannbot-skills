---
type: operator
title: block_sparse_attention Arch22 语义与十二分派
description: 定义非量化块稀疏注意力的布局、长度、mask、精度组合及 O/LSE 边界。
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
sources: [{"id": "sparse-attention-semantics-and-dispatch-source", "resource": "audit:custody/sparse-attention#semantics-and-dispatch", "title": "semantics-and-dispatch 固定来源（提取侧独立保管）", "kind": "audit-record"}]
---

# 接口与概念

## 算子算法

对于 batch b、Q head h、Q token s，稀疏行号为 `floor(s/X)`，KV head 为
`floor(h/(Nq/Nkv))`。mask 的 1 表示保留一个长度为 Y 的 KV 区间；只计算位于有效 KV
长度内的 token。scale 是显式输入，不强制为 `1/sqrt(D)`。

对非空保留集合 J：

```text
score_j = scale * dot(q, k_j)
m = max(score_j for j in J)
l = sum(exp(score_j - m) for j in J)
O = sum(exp(score_j - m) * v_j for j in J) / l
LSE = m + log(l)
```

O 与 Q 的布局和 dtype 一致；开启时 LSE 为 FP32，形状对应 Q 的 D 轴替换为 1。
输出 dtype 不代表所有中间运算都使用该精度。

## 分核策略与基本块切分

X 决定 Q 稀疏行，Y 决定保留的 KV 区间。X>0，无 128 对齐要求；Y 为 128 的正倍数。
稀疏块尺寸不等于 Cube 微块尺寸；任务不得跨越具有不同保留集合的 Q 稀疏行。
同一 KV head 对应的多个 Q head 仍可能具有不同的 mask。

## 数据路径与存储层级

| 项目 | 本条目的输入约定 |
| --- | --- |
| dtype / D | Q/K/V dtype 相同，为 FP16 或 BF16；D 相同且为 64 或 128 |
| heads | Nkv>0，Nq 为 Nkv 整数倍，K/V head 数一致 |
| layout | Q/KV 同为 TND 或同为 BNSD |
| TND | Q `[ΣSq,Nq,D]`，K/V `[ΣSk,Nkv,D]`；两份长度列表同时提供且等长，各项是 batch 长度，不是前缀和；各列表和等于对应 T |
| BNSD | Q `[B,Nq,Sq_max,D]`，K/V `[B,Nkv,Sk_max,D]`；长度列表同时提供或同时省略，省略表示满长；有效长度不超过物理 S |
| mask | `[B,Nq,ceil(Sq_max/X),ceil(Sk_max/Y)]`，值为 0/1；TND 的 max 来自长度列表，BNSD 的 max 来自物理 shape |
| 无效区域 | 超出各 batch 有效 Q/KV 范围的稀疏位置保持 0 |

BNSD 的有效长度只限制计算范围，不缩小物理 stride。K 的逻辑 GEMM 视图是
`[D, selected_KV]`，不意味着必须物理转置整个 K。

## 流水排布、同步关系与数值精度

以下是本知识对应的两种精度约定，适配组件时逐项核对：

| 中间量 | innerPrecise=0 | innerPrecise=1（仅 FP16） |
| --- | --- | --- |
| Cube 累加 | FP32 | FP32 |
| QK 写出 S | FP32 | FP16 |
| scale、max、exp、sum、rescale 状态 | FP32 | FP16 路径，需核对参考的实际表达式 dtype |
| P 供 PV 使用 | 转为输入 dtype | FP16 |
| PV 中间结果与 O 更新 | FP32 | FP16 |
| 最终 O | 输入 dtype | FP16 |
| LSE | FP32 log/add | FP16 log/add 后转 FP32 |

# 用法

## 十二分派覆盖

每行组合分别覆盖 LSE 关闭和开启，共十二种；内部 tiling key 编码由实现决定。
不能从其他架构的覆盖表只筛选非量化项，就认为已覆盖本输入域。

| Q/KV layout | dtype | innerPrecise | LSE |
| --- | --- | --- | --- |
| TND/TND | FP16 | 0 | 0 / 1 |
| TND/TND | FP16 | 1 | 0 / 1 |
| TND/TND | BF16 | 0 | 0 / 1 |
| BNSD/BNSD | FP16 | 0 | 0 / 1 |
| BNSD/BNSD | FP16 | 1 | 0 / 1 |
| BNSD/BNSD | BF16 | 0 | 0 / 1 |

# 代码模式

以下地址公式以元素为单位，是布局关系而非完整 kernel：

```text
TND:  q_base(b,s,h) = ((sum(Sq[:b]) + s) * Nq + h) * D
BNSD: q_base(b,s,h) = ((b * Nq + h) * Sq_max + s) * D
kv_head(h) = floor(h / (Nq / Nkv))
```

K/V 使用各自的 Sk、Nkv 和物理 stride。TND 的 LSE token stride 为 Nq，
BNSD 的 LSE head stride 为物理 Sq_max。

# 约束

本范围仅 `quantMode=0`、`maskType=0`，不接受 dense attenMask 或 blockTable，公开
`blockSize=0`，`preTokens/nextTokens=2147483647`，量化 scale 为空。
BF16+innerPrecise=1、任何 +4、BSND 和量化均不在本条目范围。
共享组件保留的参数或内部 paged 分支不等于公开接口支持。

空 sparse row、非正长度、非二进制 mask、无效 KV 块置 1 及非有限输入，须由项目提供明确的
参考行为。跳过空行且不写 O/LSE 不能推出 O=0 或 LSE=-inf；也不能依赖分配器初值。

# 失败表现

- 把 TND 长度列表当累计和，导致第二个 batch 起地址错误。
- 用 BNSD 有效长度缩小 stride，导致 head/batch 之间读写错位。
- 将补零 K/V 的尾列作为有效 score，导致 softmax 分母增加。
- 只看 scale 参数 dtype 或 LSE 输出 dtype，遗漏内部舍入差异。

# 验证方法

覆盖十二分派、B>1、Nq≠Nkv、两个 D、非连续保留块、X 非对齐与 X>128、Y 多个128、
KV tail、LSE 开关。布局转换对照只比较约定的有效区；padding 是否要求清零应单独冻结。
参考环境变化或在线组宽变化后重新验证 O/LSE，见[Golden 与验证](golden-and-validation.md)。

[^sparse-attention-semantics-and-dispatch-source]: 审计编号 sparse-attention-semantics-and-dispatch-source；固定来源与版本记录由提取侧独立保管，不随生成侧资料分发。
