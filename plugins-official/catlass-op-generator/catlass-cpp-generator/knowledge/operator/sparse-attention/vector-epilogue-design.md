---
type: operator
title: block_sparse_attention Arch22 多行 Vector epilogue
description: 用明确的状态布局、真实列归约和缓冲生命周期设计多行 softmax、rescale 与 LSE。
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
sources: [{"id": "sparse-attention-vector-epilogue-design-source", "resource": "audit:custody/sparse-attention#vector-epilogue-design", "title": "vector-epilogue-design 固定来源（提取侧独立保管）", "kind": "audit-record"}]
---

# 接口与概念

## 算子算法

online softmax 维护每行 m/l 和未最终归一化的 O；PV 使用转换到输入 dtype 的 P。
最终 O 除以全局 l，LSE 按精度模式计算 log(l)+m。把每个局部块先除行和再做 PV，
会改变上述在线累加定义，不能直接用普通 SoftMax 替换。

## 分核策略与基本块切分

一个 Q 任务中，各 AIV 管理各自真实行。行块大小由 padded 列宽、工作 dtype、同时存活
空间及 API repeat/stride 限制共同决定，不固定为单行或某个行数。计算基本预算为：

```text
score_bytes = rows * padded_columns * sizeof(work_dtype)
total_live_bytes = score + probability + reduction_scratch + row_states + other_live_buffers
```

扩大列组或增加 FP32 临时路径后，重新计算可容纳的行数，并保留小 shape 和行尾路径。

## 数据路径与存储层级

| 行状态表示 | 使用方式 | 不能混淆的条件 |
| --- | --- | --- |
| 每行一个连续元素 | 批量计算，再广播成行块 | 源端需有足够连续元素及安全 padding |
| 每行完整32B块，所有 lane 同值 | 直接参与按行广播算术 | 更新必须保持整个块一致 |
| 每行预留32B但仅首元素有效 | 按该布局读取或先转换 | 既不是紧凑向量，也不是完整广播块 |

状态布局覆盖 m/l/alpha 的全部生产者与消费者。归约 scratch、未 cast 的求和值、
PV 所用 P 和在途 alpha 分开保护；布局转换也计入 UB 与执行成本。

## 流水排布、同步关系与数值精度

多行提交可减少逐行调用和 V→S 往返，但不允许省略 RAW/WAR 依赖。
改变求和树或 scale 舍入路径属于数值变更，应与性能变量分开验证。
低精度 LSE 即使输出 FP32，也可能要求在 half 中完成 log/add，不能只按输出 dtype 实现。

# 用法

## 广播与归约

按目标 SDK 核对 Brcb 的连续源读取、32B 广播块、对齐和 repeat 约束。源缓冲必须覆盖
完整读取范围，尾行分配并初始化 padding；稀疏 cell 布局不能直接作为紧凑源。
通用 CATLASS 广播组件也需要匹配静态形状与实际有效行。

归约按真实列数执行。分块归约、每 repeat 归约的目的 stride 单位可能不同，必须查对应
API 的实际签名；完全未激活的块可能不写目的位置，后续归约不得消费旧 scratch。
half 分片求和若会原地改写输入，应先保存 PV 需要的 P；FP32 求和不能改成对已 cast P 求和。

## 矩形提交与尾部

只有每行满列且整体连续时，才可按总 count 提交 scale/cast。
带 padding 的矩形需分别传递真实列 mask、行 repeat 和物理 stride，Cast 两侧 stride
按各自元素字节数换算。更宽的 scale 临时区必须覆盖最大真实行数。

## 输出与 LSE

alpha 广播后对 O 做多行 rescale/add，最终 l 广播后做归一化与 cast。
先保持除法和舍入方式，不同时引入近似倒数。LSE 的计算布局可以是紧凑行向量，
写回前再转换到满足搬运对齐要求的布局；LSE 关闭时不应无条件计算。
复用 O/P/scratch 前等待此前消费者完成。

# 代码模式

```text
compact row state -> padded broadcast blocks
valid rectangle + physical pitch -> scale / exp / cast
real-column reduction -> compact m / l / alpha
alpha broadcast -> O rescale
final l broadcast -> O normalization; log(l) + m -> LSE
```

若用不重叠 scratch 分区减少复用等待，需证明写集不交叉、总容量足够和目标管线顺序；
合并前仍需保证所有归约结果已完成。不能把“同管线”当作任意 API 都可删除屏障的依据。

# 约束

参考的参数 dtype 不等于实际表达式 dtype。涉及 Python/NumPy 标量和零维数组时，
记录版本、提升模式、shape、值域及中间结果；不要一律强制 FP32 乘后转 half，
也不要一律先把 scale 转 half。详见[参数精度核对](golden-and-validation.md#参数精度核对)。

# 失败表现

单行通过而多行失败时，重点检查广播源连续性、repeat stride、状态 padding、宽精度
临时区容量以及归约是否破坏 P。仅 LSE 失败时，单独检查 log/add 的实际工作精度。
减少源级调用或屏障数量不构成性能收益证据。

# 验证方法

覆盖 0/1 行、行块前/满/跨块、真实列尾、归约边界、晚出现新最大值和长在线累加。
交叉覆盖数值分支×多行×满列/尾列，以及 D64/128 和 LSE 开关。
可污染未写 scratch/padding 检查初值依赖；通过原 O/LSE 判据后，再比较完整调用性能。

[^sparse-attention-vector-epilogue-design-source]: 审计编号 sparse-attention-vector-epilogue-design-source；固定来源与版本记录由提取侧独立保管，不随生成侧资料分发。
