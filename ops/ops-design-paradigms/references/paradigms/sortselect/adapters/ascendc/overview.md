# SortSelect AscendC 整体模型

按 `SortCheckParams` 验证 shape/dtype/axis，检查字节数和计数器溢出。L2 先把排序轴规范化为连续末轴，再调用 L0。`SelectSortSchedule` 顺序：短轴两阶段/插入 → MergeSort → 单核 Radix → 多核 Radix；N=1 走普通插入路线。

| schId | 保留的 Kernel |
|---|---|
| 5 | SIMT insertion |
| 6 | 两阶段排序，含逆映射和紧凑键恢复 |
| 0 | MergeSort，内部保留 Sort32 初始 run 和 MrgSort 归并 |
| 1 | 单核 radix |
| 2 | 多核 radix，按轴长选择 uint32/int64 计数与前缀 |

仅保留以上五类 schedule。多核 Radix 的选路不以 N 是否超过 2^30 为门槛，前面的候选未命中且资源可行即可进入；进入后按 N 选择 uint32 或 int64 计数模板。Sort32 不再作为独立模板/选路；非末轴通过 [布局包装](kernel-entrance.md#api-与非末轴转置包装) 复用同一组排序模板，不注册专用 schedule。

TilingKey 为 `(schId,isInt32,isDescend)`；sch2 的 isInt32 在 N≤2^30−1 时为 1（32 位，升/降序 key 258/65794），否则为 0（64 位，key 2/65538）；其余路线沿用各自的键约束。该字段不是输出索引 dtype。输入输出类型由框架注入。`keyParams0..5` 含义按路线变化，设计应逐字段列出 Host 来源与 Kernel 消费，不能跨 schId 复制解释。

## 符号与参数

| 符号 | 含义与单位 |
|---|---|
| O / N / I | 原排序轴前维度乘积 / 轴长 / 轴后维度乘积 |
| M=O×I | 独立排序行数；主体输入规范化为 `[M,N]` |
| L、first_rank | 每行最终输出项数、首个输出名次；中间完整排序仍用 N |
| C / C_act | 可用核数 / 实际启动核数 |
| B / Q | 当前 UB 批次行数 / 行批次任务数 |
| T / K / G | 多核块长 / 每行块数 / 每行协作核数 |
| P_act | 同时驻留外存 scratch 的活跃行数，区别于选路指标 P=ceil(M/C) |
| D / J / S | 值 / 最终索引 / 计数载体的字节数 |
| U、a、A(x) | 总 UB 字节数、UB block 字节数、按 a 向上对齐后的字节数 |

## 按模板阅读

每行是一条完整实现路线，按当前问题读取对应模板的三份文档。Tiling 给出候选计算与下发参数，Buffer 给出真实峰值和复用条件，Kernel 给出地址、阶段及同步。

| 模板 | Tiling | Buffer | Kernel |
|---|---|---|---|
| 插入排序 | [切分](templates/insertion/tiling.md) | [内存](templates/insertion/dag-buffers.md) | [计算](templates/insertion/kernel-template.md) |
| 两阶段排序 | [切分](templates/two-stage/tiling.md) | [内存](templates/two-stage/dag-buffers.md) | [计算](templates/two-stage/kernel-template.md) |
| 归并排序 | [切分](templates/merge/tiling.md) | [内存](templates/merge/dag-buffers.md) | [计算](templates/merge/kernel-template.md) |
| 单核 Radix | [切分](templates/radix-one-core/tiling.md) | [内存](templates/radix-one-core/dag-buffers.md) | [计算](templates/radix-one-core/kernel-template.md) |
| 多核 Radix | [切分](templates/radix-more-core/tiling.md) | [内存](templates/radix-more-core/dag-buffers.md) | [计算](templates/radix-more-core/kernel-template.md) |

## 示例入口

[Sort example](example/sort/README.md) 包含 Host、Kernel 及配套 `op_api/`。

模板文档承载可独立实现的算法、参数、资源设计和优化机制，包括按批或按元素的并行粒度、批量 DMA、状态通信及缓冲复用。example 是这些机制在当前 SDK 上的实现实例，用于核对 API 和适用域；生成工程不要求使用同名类、函数、字段或文件。输出区间属于生成适配合同，不能把示例的完整 Sort 接口直接当作 TopK/KthValue 已实现接口。
