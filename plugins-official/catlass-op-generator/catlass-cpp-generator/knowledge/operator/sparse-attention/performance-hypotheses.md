---
type: operator
title: block_sparse_attention Arch22 性能假设与验证
description: 用成本模型和完整调用测量选择优化候选，明确资源、数值、同步风险与否证条件。
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
sources: [{"id": "sparse-attention-performance-hypotheses-source", "resource": "audit:custody/sparse-attention#performance-hypotheses", "title": "performance-hypotheses 固定来源（提取侧独立保管）", "kind": "audit-record"}]
---

# 接口与概念

## 算子算法

设任务 t 有 M_t 个 Q 行和 R_t 个真实保留 KV token，QK 与 PV 合计的静态计算量约为
`4*Σ(M_t*R_t*D)` FLOP。mask 扫描量还随物理稀疏矩阵大小增长；若采用 int32 索引，
每个索引占4字节。该模型未计对齐、重复读取、归约、同步和 launch，不能直接预测速度。

## 分核策略与基本块切分

按实际保留 token、任务建立成本和尾波分析负载，不仅看任务数或核心占用率。
拆任务可能增加并行度，也可能保持相同的 ceil 波数并增加初始化成本。
同一 KV head 下的 Q heads 可能具有不同 mask，GQA 不自动保证 KV 复用。

## 数据路径与存储层级

从完整调用区分 metadata、mask、主计算、内部事件及后处理。
小任务可能由固定开销主导；长任务可能受矩阵、Vector 或供数限制，需由当前数据确认。
增加槽数、预取缓冲或整组 L1 驻留，均要同时核算 UB/L1/L0/workspace 和代码生成影响。

## 流水排布、同步关系与数值精度

减少 API、屏障或 launch 数量只是候选机制。保持数值路径或单独报告数值变更，
再用受控配对确认完整调用收益。聚合计数器不能拼成串行关键路径，
流水深度增加也不等于已经隐藏等待。

# 用法

## 从完整调用选择候选

先确认[测量范围与版本](execution-and-measurement.md)，再比较逐例主要开销和汇总。
可将某分量理想设为零、保持其他项不变，估算优化上限；上限本身不是可实现收益。
核周期覆盖、MAC 活动率和 Vector/Scalar 活动率采用不同分母时，不能直接互相替代。
没有足够证据时先做能区分原因的诊断，不按利用率阈值自动选择机制。

## 候选条件与负向边界

以下均为需要在当前实现验证的候选，不给出通用收益承诺。它们提炼了源码约束和测量中
需要防止的过度归因；不把某一组输入上的负结果扩大为永久禁用规则。

| 候选机制 | 适用前提 | 风险与否证条件 | 验证重点 |
| --- | --- | --- | --- |
| 少任务同步限核 | `totalTasks` 小于物理 AIC 数，且任务可独立分配 | 主核 launch grid 与 tiling 中的 grid-stride 步长必须同时取 `min(totalTasks, physicalAic)`；只改一侧会漏算或重复任务，缩核也可能无收益 | 核对全部任务覆盖和槽索引；冻结精度回归通过后，以同输入、同设备、同采样口径比较完整调用 |
| 多行 softmax/rescale | 逐行控制与标量往返占明显成本 | 广播/转换/scratch 开销抵消收益；小 shape 退化 | 状态布局、真实列归约、行尾及完整调用 |
| 更宽在线组 | 矩阵特化支持合法多微块累加 | UB 行分片、L1 压力、舍入变化和启动成本增加 | 非连续 KV、短尾、P 通知次数、O/LSE |
| 更深跨组流水 | 等待有可重叠工作，alpha 按组保存 | 长任务改善但短任务建立与排空更慢 | 单/少组、多次环转、共享 L0、逐类时延 |
| S 行片预取或 P/V 整组驻留 | 容量与事件可支持额外在途数据 | 预取控制、事件占用和布局转换抵消搬运节省 | 缓冲生命周期、父 layout 偏移、累加槽 |
| 提前 P ready | P 已写完，后续仅更新 PV 不读的行状态 | 错误复用概率/alpha，重复或漏发通知 | 未 cast 求和值、零行子核、最终 m/l |
| mask 预处理融合 | 可在同一 launch 内明确所有权和全核同步 | 空核提前退出、同步区冲突、资源生命周期重叠 | 每次真实压缩、barrier 前后顺序、并发限制 |
| 均匀 batch 元数据设备推导 | 所有 batch 的 Sq/Sk 相同且可由紧凑参数重建 | ABI 镜像错位、整数溢出、非均匀回退或代码生成退化 | host/device 公式一致、回退、逐调用 copy 预期 |
| mask 内循环向量化或行流水 | 分类收益足以覆盖短块开销 | 同步和分类成本增大；快路径切换时缺失 WAR 保护 | 混合块、尾块、越界行转换与完整调用 |
| 减少行片或 scratch 复用屏障 | UB 容量足够且写集/依赖明确 | 更高资源压力或错误移除必要同步 | 各精度分支容量、合并前就绪、全部消费者 |

增加或融合资源后，重新核对所有消费者和 ABI 镜像。仅声明“未改分支源码”不足以
排除重编译造成的性能变化，回退路径也需要进入测量。

### 可迁移的机制判断

- **限核必须与任务步长一致。** 当任务数少于可用 AIC 数时，缩小物理 launch grid
  只有在 grid-stride 步长和任务覆盖公式同步更新时才正确；收益取决于尾波、负载和
  固定开销，不能仅凭减少空闲核推断完整调用会更快。
- **预取先证明存在可隐藏等待。** 对 S、K 或下一任务 Q 提前发起搬运，可能同时增加
  地址计算、状态准备、事件和槽位压力。先用设备 trace 定位等待，再验证依赖距离和
  有界缓冲；搬运发得更早本身不是流水收益的证据。
- **窗口宽度与缓冲组织是资源交易。** 更大的 KV 窗口或更少的缓冲可能减少循环/归约
  次数，也可能提高 UB/L1 占用、降低并行驻留或增加尾块浪费。不要把窗口、缓冲和
  行距同时改动；按单变量比较完整调用，并覆盖短任务与尾块。
- **索引缓存需计入控制成本。** 少读 GM 标量不必然更快；缓存分支、寄存器占用和索引
  更新可能抵消节省。以热点路径的设备计时和 profiler 为准。
- **LSE 直写受所有权和对齐约束。** 仅当输出行独占、地址满足写粒度对齐且 padding/
  尾行处理明确时，才考虑跳过 staging/scatter。局部省去一次 launch 不代表整体收益；
  先分解后处理在完整调用中的占比，并保留不满足条件时的通用路径。
- **连续归约合并适合归约调用开销占主导的路径。** 满窗口可尝试减少分段归约与循环
  控制，但 partial 布局、非满窗口和数值更新顺序应独立保持/验证，不能将满窗改写
  无条件推广到尾窗。
- **按瓶颈流水线选优化对象。** 若 profiler 显示 AIV Vector 路径显著繁忙而 AIC MAC
  时间相近，优先检查 softmax、归约和归一化；若矩阵流水受供数约束，再评估 K/V 搬运。
  Pipe/利用率计数可以重叠，不能直接相加成 kernel 时长或关键路径。
- **等价代数变换仍需分别过精度和性能门。** 例如逐元素除法改为逐行倒数再乘法，可能
  改变舍入，也可能被额外广播/调度抵消；精度通过不能替代同输入配对性能验证。

# 代码模式

```text
fixed baseline + one attributable mechanism
    -> unchanged input / reference / accuracy criterion
    -> loaded-version and full-call coverage checks
    -> paired baseline / candidate samples, including regressions
    -> independent confirmation where needed
```

均匀长度元数据推导可研究以下等价关系；这是设计公式，不代表已实现优化：

```text
q_prefix(b) = b * sq
kv_prefix(b) = b * sk
tasks_per_head = floor(sq / X) * ceil(X / 128) + ceil((sq mod X) / 128)
task_prefix(b) = b * tasks_per_head * Nq
```

须证明与 host 侧计算结果一致，检查溢出边界，并为非均匀输入保留正确路径。
消除 H2D 后按实际数据路径更新测量预期，并分别验证均匀和非均匀输入。

# 约束

不存在可复用到所有设备、shape 和会话的固定收益阈值。
局部阶段更快但完整调用退化时，以约定的完整指标判断；不同消融的收益不能相乘推算验收成绩。
同会话效应与跨会话结果分开比较，保留基线和候选两侧的波动。

惩罚型探针不一定可线性反推优化收益。例如缩小缓冲导致更多行片，还可能引入额外复用和
流水断裂；反向扩大缓冲并不保证回收同样比例的时间。必要时做双向探针或直接测量候选。

# 失败表现

常见误判包括：少了几条屏障就宣布更快；将 mask 头部代理测量当成全调用收益；
只看有利样本；忽略旧库误装载；把同一产物的运行波动记作优化。
不能因单次胜出而忽略采样离散度，也不能因一次负结果宣称某类组件永久无效。

# 验证方法

先确保 O/LSE 判据通过，再测匹配版本的完整调用。采用 A/A 检查测量链，
固定输入与采样规则，保存正反顺序、全部样本、逐例退化和汇总方式。
对接近噪声范围或用于达标声明的结果，根据实测方差增加独立确认；
未建立收益时如实保留不确定性，不将静态模型写成实测结论。

[^sparse-attention-performance-hypotheses-source]: 审计编号 sparse-attention-performance-hypotheses-source；固定来源与版本记录由提取侧独立保管，不随生成侧资料分发。
