---
type: operator
title: block_sparse_attention Arch22 稀疏任务与四阶段流水
description: 建立 Q 行所有权、逻辑 KV 流、GM 槽、在线状态及跨核通知的生命周期约束。
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
sources: [{"id": "sparse-attention-scheduling-and-pipeline-source", "resource": "audit:custody/sparse-attention#scheduling-and-pipeline", "title": "scheduling-and-pipeline 固定来源（提取侧独立保管）", "kind": "audit-record"}]
---

# 接口与概念

## 算子算法

保留的 KV 块按原编号组成逻辑流，QK 和 PV 使用同一映射。
最后一个物理 KV 块只贡献真实剩余 token；物理间隙与补齐尾列均不进入 softmax。
在线组宽改变归约和状态更新粒度，不改变被保留的集合。

## 分核策略与基本块切分

一种可用的 Q 切分是：任务固定 batch、Q head，在同一 X 稀疏块中处理至多128行。
每 head 的任务数为：

```text
floor(Sq / X) * ceil(X / 128) + ceil((Sq mod X) / 128)
```

跨 batch/head 求和得到任务总数。每个输出行只有一个 owner；该方案不做 KV split，
因此无需跨核合并 softmax 状态。核数按目标平台可用资源和任务数选择。
按任务数量均分不等于按保留 token 数均分；调度变更须观察每核真实负载。
X>128 时可以拆任务，但不能跨 X 边界。

## 数据路径与存储层级

| 数据 | 生产与消费 | 释放条件 |
| --- | --- | --- |
| 索引/count | Vector 压缩 mask，所有 Q 任务读取 GM | 本次调用的最后读取完成 |
| Q tile | GM→L1，当前任务的多次 QK 复用 | 最后一个 QK 消费完成 |
| K/V tile | 稀疏 GM→L1→L0，分别用于 QK/PV | 对应矩阵读取完成，映射一致 |
| S | QK Fixpipe→GM→Vector UB | 当前组 softmax 最后读取完成 |
| P | Vector cast→GM→PV | 两个子核均写完才可读，最后 PV 消费后可回收 |
| local O | PV Fixpipe→GM→Vector | 对应 rescale 完成 |
| m/l/alpha/Oacc | UB 或按组保存的状态 | 依对应组和最终化的最后消费者确定 |

预算按最大同时存活空间计算：S 为 `M*N*sizeof(S)`，P 为 `M*N*sizeof(input)`，
local O 为 `M*D*sizeof(Otmp)`；在途组数倍增对应交接区。
host 分配量、设备 slot stride 和各精度分支必须一致，不采用固定的每核 workspace 常数。

## 流水排布、同步关系与数值精度

同组依赖为 C1(QK)→V1(online softmax)→C2(PV)→V2(rescale/finalize)。不同组可交错，
但生产者推进后，延迟消费者仍必须使用自己的 alpha、shape、first/last 标识。
最终 V2 读取最终 m/l 前须完成全部 V1。

在 AIC 对应两个 AIV 子核、Vector 沿行切分的组织中，奇数行和零行子核也必须配平外层通知。
P 可读需要两个子核都完成对应部分的写回；本地 HardEvent、跨核 flag 和全核 barrier
分别承担不同范围的依赖，不能互相替代。GM 槽增加也不意味着 L0 份数增加。

# 用法

## 稀疏索引与边界

将每个 mask row 压缩成有序索引/count，确保计数等于实际保留块数、索引无重复，
索引0与未写位置可区分。预处理若独立 launch，计时包含该 launch；若融合，
消费者必须等 GM 写完成且跨核可见后才读取。禁止跨不同输入复用旧索引。

物理 mask 列数控制扫描范围，保留块数控制压缩列表长度。对窗口容量 W，分别测试
物理列尾和实际保留 W−1/W/W+1 块，不能只依赖 shape 或用例名判断已跨窗口。
K/V 必须共同处理 Y>128 的展开、非连续块和物理末块截断。

## 在线组与流水深度

增大在线组可能减少归约、local O 写回和握手，也可能增加 UB/L1/GM、启动成本和舍入误差。
先确认矩阵特化支持合法子块：首片初始化 C，后片累加，末片写回；不能只扩大单次 MMAD
的 actualShape。AB 供数槽与 C 累加槽分别管理，共享 QK/PV 不得覆盖在途结果。

改变消费距离前，区分初始可用信用和实际完成通知。通知数量按真实消费次数发布，
不是最大组宽；尾组不能多发，零行子核不能漏发。整槽可保守地在最后 V2 完成后释放，
结束时排空全部在途工作。更深流水可能帮助长任务而增加短任务开销，应分别测量。

### 任务内 KV 流水与跨任务接续（静态审计经验）

必须区分两个流水层级：

1. **任务内流水**：一个输出任务由单一 owner 完整遍历其选中的 KV 序列。可在该 KV 窗口循环内让 QK/在线 softmax 领先 PV，并以有界 lookahead、循环槽和事件同步保护 S/P/O 等工作区；任务结束时必须排空在途窗口。
2. **跨任务接续**：同一逻辑核之后还会领取其他任务，不代表前一任务的 KV 被拆到其他核，也不代表已有任务间预取。若前一任务尾部仍有在途消费者，默认需等其完成再复用任务状态/槽位；跨任务重叠是另一个优化，必须单独设计并验证。

### 接续优化的适用边界

同一 X 稀疏块包含多个 Q tile 时，可把“同一 owner 复用 K/V 的分组调度”作为独立候选，
但必须保持每个输出任务各自的 online-softmax 状态与完整 KV 所有权；它与把 KV 拆分到
多个核、或在不同任务间预取不是同一种变换。任何跨任务重叠都要先用设备 trace 证明边界
存在可隐藏空泡，再做同输入配对验证；不能由任务网格、每核任务数量或单个慢用例推断因果。

预取下一任务的 Q 或下一 KV 子块，可能减少等待，也可能引入索引解码、地址准备、事件和
寄存器压力。缓存稀疏索引、扩大在线组、改变缓冲数量或直接写 LSE 也都是待测机制，必须
分别说明依赖、所有权和容量条件；少一次访存或 launch 不等于完整调用变快。对任何单例
收益都应检查它是否影响目标集合的总体瓶颈。

用 profiler 判断主核瓶颈时，分别观察 AIV Vector、AIC MAC 与搬运/同步路径；当向量路径
突出而矩阵计算相近时，优先定位 softmax、归约及归一化，而非只增加 K/V 预取。流水线
计数可能并行重叠，不要把它们简单相加为关键路径时长。代数重写（如除法改倒数乘法）
可能改变舍入，应将精度与性能验证分开；设备状态异常或配对链路失败时，不形成性能结论。

# 代码模式

```text
task -> (batch, q_head, sparse_row, q_begin, valid_rows)
group -> (selected_tokens, valid_columns, first, last)
slot(group) -> S, P, local_O, alpha_of_this_group
producer -> completion_event -> consumer -> last_consumer_done -> reuse
```

若要提前通知 P ready，需证明所有 P 已写回且余下工作不修改 PV 所读数据；
未 cast 的求和值必须保存，m/l 更新仍需在下一 V1 或最终 V2 消费前完成。
P ready 不代表整个组状态完成，也不允许提前复用 slot。

# 约束

融合 mask 预处理与主计算时，显式定义逻辑 AIV 工数、物理 launch 网格和空核路径。
所有参与全核 barrier 的物理核必须到达同一同步点，逻辑网格外核只能在之后退出；
预处理与主阶段的资源构造/析构、UB 峰值和同步 flag 冲突均须重新核对。
跨 stream 排队正确不等于已证明设备级同步区在真并发下安全。

行片预取需区分 GM 行距、UB pitch 和状态偏移；每个 ready 完成消费后才能释放事件。
释放事件 ID 本身不是完成等待。持久占用的事件 ID 与当前在途通知分开预算。

# 失败表现

任务跨 X 边界、QK/PV 索引不一致、尾列参与归约会造成确定性数值错误。
旧组使用最新 alpha、初始信用被当作完成、零行子核漏通知、slot 过早复用则可能产生
偶发错误或卡死。缩短流水可辅助定位，但不能代替最终实现的验证。

# 验证方法

手工枚举小 shape 的 owner，覆盖 X 非整除、B>1、Nq>Nkv。再覆盖单组、短于/等于/超过
流水深度、重复环转、奇数/零行子核、非连续块、索引跨窗、多任务及延后同步的排队调用。
改变组宽后从实际输入重新计算在线组数，验证 O/LSE 并测完整调用，不能沿用旧几何标签。

[^sparse-attention-scheduling-and-pipeline-source]: 审计编号 sparse-attention-scheduling-and-pipeline-source；固定来源与版本记录由提取侧独立保管，不随生成侧资料分发。
