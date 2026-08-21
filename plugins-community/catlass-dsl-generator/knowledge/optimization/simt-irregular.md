---
type: CATLASS DSL Optimization Guide
title: CANN Performance：SIMT 不规则访存与冲突处理
description: 从 Histogram、Scatter 和 MoE Routing 专题提取 grid-stride、原子/单写者、二维线程布局、向量 load 与 GridDim 扫描候选。
tags: [catlass-dsl, optimization, simt, histogram, scatter, gather, moe, atomic]
status: draft
generated: {by: process:cann-samples-performance-extract, at: '2026-08-12T00:00:00Z'}
verified:
  - {by: process:cann-samples-source-audit, at: '2026-08-12T00:00:00Z'}
sources:
  - id: histogram
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/2_Performance/simt_histogram_story/README.md
    title: SIMT Histogram optimization story
  - id: scatter
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/2_Performance/simt_scatter_story/README.md
    title: SIMT Scatter conflict handling story
  - id: routing
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/2_Performance/moe_init_routing_story/README.md
    title: MoE Init Routing optimization story
operator_families: [histogram, scatter, gather, moe-routing, sort]
arch: [c310]
---

# 接口与概念

Histogram 把 scalar `i=0..N` 改为 SIMT grid-stride，使用线程并行 GM 读和 UB 原子计数；后续把标量
load 合并为向量 load，并用资源报告约束 launch bounds，再扫描 GridDim。样例也展示“load 指令变少但
总时延不变”可证伪 GM 瓶颈。[^histogram]

Scatter 的第一原则是先确定写语义：index 唯一时每线程直接写；重复 index 时，输入若已按目标地址
分组，则每组选择单写者/规约者；二维数据再让一维线程选 group、另一维覆盖连续列。[^scatter]
MoE Routing 把排序、直方图和离散 Gather 分别多核化；Gather 先测试双/多 buffer 和多行 UB 缓存，
对小 H 离散包再测试二维 SIMT 直接 GM 访问；有序 expert id 的边界检测可拆成独立线程任务。[^routing]

# 用法

Scalar 串行且元素独立时先做 grid-stride；写目标可能冲突时先选 atomic 或单写者语义；地址第一维
离散、第二维连续时测试二维线程布局；只有 profile 支持 GM 指令/带宽瓶颈时测试宽 load；最后以
资源/寄存器为边界扫描 thread/block/GridDim。

# 代码模式

```python
tid = global_thread_id()
stride = total_threads()
for i in range(tid, n, stride):
    key = gm_input[i]
    atomic_add(local_hist[key], 1)

# 已按目标地址分组的 scatter：一个 owner 写一次。
group = thread_dim0
col = thread_dim1
if is_owner(group) and col < width:
    gm_out[target[group], col] = reduce_group_value(group, col)
```

未分组的重复 index 不能直接套单写者方案；排序/owner-map 的前处理必须计入端到端。宽 load 要满足
对齐和 tail，二维 Gather 要保证 `(row_thread, col_thread)` 覆盖无重叠无缺口。[^histogram][^scatter]

# 约束

- 适用：`c310` 且当前 CATLASS lowering 支持所需 SIMT/atomic；数据量、冲突率、分布和 H/width 必须记录。
- 保持：atomic/owner 的数学写语义、稳定性要求、索引边界、dtype、tail 和输出初始化不变。
- 代价：atomic contention、前处理排序、线程寄存器、Grid 调度和直接 GM 小事务；SIMT GM 路径不保证
  比 MTE 多 buffer 更高带宽，大 H 时优先复测 SIMD/MTE。
- 可证伪预期：scalar 串行周期、每线程工作或 GM 指令下降，并转化为总延迟改善。

# 失败表现

- 重复 index 非确定错：遗漏冲突语义，回退 atomic 或构造 owner map。
- GridDim 增大后更慢/编译 spill：跨过资源拐点，回退上一个稳定配置。
- 宽 load 无收益：计算/atomic 才是瓶颈，恢复简单 load。
- 小包 SIMT 有效但大 H 退化：切回 MTE 多 buffer，按 H 设模板门限。
- 多核 sort 合并占主导：并行分片过细或全核同步过重，减少分片。

# 验证方法

覆盖唯一/全冲突/混合/偏斜分布、空输入、tail、越界 index、不同 H/width 和多次重复；对非交换操作
检查确定性。记录 atomic 冲突、scalar/vector、GM/MTE、寄存器、每线程元素与 GridDim 扫描曲线；
同配置 benchmark 超过噪声并 fresh 复测后才接受。

[^histogram]: 固定提交中的 grid-stride、UB atomic、宽 load、launch bounds 和 GridDim 递进实验。
[^scatter]: 固定提交中的唯一 index 直写、分组单写者和二维线程布局。
[^routing]: 固定提交中的多核 sort/histogram/gather、buffer/UB 利用及 SIMT 边界/Gather 方案。
