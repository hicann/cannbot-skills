---
type: CATLASS DSL Optimization Guide
title: CANN Performance：MatMul 与 Grouped MatMul 模板选择
description: 从 MatMul/GMM 递进样例提取按 Bound 选模板、组间连续排核、动态 M 均衡、Scale/Weight 布局和 L2 bypass 候选。
tags: [catlass-dsl, optimization, matmul, grouped-matmul, quantization, scheduling, l2]
status: draft
generated: {by: process:cann-samples-performance-extract, at: '2026-08-12T00:00:00Z'}
verified:
  - {by: process:cann-samples-source-audit, at: '2026-08-12T00:00:00Z'}
sources:
  - id: matmul-guide
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/2_Performance/matmul_story/docs/matmul_performance.md
    title: Non-quantized MatMul performance guide
  - id: matmul-tutorial
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/2_Performance/matmul_story/matmul_tutorials/README.md
    title: MXFP4 MatMul staged tutorial
  - id: grouped-guide
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/2_Performance/grouped_matmul_story/docs/quant_grouped_matmul_mx_performance.md
    title: Quantized Grouped MatMul performance guide
operator_families: [matmul, grouped-matmul, quantized-matmul]
arch: [c310]
---

# 接口与概念

MatMul 专题把 Cube、MTE2、MTE1、Fixpipe 理论耗时作为模板路由条件，再按单轴顺序测试 ping-pong、
SWAT、尾轮均衡、unit flag、Half-L1 Bank 隔离、Scale 搬运合并和 A 全载。[^matmul-guide]
MXFP4 教程保留从 baseline 到上述七步的独立可执行版本，使每步可单独对照。[^matmul-tutorial]

Grouped MatMul 在单 MatMul 之外增加三类问题：不同 group 的边界不能复用；动态 `M_e` 导致块量不等；
每组都从 core 0 排核会累积空洞。固定指南用跨 group 连续排核、动态 M 基本块均衡和尾轮再切分处理
这些问题，并给出 2/3-buffer、ScaleB 私有 NZ、WeightNZ 及 bypass L2 候选。[^grouped-guide]

# 用法

先以同一 tile 估算/测量四条流水，只选择当前最长轴：MTE2 测 SWAT、Scale 合并、预排布或 L2
bypass；MTE1 测 Bank 隔离；Fixpipe 测 unit flag；Cube 已连续但总利用率低则测 group/M/tail 分核。
只有 profile 仍有搬运抖动且容量富余，才把 L1 stage 从 2 增至 3/4。

# 代码模式

```python
# group 间连续排核，而不是每组从 0 重启。
logical_base = 0
for group in groups:
    for local_tile in range(group.tile_count):
        owner = (logical_base + local_tile) % core_count
        schedule(owner, group, local_tile)
    logical_base += group.tile_count

# 动态 M：以真实 M_e 计算块数、tail 和地址；零大小 group 不产生任务。
for group, m_e in enumerate(group_sizes):
    if m_e == 0: continue
    dispatch_balanced_m_tiles(group, m_e, base_m, base_n)
```

Scale/Weight 预排布候选必须同步调用侧 storage contract；L2 bypass 仅针对一次性大权重流，需证明它
不会驱逐更有复用价值的数据，也不会改变可见结果。3-buffer 只增加 L1 stage，不能假设 L0 也有
第三槽。[^grouped-guide]

# 约束

- 适用：`c310` MatMul/GMM；FP16/BF16/MXFP4/MXFP8/HIF8 等 dtype 与 ND/DN/NZ/ZN 必须写入实验。
- 保持：group 顺序、零 group、M/K 分组语义、各组 GM 基址、tail、scale 粒度、累加 dtype 和输出拼接不变。
- 代价：额外 stage 占 L1；预排布增大 padding/调用侧工作；bypass 可能伤害复用；动态调度增加 scalar
  地址计算；K 分组需要部分和协议，不能冒充 M 分组。
- 可证伪预期：目标 pipe 等待/字节、L2 污染或 per-core 长尾下降，并改善同配置总延迟。
- `1_Features` 中已有机制细节；本条只增加“按 Bound 路由”和 GMM 的组间/动态 shape 决策层。

# 失败表现

- 某个 group 起错：累计 GM 偏移、零 group 或 trans/layout 合同错误，恢复逐组 baseline。
- 正确但负载仍不均：tile cost 不能只用块数近似，改用实测/拟合代价或回退。
- 3/4-buffer 编译失败、base tile 被迫缩小或 occupancy 下降：回退 2-buffer。
- bypass L2 后其它输入变慢：复用集假设错误，恢复默认 cache 路径。
- NZ/ScaleB 私有布局端到端变慢：离线转换或 padding 未摊销，保留 ND 路径。

# 验证方法

覆盖 group 数 1/多、含 0、极不均匀 M、M/K 分组、transB、尾 M/N/K 和各量化布局；在 host 枚举
逻辑 tile 证明每组覆盖一次。正确后比较各 group/core timeline、L2、MTE2/MTE1/Fixpipe/Cube、
stage 容量和总延迟；多组 group-list 与相邻 shape 扫描，fresh best 复测后才形成 learned 结论。

[^matmul-guide]: 固定提交中的流水建模与按 Bound 选择 DB、SWAT、unit flag、Bank、Scale、全载策略。
[^matmul-tutorial]: 固定提交中的 MXFP4 baseline 至七个单轴优化版本入口。
[^grouped-guide]: 固定提交中的分组语义、2/3-buffer、布局/L2 候选及 group/M/tail 负载均衡。
