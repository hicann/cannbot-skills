---
type: CATLASS DSL Optimization Guide
title: CANN Performance：Attention 多级流水与负载均衡
description: 从 Flash Attention Lite 与全量化 FIA 递进样例提取片内交换、CV/L0/I-O 多级流水、代价感知分核和 DN 归约候选。
tags: [catlass-dsl, optimization, attention, flash-attention, pipeline, mixed, load-balance]
status: draft
generated: {by: process:cann-samples-performance-extract, at: '2026-08-12T00:00:00Z'}
verified:
  - {by: process:cann-samples-source-audit, at: '2026-08-12T00:00:00Z'}
sources:
  - id: falite
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/2_Performance/flash_attn_lite_story/README.md
    title: Flash Attention Lite v0-v5 optimization story
  - id: fia
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/2_Performance/full_quant_fused_infer_attention_score_story/README.md
    title: Full-quant FIA performance guide
operator_families: [flash-attention, attention, mixed]
arch: [c310]
---

# 接口与概念

两个固定专题共同给出一条由外到内的 Attention 优化轴：按实际序列长度估算 tile 代价并均衡分核；
将 C1→V1→C2→V2 从同 tile 串行改为跨 tile 错位；把 S、ΔO 从 L0C 经 Fixpipe 直送 UB；
把 P 从 UB 送 L1；再分别对 L0、K/V L1 和 task 级 Q/O 做双槽所有权。[^falite][^fia]

Flash Attention Lite 的 v2 单槽用 `DONE` 保护整套资源，v3 改为按 slot 的 S/P/O ready 事件，
v4 为 L0A/L0B/L0C 增加双槽并仅保留真实正反依赖，v5 把 V 预取前移至 C1，并让 task t 的
输出 MTE3 与 task t+1 的输入/计算错位。[^falite]

FIA 还提供两个正交候选：对变长 KV tile 使用动态目标负载的连续贪心分配；把 MM1 方向改为
`K × Q^T`，让 Vector 在转置后的 DN 组织上用并行 `vmax/vadd` 代替逐行归约。[^fia]

# 用法

按瓶颈一次只测试一层：核间长尾先测代价感知分核；GM 中间流量高先测 L0C→UB/UB→L1；
AIC/AIV 交替空闲先测 CV 双槽；MTE1/MMAD/Fix 互等再测 L0 双缓冲；task 边界有空洞再测 I/O
双槽；Vector 归约单发占主导时再测 DN 方向。

# 代码模式

```python
# CV slot 协议：不同 tile 可重叠，同一 slot 必须闭环。
slot = tile_id & 1
cube_c1(tile_id, s_ub[slot]); set(s_ready[slot])
wait(s_ready[slot]); vector_v1(tile_id, s_ub[slot], p_l1[slot]); set(p_ready[slot])
wait(p_ready[slot]); cube_c2(tile_id, p_l1[slot], do_ub[slot]); set(o_ready[slot])
wait(o_ready[slot]); vector_v2(tile_id, do_ub[slot], state)

# 代价感知连续分核。
remaining_cost = sum(tile_cost)
for core in range(core_count):
    target = remaining_cost / (core_count - core)
    assign_next_contiguous_tiles_until_half_tile_rule(target)
```

外层 I/O 双槽必须由最后读取 Q/OAcc 的阶段释放；K/V L1 slot 只有在 C2 读完 V 后才允许下一代
C1 覆盖。DN 候选还必须成套修改 MM1 方向、Vector 坐标和 MM2 输入转置。[^falite][^fia]

# 约束

- 适用：`c310` mixed Attention；shape/tile、BF16/FP8/HIF8 路径及实际序列分布须单独记录。
- 保持：online-softmax 的 `m/l/O` 更新顺序、mask、尾 KV、量化 scale、累加/写回 dtype 不变；
  AIC/AIV 对 slot、SubBlock 和 tile generation 使用同一映射。
- 代价：每层双槽增加 L1/L0/UB 与事件；DN 可能增加转置/布局成本；负载均衡可能拆分 batch，
  引入 Flash-Decoding workspace、归约和同步。
- 可证伪预期：GM 字节、跨核等待、阶段空洞或最长核时长下降，并转化为高于噪声的总延迟改善。
- 不把文档中的单 shape、单次 Task Duration 或理论比例当作 CATLASS 性能结论。

# 失败表现

- tile 1 起错或 hang：slot generation、ready/free 或跨核方向不闭合，回退单槽。
- 长序列尾部错：online-softmax 状态、tail mask 或拆核归并次序错误。
- 容量/编译失败或并行 block 下降：减少一层双槽，保留更外层高收益流水。
- DN 正确但变慢：转置/重排成本超过归约并行收益，恢复 ND。
- 各阶段 active time 改善但总延迟不变：瓶颈迁移或 overlap 指标被误加，回退该单轴候选。

# 验证方法

正确性覆盖单/双/奇数 KV tile、短/长和不等长 batch、mask 尾块、量化边界与多 task；用事件 trace
检查每个 slot 的 producer→consumer→release。随后同配置比较 per-core 最长时长、GM 字节、AIC/AIV
等待、MTE1/MTE2/MTE3、Cube/Vector 与总延迟，多次 benchmark 定义噪声阈值并 fresh 复测 best。

[^falite]: 固定提交中 v0-v5 的 GM/片内路径、CV 双槽、L0 双缓冲、KV 预取和 task I/O 双槽。
[^fia]: 固定提交中变长 tile 代价分核、CV/Cube 流水、DN 方向和双缓冲策略。
