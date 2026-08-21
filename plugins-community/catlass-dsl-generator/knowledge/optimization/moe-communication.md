---
type: CATLASS DSL Optimization Guide
title: CANN Performance：MoE 通信前量化
description: 从 MoE Dispatch/Combine 专题提取通信前量化、搬运、反量化三阶段候选及端到端门禁。
tags: [catlass-dsl, optimization, moe, communication, quantization, dispatch, combine]
status: draft
generated: {by: process:cann-samples-performance-extract, at: '2026-08-12T00:00:00Z'}
verified:
  - {by: process:cann-samples-source-audit, at: '2026-08-12T00:00:00Z'}
sources:
  - id: moe-comm
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/2_Performance/moe_dispatch_and_combine_story/README.md
    title: MoE Dispatch and Combine communication optimization story
operator_families: [moe-dispatch, moe-combine, communication]
arch: [c310]
---

# 接口与概念

固定专题的核心候选是在 Dispatch/Combine 通信前对 payload 量化，传输低比特数据及其 scale，接收后
反量化，再进入后续专家计算或聚合；实现把“量化→通信搬运→反量化”作为有硬件同步的三段协议。
[^moe-comm] 这是以额外计算和误差换通信字节，必须按端到端而非单 kernel 判断。

# 用法

只有通信字节/带宽处于临界路径、量化格式已有明确数值合同时才实验。先单独测原 payload 搬运，再加入
量化/反量化但关闭真实通信以测计算成本，最后测完整 Dispatch+Combine；不能同时改变路由、expert
排序或并行规模。

# 代码模式

```python
scale = compute_scale(payload_tile)
q = quantize(payload_tile, scale, rounding=approved_rounding)
signal(quant_ready)
wait(quant_ready)
communicate(q, scale)
signal(receive_ready)
wait(receive_ready)
payload_fp = dequantize(q, scale, accumulation_dtype=approved_dtype)
```

每个通信 tile 的 payload、scale 和元数据必须共享相同路由/专家坐标；producer 在量化与 scale 写完后
才能发布 ready，consumer 读取完成前 slot 不能复用。[^moe-comm]

# 约束

- 适用：`c310` MoE Dispatch/Combine，通信处于临界路径；dtype、量化粒度、EP 规模和 token 分布需记录。
- 保持：路由、token 顺序、expert ownership、重复 token 聚合、scale 对应关系和同步顺序不变。
- 数值合同必须覆盖 rounding、饱和、零点、NaN/Inf、反量化与 Combine 累加 dtype。
- 代价：量化/反量化 Vector 周期、scale/metadata 字节、额外 buffer、同步和精度损失。
- 可证伪预期：净通信字节和临界通信时长下降，且量化开销后端到端仍超过噪声门槛。

# 失败表现

- token 仅跨 rank/专家时错：payload 与 scale 路由或 offset 不一致，回退原通信格式。
- 数值超容差：量化粒度/舍入不满足合同，恢复原 dtype。
- 通信下降但端到端更慢：消息太小或 Vector 已是瓶颈，撤销通信前量化。
- 偶发 hang：阶段事件、slot generation 或通信完成语义不闭合。

# 验证方法

覆盖空 expert、极端倾斜 token、不同 EP 规模、尾 tile、全零/大值/特殊值和 Dispatch→Combine 往返；
逐 tile 核对 payload-scale 坐标。保存通信前后字节、量化/反量化、通信、等待和端到端原始时长；同配置
多次 benchmark、完整 golden 与 fresh best 复测均通过后，才可记录为 learned。

[^moe-comm]: 固定提交中的 Dispatch/Combine 内存流程和通信前量化、通信、反量化同步阶段。
