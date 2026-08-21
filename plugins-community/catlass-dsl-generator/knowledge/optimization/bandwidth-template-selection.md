---
type: CATLASS DSL Optimization Guide
title: CANN 样例：带宽扫描与模板选择
description: 用隔离读、读写、读算写数据流和 UB tile/buffer 成对扫描，为 CATLASS 搬运候选建立实测上限。
tags: [catlass-dsl, optimization, bandwidth, ub, buffer, profiling, template-selection]
status: draft
generated: {by: process:cann-samples-feature-extract, at: '2026-08-12T00:00:00Z'}
verified:
  - {by: process:cann-samples-source-audit, at: '2026-08-12T00:00:00Z'}
sources:
  - id: bandwidth-readme
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/1_Features/hardware_features/mem_bandwidth/README.md
    title: GM-UB bandwidth experiment contract
  - id: bandwidth-source
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/1_Features/hardware_features/mem_bandwidth/src/bw_common.h
    title: Bandwidth tiling and buffer implementation
operator_families: [elementwise, copy, reduction, mixed]
arch: [c310]
---

# 接口与概念

固定样例用三条最小数据流分离纯读、读写拷贝、读+Vector 计算+写，并成对扫描每 buffer 的 UB
tile 字节数和 buffer 数。每核 GM 区间按 block 切分，最后一 tile 单独处理 tail；多 buffer 仅用于
建立相邻 tile 的搬运/计算重叠。[^bandwidth-readme][^bandwidth-source]

这类 microbenchmark 不是业务 kernel 的性能证明，而是给候选提供同设备、同 dtype、同方向下的
可达带宽曲线，帮助判断应优先扩大搬运粒度、增加 stage，还是减少总字节。

# 用法

先按业务数据路径选择最接近的模型：只读、copy，或 load-compute-store。扫描 `(tile_bytes,
buffer_count)` 的合法组合，得到吞吐平台区和容量边界；再把业务 kernel 的有效字节/时间与该曲线
比较。若小 tile 随粒度增长明显改善，优先测试合并搬运；若 1→2 buffer 改善而更深无收益，使用
双缓冲；若已经接近曲线平台，优先减少字节或融合 GM round trip。

# 代码模式

```python
for tile_bytes, stages in legal_pairs:
    assert stages * tile_bytes * queue_count <= usable_ub_bytes
    run_same_device_same_dtype_same_direction(
        total_bytes=fixed_total_bytes,
        tile_bytes=tile_bytes,
        stages=stages,
        warmup=warmup_count,
        repeats=repeat_count,
    )
    record(raw_times, effective_bytes, mte2, mte3, vector)
```

队列容量按“每 buffer 字节 × buffer 数 × 队列数”计算；读算写有独立输入/输出队列时必须计两份，
不能把 `tile_bytes` 误当整条队列容量。[^bandwidth-readme]

# 约束

- 适用范围：`c310` 的 GM↔UB 搬运候选；dtype、方向、总字节、多核数和计算体必须与比较目标匹配。
- 保持条件：每组扫描处理相同有效元素；tail 不重读/漏写；带宽分母按模型区分读字节与读+写字节。
- 资源代价：更大 tile/stage 增加 UB，可能减少 occupancy；过小总数据受启动开销主导。
- 可证伪假设：增加粒度/stage 后有效带宽上升且业务 kernel 对应等待下降；若 microbenchmark 改善
  但业务不变，说明业务另有依赖、复用或计算瓶颈。
- 默认扫描表只是固定样例的取点，不是通用最优参数；所有组合需通过当前 kernel 的容量计算。

# 失败表现

- 非法组合编译/运行失败：UB 容量或队列数计算错误，缩小 tile/stage。
- 带宽虚高：分母重复计数、只计有效字节却读写 padding，或采集未覆盖完整 kernel。
- 结果随重复次数漂移：热身、频率、并发占用或采集边界不稳定，停止模板选择。
- microbenchmark 最优点使业务更慢：occupancy、tail、同步或计算比例不同，回退业务 baseline。

# 验证方法

按项目约定先查询空闲 NPU，在非沙箱环境 `source env.sh` 后采集。固定设备、dtype、核数、总字节、
warmup 和重复次数，保存每次原始时长而非只保存均值。验证每组输出/tail，再绘制 tile_bytes ×
buffer_count 的中位数和离散度；业务候选必须另跑完整正确性、同配置 benchmark/profile 和 fresh
best 复测，microbenchmark 曲线本身不能进入 learned 性能结论。

[^bandwidth-readme]: 固定提交中三种数据流、带宽口径、tile/buffer 扫描和容量约束。
[^bandwidth-source]: 固定提交中多核切分、队列初始化和 tail 实现。
