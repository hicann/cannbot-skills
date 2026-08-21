---
type: CATLASS DSL Profiling Guide
title: MindStudio Insight 可视化瓶颈分析
description: 使用热力图、Roofline、Cache、流水图和源码热点将 msOpProf 结果定位到 CATLASS 代码。
tags: [catlass-dsl, profiler, msopprof, visualization, roofline, timeline]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-07-28T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-07-28T00:00:00Z'}
sources:
  - id: visualization
    resource: https://gitcode.com/Ascend/msopprof/blob/b362f30e7a49ccc5fb80f93f2026332f6001bb82/docs/zh/user_guide/msopprof_user_guide.md
    title: msOpProf 模式用户指南
operator_families: [elementwise, matmul]
arch: [c310]
---

# 接口与概念

`visualize_data.bin` 可导入 MindStudio Insight，按详情、Roofline、Cache、时间线
和源码视图逐层分析；部分通算流水也会生成 `trace.json`，可由 Insight 或
`chrome://tracing` 打开。不同图回答不同问题，不能用单一截图替代原始 CSV 和
可复现命令。[^visualization]

推荐分析顺序：

```text
Details：是否存在计算、内存或核间负载异常
  -> Roofline：理论上更像 compute、memory 还是 latency bound
  -> Cache/Memory：数据通路和复用是否合理
  -> Timeline：哪些 pipe 没有重叠、哪里在等待
  -> Source：具体代码行和指令是否解释上述现象
```

# 用法

## 功能与采集开关

| 分析目标 | 主要产物/开关 | 说明 |
| --- | --- | --- |
| 计算内存热力图 | `Default` / `visualize_data.bin` | 全局计算、内存和基础信息 |
| Roofline | `--aic-metrics=Roofline` | 与 Default 绑定 |
| 核间负载 | `--aic-metrics=Occupancy` | 比较耗时、吞吐和 Cache |
| Cache/源码搬运 | `--aic-metrics=MemoryDetail` | 展开 L2 和 MTE1/MTE2 信息 |
| 源码热点 | `--aic-metrics=Source` + `-g` | 源码、PC、执行次数、搬运量 |
| Pipe 流水 | `--aic-metrics=PipeTimeline` | 查看各 Pipe 运行情况 |
| 指令流水图 | `--aic-metrics=instrTimeLine` | 可用 `--instr-timeline-pipe` 限定 |
| Warp Stall | `--aic-metrics=PcSampling` | 产品限定 |

产品支持和参数互斥会随 CANN 版本变化；采集前使用本机 help 核对。固定指南中
PipeTimeline 与 instrTimeLine 不能同时启用，部分 Source、MemoryDetail、
Roofline 和 replay range 组合也不兼容。[^visualization]

## Details 与 Roofline

Details 视图先检查：

- Core Occupancy 是否存在明显慢核或吞吐不均；
- Compute Workload 中 Cube/Vector 是否被充分使用；
- Memory Workload 中 MTE 请求、带宽和利用率是否集中在某条通路；
- 当前显示的是活跃带宽、峰值占比还是 total-cycle 带宽。

Roofline 用算术强度和性能上限区分 Compute Bound 与 Memory Bound；当算子未接近
Roofline 上限时，还要结合最大 pipeline ratio 区分计算、内存或流水 latency。
Roofline 是模型结论，最终仍需由 CSV 数据量和时间线验证。[^visualization]

## Cache、指令流水图与源码

Cache 热力图展示 cacheline Hit/Miss，并能在具备 `-g` 和 Source 数据时跳到相关
源码或指令。低命中单元格只有在请求量和 MTE2 时间显著时才是高优先级。[^visualization]

指令流水图可限定 pipe，减少密集指令造成的数据丢失：

```bash
msprof op \
  --aic-metrics=instrTimeLine \
  --instr-timeline-pipe="mte1|vector" \
  --output=./artifacts/instr-timeline \
  ./run_operator
```

查看是否存在：

- MTE2→Vector/Cube→MTE3 串行而未重叠；
- Scalar 或 wait 长时间占用；
- Set/Wait、barrier 或数据依赖形成空洞；
- 边界 block 比主体 block 更慢；
- 同一数据在 GM 与片上存储之间往返。

源码热点图把代码行映射到指令 PC、PIPE、执行次数、L2 模拟命中率和 GM 搬运量；
其中时间线/源码维度的 L2 命中率是模拟数据，Details 核维度的命中率来自真实
采集，两者不可直接当作同一测量值。[^visualization]

# 代码模式

## 从图形形成可检验假设

```json
{
  "observation": "MTE2 and Vector are mostly serialized on the slowest block",
  "evidence": [
    "PipeUtilization.csv:aiv_mte2_time(us)",
    "ResourceConflictRatio.csv:aiv_vec_mte_cflt_ratio",
    "visualize_data.bin:Timeline:block=last"
  ],
  "hypothesis": "boundary-tile load prevents overlap",
  "single_change": "separate the tail load length while preserving pipeline depth",
  "validation": ["correctness", "same profile contract", "end-to-end latency"]
}
```

图形观察必须转写为字段、block、源码位置和单变量修改，不能只写“流水不好”。

## Source 映射采集

```bash
msprof op \
  --aic-metrics=Source,Default \
  --kernel-name="target*" \
  --launch-count=1 \
  --output=./artifacts/source \
  ./run_operator
```

最终 kernel 二进制必须含 `-g`。调试信息可能包含源码路径，采集产物按敏感构建
产物管理。

# 约束

- `visualize_data.bin` 需要 MindStudio Insight；`trace.json` 才能直接用
  `chrome://tracing` 打开。
- Cache 与源码跳转要求对应构建的调试信息，重新编译的二进制不能替代采集产物。
- PipeTimeline、指令时间线、Warp 和通算流水的产品/算子支持范围不同。
- 指令时间线每个 pipe 有条数限制；指令密集时可能丢数据，应减少循环或限 pipe。
- Pipe 流水基于采样，展示核数不等同于应用实际启动核数。
- 添加 MarkStamp/PrintTimeStamp 或 Warp 采集代码可能扰动耗时，不与无插桩数据
  直接比较。
- MC2/LCCL、推理产品和 SIMT/SIMD VF 对 Cache、Source、Timeline 的支持各异。

# 失败表现

| 表现 | 首查 |
| --- | --- |
| Insight 无法导入 | 文件是否完整、Insight 与 CANN/msOpProf 版本 |
| Details 有数据但 Source 为空 | `-g`、Source 指标、产品与算子支持 |
| Cache 不能跳源码 | MemoryDetail、Source、调试信息是否同时满足 |
| 指令流水图缺事件 | pipe 过滤、每 pipe 条数、密集循环、产品支持 |
| Pipe 流水只显示少量核 | 采样行为，不要误判为只启动这些核 |
| Timeline 和 Details 的 L2 不一致 | 模拟代码/指令维度与真实核维度的差异 |
| Roofline 与实际优化方向冲突 | 数据量、频率、绝对 pipe time 和 latency bound |
| 插桩后耗时变差 | MarkStamp/PrintTimeStamp/Warp 采集扰动 |

# 验证方法

```bash
find ./artifacts -name visualize_data.bin -type f -print
find ./artifacts -name trace.json -type f -print
find ./artifacts -name OpBasicInfo.csv -type f -print
```

导入后保存视图名称、Device/Core/Block、时间范围、源码行、对应 CSV 字段和原始
产物路径。任何优化结论都要回到无额外插桩的相同 workload 复测。本文未实际导入
MindStudio Insight。

[^visualization]: 固定提交的热力图、Roofline、Cache、流水、源码热点和 Warp Stall 功能及限制说明。
