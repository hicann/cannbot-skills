---
type: CATLASS DSL Optimization Guide
title: 优化证据门禁
description: 将源码启发、profile 假设、正确性和 benchmark 结论严格分离。
tags: [catlass-dsl, optimization, benchmark, profiling, correctness]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
sources:
  - id: readme
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/README.md
    title: CATLASS DSL validation layers
  - id: examples
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/basic_mmad/README.md
    title: MMAD build and run examples
operator_families: [matmul, elementwise, mixed]
arch: [c310]
---

# 接口与概念

仓库把 pytest、lit、build-only 和上板运行分成不同验证层。[^readme] MMAD 文档
也分别给出 build-only 与 device run 入口。[^examples] 因此源码只可支持“机制
存在”和“候选假设”，不能直接支持“性能提升”。

# 用法

优化轮次按以下门禁推进：

1. baseline 绑定 commit、配置、设备环境和 benchmark JSON。
2. hypothesis 引用 fresh profile 或本目录/算子目录的具体来源。
3. 单轴修改后运行完整正确性。
4. 正确后才运行同配置 benchmark。
5. 超过噪声阈值才接受，并从 best SHA 最终 fresh 复测。

# 代码模式

Benchmark 结果的关键结构：

```json
{
  "performance": {
    "status": "passed",
    "reference": {
      "mean_ms": 1.20,
      "median_ms": 1.19,
      "min_ms": 1.17,
      "std_ms": 0.03,
      "trials": [1.17, 1.19, 1.24]
    },
    "candidate": {
      "mean_ms": 1.08,
      "median_ms": 1.07,
      "min_ms": 1.05,
      "std_ms": 0.02,
      "trials": [1.05, 1.07, 1.12]
    },
    "speedup": 1.1111
  }
}
```

当 `metric_path=performance.candidate.mean_ms` 且 direction 为 decrease：

```text
observed_improvement = (best_mean - candidate_mean) / best_mean
required_improvement =
max(min_improvement_fraction,
    best_std / best_mean,
    candidate_std / candidate_mean)
accept iff observed_improvement >= required_improvement
```

示例的噪声门槛为 `max(min_fraction, 0.025, 0.0185)`；观测改善为 10%，只有
它不小于最终门槛时才接受。

# 约束

- 适用：所有性能优化，不因修改小而跳过。
- 代价：多次测量增加时间，但避免把噪声写成知识。
- 正确性门禁：required cases 全部通过且证据绑定候选 commit。
- 性能门禁：benchmark 配置、metric path、设备和输入与 baseline 一致。
- correctness 不是 `passed` 时，performance 必须为 `not_run`。
- 比较前验证 result SHA-256、候选 commit 与 benchmark command id。

# 失败表现

- `performance.status=not_run` 却声称 speedup：非法结论。
- 只比较 `min_ms`：容易接受偶然快值。
- mean 改善小于 std 比例：噪声内变化，不更新 best。
- 旧 benchmark、不同配置、错误 commit 或缺失要求的 profile：证据失效。

# 验证方法

最终只把 fresh full test 或 profiling/benchmark 直接证明的条件化结论写入
`learned/`。未运行项记录 `not_run`，不得标为已验证性能。

# Launch、profile 与知识准入

## Launch 覆盖是 profile 的前提

优化前后必须证明 device 侧实际并行度覆盖候选 kernel 的全部任务。不要只根据 CLI 或
wrapper 参数名判断 grid 已生效；应检查多 task 输出覆盖、launch argv、cache key，以及
msprof summary 中目标 kernel 的 `Block Num`、`Mix Block Num`、`Task Duration(us)`、
`aicore_time(us)`、`aiv_time(us)` 和 `cube_utilization(%)`。[^examples]

实际 grid 小于任务数时，部分任务不会写回，输出可能保留初始值；这种 profile 不能用来
接受候选。先用少量 focus workload 证伪，再扩大到全部 workload。保存 device、
launch argv、cache key 和 profile 路径，避免旧 binary 或旧 grid 混入比较。

## 结论分类

- `有效`：同一配置下正确性通过，profile 或 benchmark 有可复现收益。
- `条件有效`：收益依赖 SoC、dtype、layout、任务粒度、launch grid 或 workspace；必须
  明确写出这些条件。
- `无效`：编译或正确性失败、性能回退、收益低于噪声，或仅证明当前候选不可用。

无效尝试可作为反例或 blocker 记录，但不得写成优化推荐。编译器 lowering、VF stack、
空 vector region、动态 pointer/tensor load 等错误应保留原始片段，并与运行时或 profile
环境干扰分开。只有完整测试或 profiling 直接支持的结论才能进入 `learned/`。

[^readme]: 固定提交 README 对不同验证层的定义。
[^examples]: 固定提交 MMAD 文档中独立 build 与 run 工作流。
