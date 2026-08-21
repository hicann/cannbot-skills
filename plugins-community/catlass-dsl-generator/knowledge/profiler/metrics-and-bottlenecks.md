---
type: CATLASS DSL Profiling Guide
title: msOpProf CSV 指标解读与瓶颈归因
description: 解读基础耗时、计算、搬运、Cache、局部存储和资源冲突指标，并映射到 CATLASS 优化候选。
tags: [catlass-dsl, profiler, msopprof, metrics, csv, bottleneck]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-07-28T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-07-28T00:00:00Z'}
sources:
  - id: performance-data
    resource: https://gitcode.com/Ascend/msopprof/blob/b362f30e7a49ccc5fb80f93f2026332f6001bb82/docs/zh/user_guide/msopprof_performance_data.md
    title: msOpProf 模式性能数据
operator_families: [elementwise, matmul]
arch: [c310]
---

# 接口与概念

msOpProf 的 CSV 分为“执行条件”“计算占用”“内存通路”“流水耗时”和“冲突等待”
五类。字段会随产品和 AIC/AIV 核类型变化，`ai*` 表示可能展开为 `aic` 或
`aiv`；解释数据时必须保留原始列名和芯片型号。[^performance-data]

| 文件 | 先回答的问题 |
| --- | --- |
| `OpBasicInfo.csv` | 测了哪个算子、耗时、Block Dim、频率是否一致 |
| `ArithmeticUtilization.csv` | Cube/Vector 指令占用了多少 cycle、做了多少计算 |
| `PipeUtilization.csv` | Vector/Cube/Scalar/MTE/Fixpipe 时间花在哪里 |
| `Memory.csv` | GM、L1、L2、UB 通路的数据量和带宽利用率 |
| `MemoryL0.csv` | L0A/L0B/L0C 读写带宽 |
| `MemoryUB.csv` | Vector/Scalar/MTE 与 UB 的读写带宽 |
| `L2Cache.csv` | 读写命中、缺失及总命中率 |
| `ResourceConflictRatio.csv` | bank、bank group、执行资源和 MTE 等待 |

# 用法

## 1. 先确认基础条件

从 `OpBasicInfo.csv` 核对 `Op Name`、`Op Type`、`Task Duration(us)`、
`Block Dim`、`Mix Block Dim`、`Device ID`、`Current Freq` 和 `Rated Freq`。
`Task Duration(us)` 包含调度、设备执行和结束响应时间，不能直接当作某条设备
指令的时间。[^performance-data]

比较 baseline 和 candidate 前先拒绝以下样本：

- shape、dtype、layout 或输入数据不同；
- Block Dim、Mix Block Dim 或 Device 不同；
- 当前频率差异足以解释延迟变化；
- correctness 未通过；
- metric set、warm-up 或 replay mode 不同。

## 2. 计算与流水

`ArithmeticUtilization.csv` 提供 `aic_cube_ratio`、`aiv_vec_ratio`、各 dtype
占比、FLOPs 和指令数。`PipeUtilization.csv` 进一步给出
`*_cube_time(us)`、`*_vec_time(us)`、`*_scalar_time(us)`、
`*_mte{1,2,3}_time(us)`、`fixpipe`、ICache miss 和活跃带宽。[^performance-data]

归因顺序：

```text
绝对 Task Duration 是否异常
  -> 最慢核与核间离散度
  -> 最大的绝对 pipe time
  -> 对应 ratio、active bandwidth、wait/stall
  -> 源码或指令时间线验证
```

ratio 高只说明占本核 total cycle 的比例高，不自动等于该单元已达到峰值，也不
说明降低 ratio 一定能降低总耗时。

## 3. 内存层级

`Memory.csv` 覆盖 GM↔UB、GM↔L1、L0C↔L1/GM 的数据量、带宽和使用率；
`MemoryL0.csv` 覆盖 L0A/L0B/L0C；`MemoryUB.csv` 覆盖 Vector、Scalar、
GM 与 UB 的读写带宽。单位 `GB/s` 表示每秒传输的数据量。[^performance-data]

| 现象 | CATLASS 首查 |
| --- | --- |
| GM→UB 数据量超出理论输入 | 重复 load、边界 tile、recast、layout/stride |
| UB→GM 数据量超出理论输出 | 中间结果回写、重复 store、缺少驻留 |
| GM→L1 重复且 MTE2 时间高 | K 维循环复用、L1 双缓冲、tile 容量 |
| L1→L0A/B 或 L0C 写回受限 | Cube tile、搬运参数、Fixpipe、pipeline |
| UB 带宽高但 Vector 利用低 | 中间 tensor、广播/重排、bank 冲突 |

先用 shape×dtype 计算理论最小字节数，再与 `*_datas(KB)` 比较；只有数据量合理
时，带宽利用率才适合用来判断通路是否接近瓶颈。

## 4. Cache 与冲突

`L2Cache.csv` 提供读写 hit/miss 和 hit rate；低命中率会影响 MTE2，但命中率
必须结合总请求量、MTE2 时间和主存数据量判断。小样本的高命中率变化可能对总
时延没有意义。[^performance-data]

`ResourceConflictRatio.csv` 中常用字段包括：

- `*_vec_bankgroup_cflt_ratio`：常与 block stride 不合理相关；
- `*_vec_bank_cflt_ratio`：常与读写指针地址映射相关；
- `*_vec_resc_cflt_ratio`：多个指令竞争同一执行资源；
- `*_vec_mte_cflt_ratio`：Vector 与搬运资源冲突；
- `*_mte{1,2,3}_wait_ratio`、`*_vec_wait_ratio`、`*_cube_wait_ratio`：流水等待。

冲突 ratio 是候选线索，必须由热点代码、指令流水或单变量实验确认。[^performance-data]

# 代码模式

## 理论搬运下界

```python
def tensor_bytes(shape, itemsize):
    elements = 1
    for extent in shape:
        elements *= extent
    return elements * itemsize


input_bytes = tensor_bytes((256, 1024), 2)
weight_bytes = tensor_bytes((1024, 512), 2)
output_bytes = tensor_bytes((256, 512), 4)
minimum_gm_read = input_bytes + weight_bytes
minimum_gm_write = output_bytes
```

把理论下界、CSV 实际数据量和两者比值写入优化证据；不要只保存截图。

## 保留每核分布

```python
import csv
import statistics

with open("PipeUtilization.csv", newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

field = "aiv_time(us)"
values = [float(row[field]) for row in rows if row.get(field) not in (None, "", "NA")]
print({
    "count": len(values),
    "min": min(values),
    "median": statistics.median(values),
    "max": max(values),
    "spread": max(values) - min(values),
})
```

平均值会掩盖慢核；性能结论至少保留 min/median/max、最慢 block 和原始 CSV。

# 约束

- AIC、AIV 和统一 AI Core 的字段命名不同，不能用缺失列补零后混算。
- ratio 的分母是相应 Cube 或 Vector 核的 total cycle，不同核类型不可直接相加。
- active bandwidth 与按 total cycle 计算的 bandwidth 含义不同。
- `L1_to_GM` 等带 `(estimate)` 字段是估算值，证据中必须保留该限定。
- `MemoryDetail` 失败时部分 MTE1/MTE2 活跃带宽为 `NA`，不能解读为 0。
- Block Dim 是逻辑核数；核数改变时，单核数据量、尾块和调度开销都会改变。
- 不同产品暴露的列集合不同；本文给出解释框架，不承诺所有列在 c310 环境存在。

# 失败表现

| 表现 | 首查 |
| --- | --- |
| candidate ratio 下降但更慢 | 绝对 pipe time、Task Duration、频率和数据量 |
| 平均耗时正常但尾延迟高 | 最慢 block、边界 tile、核间 spread |
| MTE2 ratio 高 | GM 数据量、L2 命中、GM→UB/L1 带宽、重复搬运 |
| Vector wait 高 | flag/barrier、依赖链、bank/MTE/resource conflict |
| Scalar 时间高 | 循环控制、动态 shape、wait、ICache miss |
| L2 命中低但时延不变 | 总请求量是否足够大、是否真正受 MTE2 限制 |
| 活跃带宽缺失 | MemoryDetail、芯片支持和动态插桩状态 |
| CSV 列与指南不同 | 芯片、CANN/msOpProf 版本和核类型 |

# 验证方法

```bash
for name in OpBasicInfo ArithmeticUtilization PipeUtilization Memory \
  MemoryL0 MemoryUB L2Cache ResourceConflictRatio; do
  find ./artifacts/profiler -name "${name}.csv" -type f -print
done
```

瓶颈结论必须同时引用：基础条件、绝对耗时、相关 ratio/带宽/数据量、每核分布和
源码或时间线证据。优化前后只改变一个候选因素，并重新运行 correctness 与相同
profile 契约。本文未在 NPU 上验证字段可用性。

[^performance-data]: 固定提交中八类 msOpProf CSV 的产品差异、字段定义和单位说明。
