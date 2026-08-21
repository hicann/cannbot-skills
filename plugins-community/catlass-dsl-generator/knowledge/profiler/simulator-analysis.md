---
type: CATLASS DSL Profiling Guide
title: msOpProf Simulator 仿真采集与指令级分析
description: 编译 CATLASS 仿真目标，采集逐行和逐指令性能数据，并用流水与热点图验证优化假设。
tags: [catlass-dsl, profiler, msopprof, simulator, instruction]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-07-28T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-07-28T00:00:00Z'}
sources:
  - id: simulator-guide
    resource: https://gitcode.com/Ascend/msopprof/blob/b362f30e7a49ccc5fb80f93f2026332f6001bb82/docs/zh/user_guide/msopprof_simulator_user_guide.md
    title: msOpProf Simulator 模式用户指南
  - id: simulator-data
    resource: https://gitcode.com/Ascend/msopprof/blob/b362f30e7a49ccc5fb80f93f2026332f6001bb82/docs/zh/user_guide/msopprof_simulator_performance_data.md
    title: msOpProf Simulator 模式性能数据
  - id: simulator-scenarios
    resource: https://gitcode.com/Ascend/msopprof/blob/b362f30e7a49ccc5fb80f93f2026332f6001bb82/docs/zh/user_guide/msopprof_usage.md
    title: msOpProf 使用场景
operator_families: [elementwise, matmul]
arch: [c310]
---

# 接口与概念

msOpProf Simulator 用软件仿真器生成逐代码行、逐指令和流水时序数据，适合解释
上板指标背后的指令排布、数据搬运和同步原因。仿真结果用于形成和筛选假设，真实
收益仍以上板相同 workload 的 correctness 与延迟为准。[^simulator-guide]

输入方式有三种：

- application：拉起仿真可执行文件或 Python 程序；
- `--config`：输入算子 `.o` 的配置；
- `--export`：重新解析已有的仿真结果目录。

核心产物是 `core*_code_exe.csv`、`core*_instr_exe.csv`、
`visualize_data.bin` 和每核/汇总 `trace.json`。[^simulator-data]

# 用法

## CATLASS 仿真目标

模板库使用 `--simulator` 构建仿真可执行文件，并按构建输出加载对应 SoC 的
Simulator runtime：

```bash
bash scripts/build.sh --simulator 00_basic_matmul
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib/libruntime_camodel.so:/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib/libnpu_drv_camodel.so
```

实际 `Ascendxxxyy` 和库路径以本机安装与构建回显为准。若要源码、调用栈和热点
映射，最终 kernel 使用工具链支持的 `-g -O2`；指南明确 msOpProf 不支持
`-O0`。[^simulator-scenarios][^simulator-guide]

## 最小仿真采集

```bash
python3 -c 'import acl; print(acl.get_soc_name())'
msprof op simulator \
  --soc-version=Ascendxxxyy \
  --output=./artifacts/simulator \
  ./00_basic_matmul 256 512 1024 0
```

对 Python 入口：

```bash
msprof op simulator \
  --soc-version=Ascendxxxyy \
  --kernel-name="target*" \
  --launch-count=1 \
  --output=./artifacts/simulator \
  python3 minimal_repro.py
```

仿真会依次运行应用中的算子；即使指定 `--kernel-name`，仍建议移除无关 kernel，
避免仿真时间被前序算子放大。[^simulator-scenarios]

## 选择核、范围与超时

```bash
msprof op simulator \
  --soc-version=Ascendxxxyy \
  --core-id="0|1" \
  --timeout=10 \
  --aic-metrics=PipeUtilization,ResourceConflictRatio \
  --output=./artifacts/simulator \
  ./run_operator
```

- `--core-id` 用于只解析部分逻辑核；分布不均匀时不能只看 core 0。
- `--timeout` 到期会终止仿真并解析已生成的数据，适合重复多、耗时长的算子。
- Simulator 默认采集 PipeUtilization 和 ResourceConflictRatio。
- `TRACE_START`/`TRACE_STOP` 配合 `-DASCENDC_TRACE_ON` 可限定单核代码范围，但
  嵌套或不配对会导致流水无法正常绘制。[^simulator-guide]

## 输出结构

```text
OPPROF_{timestamp}_XXX/
├── dump/
└── simulator/
    ├── core0.veccore0/
    │   ├── core0.veccore0_code_exe.csv
    │   ├── core0.veccore0_instr_exe.csv
    │   └── trace.json
    ├── core0.cubecore0/
    │   └── ...
    ├── visualize_data.bin
    └── trace.json
```

`core*_code_exe.csv` 包含 `code`、`call_count`、`cycles` 和
`running_time(us)`；`core*_instr_exe.csv` 进一步包含 `instr`、PC `addr`、
`pipe` 和搬运地址、长度、步幅等 `detail`。[^simulator-data]

# 代码模式

## 从最慢代码行下钻到指令

```python
import csv

with open(
    "simulator/core0.veccore0/core0.veccore0_code_exe.csv",
    newline="",
    encoding="utf-8",
) as handle:
    rows = list(csv.DictReader(handle))

rows.sort(key=lambda row: float(row["cycles"]), reverse=True)
for row in rows[:10]:
    print(row["code"], row["call_count"], row["cycles"], row["running_time(us)"])
```

对最高 cycle 的代码行，在相同核的 `*_instr_exe.csv` 中检查指令、pipe、执行
次数和 detail，再到 `trace.json` 确认它与相邻 MTE/Vector/Cube 指令的重叠。

## CATLASS 假设映射

```text
MTE2 指令多且 detail 显示重复地址
  -> 检查 GM load 复用和 tile 驻留
MTE1/L0 搬运占据主体
  -> 检查 L1 tile、L0A/B 双缓冲和 K 循环
Vector 指令 cycle 高且 UB bank 冲突
  -> 检查 layout、stride、交错和地址映射
Scalar/cache_time/ccu_time 高
  -> 检查动态循环、控制流、发射和 ICache
SET_FLAG/WAIT_FLAG 间出现长空洞
  -> 检查 flag 生命周期和 pipeline 依赖
最慢行只在边界核出现
  -> 检查尾块 extent、分支和补零
```

# 约束

- 仿真结果不等同于上板耗时，不能单独证明性能提升。
- SoC 型号、Simulator 库、kernel 架构和 CANN/msOpProf 版本必须匹配。
- `-g` 包含源码信息，应限制二进制和结果访问权限。
- `--core-id` 不代表只仿真这些核，而是限制解析范围；PMSampling 会解析全部核。
- `--export` 目录只应包含多核数据和名为 `aicore_binary.o` 的 kernel 文件；
  缺少对象时流水无法映射源码。[^simulator-guide]
- `TRACE_START`/`TRACE_STOP` 只适合支持的单核范围，不用于跨核或嵌套范围。
- Simulator 热点字段存在产品差异；Process Bytes、GPR、UB bank conflict 等
  不保证在所有产品上同时可用。

# 失败表现

| 表现 | 首查 |
| --- | --- |
| 仿真程序无法启动 | SoC、`LD_LIBRARY_PATH`、`LD_PRELOAD`、构建架构 |
| 仿真耗时过长 | 移除无关算子、`--timeout`、缩小 shape、限定解析核 |
| 没有源码行 | `-g`、匹配的 kernel 对象、`aicore_binary.o` |
| `trace.json` 范围异常 | TRACE_START/STOP 是否成对、是否嵌套 |
| core 0 看起来正常 | 检查全部核和边界核，不假设负载均匀 |
| CSV 有 cycle 但 Insight 无热点 | `visualize_data.bin`、版本兼容和调试信息 |
| 仿真优化但上板无收益 | 频率、缓存、调度、真实带宽和测量契约 |
| Process Bytes 为 NA | 指令是否涉及 GM、产品是否支持该字段 |

# 验证方法

```bash
find ./artifacts/simulator -name '*_code_exe*.csv' -type f -print
find ./artifacts/simulator -name '*_instr_exe*.csv' -type f -print
find ./artifacts/simulator -name trace.json -type f -print
find ./artifacts/simulator -name visualize_data.bin -type f -print
```

保存最慢核、最慢代码行、对应指令、pipe、调用次数、cycles、detail 和时间线区间。
由仿真提出的修改必须重新执行 correctness，并在真实设备用相同上板 profile 契约
复测。本文只核对固定提交的 Simulator 指南，未运行仿真器。

[^simulator-guide]: 固定提交的 Simulator 输入方式、命令参数、流水、热点和限制说明。
[^simulator-data]: 固定提交的 Simulator 输出目录及逐行、逐指令、可视化和 trace 文件字段说明。
[^simulator-scenarios]: 固定提交的 CATLASS、Kernel、API、PyTorch 和 Triton 仿真采集示例。
