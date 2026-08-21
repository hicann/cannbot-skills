---
type: CATLASS DSL Profiling Guide
title: msOpProf 上板采集与 CATLASS 调用流程
description: 使用 msOpProf 对 CATLASS、Python 和单算子程序进行可复现的上板性能采集。
tags: [catlass-dsl, profiler, msopprof, collection, benchmark]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-07-28T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-07-28T00:00:00Z'}
sources:
  - id: board-guide
    resource: https://gitcode.com/Ascend/msopprof/blob/b362f30e7a49ccc5fb80f93f2026332f6001bb82/docs/zh/user_guide/msopprof_user_guide.md
    title: msOpProf 模式用户指南
  - id: scenarios
    resource: https://gitcode.com/Ascend/msopprof/blob/b362f30e7a49ccc5fb80f93f2026332f6001bb82/docs/zh/user_guide/msopprof_usage.md
    title: msOpProf 使用场景
operator_families: [elementwise, matmul]
arch: [c310]
---

# 接口与概念

msOpProf 使用 `msprof op [options] app [arguments]` 拉起应用，在真实 AI 处理器上
多次采集指定算子的性能计数器并自动解析。它支持 Kernel 直调、单算子 API、
PyTorch、Triton、CATLASS 和 MC2 等调用场景，也支持用 `--config` 输入算子
`.o` 文件。[^board-guide][^scenarios]

一次可比较的采集必须绑定以下条件：

```text
commit + kernel binary + shape/dtype/layout + block/grid + device
+ CANN/msOpProf version + metric set + replay mode + warm-up + command
```

先单独运行 correctness 和稳定性基线，再由 msOpProf 拉起完全相同的 workload。
Profiler 成功不代表结果正确，未经 oracle 验证的数据不能进入优化结论。

# 用法

## 最小上板采集

对只启动一个目标 kernel 的程序：

```bash
mkdir -p ./artifacts/profiler
msprof op \
  --output=./artifacts/profiler \
  --aic-metrics=Default \
  ./run_operator 256 512 1024 0
```

对 CATLASS Python/DSL 入口，将解释器和脚本视为应用及参数：

```bash
msprof op \
  --output=./artifacts/profiler \
  --kernel-name="matmul*" \
  --launch-count=1 \
  --aic-metrics=Default \
  python3 run_operator.py
```

若不指定 `--kernel-name`，只采集应用调度的第一个算子；`--launch-count` 默认
为 1。多算子应用应显式记录 kernel 前缀、跳过数量和最大采集数量。[^board-guide]

## CATLASS 模板库流程

CATLASS 模板库的基本矩阵乘样例可先生成上板可执行文件，再从产物目录采集：

```bash
bash scripts/build.sh 00_basic_matmul
cd output/bin
msprof op \
  --output=../../artifacts/profiler \
  ./00_basic_matmul 256 512 1024 0
```

具体产物名和参数以当前仓库为准，不把文档样例中的版本、shape 或 device 当作
项目默认值。[^scenarios]

## 精确筛选与指标选择

筛选顺序是：

```text
--launch-skip-before-match
  -> --mstx 范围
  -> --kernel-name
  -> --aic-metrics
  -> --launch-count / --kill
```

常用指标组合：

| 目标 | 建议起点 |
| --- | --- |
| 常规基线 | `--aic-metrics=Default` |
| 仅落盘基础信息 | `--aic-metrics=BasicInfo` |
| Roofline | `--aic-metrics=Roofline` |
| MTE1/MTE2 活跃带宽和源码搬运 | `--aic-metrics=MemoryDetail` |
| 源码热点 | `--aic-metrics=Source`，编译时带 `-g` |
| 核间负载 | `--aic-metrics=Occupancy` |
| 指定 Kernel 代码段 | `--aic-metrics=KernelScale,...` |

`Default` 包含 ArithmeticUtilization、L2Cache、Memory、MemoryL0、MemoryUB、
PipeUtilization 和 ResourceConflictRatio。扩展指标存在芯片和 replay mode
限制，执行前应以本机 `msprof op --help` 与当前 CANN 文档为准。[^board-guide]

## 输出结构

单卡单算子默认生成：

```text
OPPROF_{timestamp}_XXX/
├── dump/
├── ArithmeticUtilization.csv
├── L2Cache.csv
├── Memory.csv
├── MemoryL0.csv
├── MemoryUB.csv
├── OpBasicInfo.csv
├── PipeUtilization.csv
├── ResourceConflictRatio.csv
└── visualize_data.bin
```

多卡或多算子会增加 Device、OpName 和调度序号层级；不要假定 CSV 固定在结果根
目录。`dump/` 是过程数据，CSV 用于离线核对，`visualize_data.bin` 用于
MindStudio Insight。通算场景还可能生成 `trace.json`。[^board-guide]

# 代码模式

## 可复现采集清单

```bash
python3 -c 'import platform; print(platform.python_version())'
msprof op --help
git rev-parse HEAD
sha256sum ./run_operator
```

在结果旁保存独立 manifest：

```json
{
  "shape": [256, 512, 1024],
  "dtype": "float16",
  "layout": "ND",
  "device": 0,
  "block": 20,
  "metrics": "Default",
  "replay_mode": "kernel",
  "warm_up": 5,
  "correctness": "passed"
}
```

## 多算子最小化

```bash
msprof op \
  --launch-skip-before-match=0 \
  --kernel-name="target_kernel*" \
  --launch-count=1 \
  --warm-up=5 \
  --output=./artifacts/profiler \
  python3 minimal_repro.py
```

优先把应用缩减为单个目标 kernel；仅在无法缩减时使用跳过、名称和范围筛选。

# 约束

- 采集前应用必须能够独立正确运行；同一 Device 不支持同时启动多个采集任务。
- 官方指南建议单次采集控制在 5 分钟内，并为运行环境准备充足内存。[^board-guide]
- 输出目录不应包含软链接；输出父目录、配置文件和可执行文件权限必须可信。
- `LD_LIBRARY_PATH` 只指向可信、权限受控且不含软链接的目录。
- `--replay-mode=kernel` 和 `range` 会清理 L2 cache，`application` 不会；不同
  replay mode 的 L2 数据不能直接横向比较。[^board-guide]
- `--warm-up` 会影响频率稳定性；基线和候选必须使用相同值。
- `--kill=on` 会提前终止应用，可能影响最后一个通算融合算子的流水完整性。
- `-g` 生成的二进制和 profile 结果可能含源码、路径和数据特征，应限制访问。

# 失败表现

| 表现 | 首查 |
| --- | --- |
| 采到错误 kernel | `--kernel-name`、启动顺序、`--launch-skip-before-match` |
| 只生成一个算子 | `--launch-count` 默认值是否仍为 1 |
| 没有扩展指标 | `--aic-metrics` 是否只使用了 Default |
| 结果波动大 | warm-up、频率、并发任务、输入和 replay mode |
| 找不到 CSV | 多卡/多算子目录层级和调度序号 |
| Source/热点图为空 | 编译是否带 `-g`，是否启用 Source，芯片是否支持 |
| MemoryDetail 栏位为 NA | 动态插桩是否成功、产品是否支持 |
| 输出目录权限报错 | 父目录属主、组/其他用户写权限、软链接 |
| 应用被提前停止 | `--kill=on` 或手动 `CTRL+C` |

# 验证方法

```bash
test -d ./artifacts/profiler
find ./artifacts/profiler -name OpBasicInfo.csv -type f -print
find ./artifacts/profiler -name PipeUtilization.csv -type f -print
```

验证记录至少包含完整命令、退出码、`Profiling running finished` 回显、结果目录、
基础 CSV、manifest 和 correctness 结果。候选优化必须与相同采集契约的 baseline
对比。本文只核对固定提交的用户指南，未执行 NPU 采集。

[^board-guide]: 固定提交的 msOpProf 上板模式、参数、输出和约束说明。
[^scenarios]: 固定提交的 CATLASS、Kernel 直调、单算子 API、PyTorch、Triton 与 MC2 采集示例。
