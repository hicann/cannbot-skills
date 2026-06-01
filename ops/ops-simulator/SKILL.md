---
name: ops-simulator
description: NPU 仿真器技能。提供 CANN Simulator 的使用指导，包括精度仿真、性能仿真、流水线分析。当需要在无 NPU 硬件环境下验证算子功能、分析性能瓶颈、定位精度问题时使用。
---

# NPU 仿真器使用指南

## 概述

CANN Simulator 是一款面向算子开发场景的 SoC 级芯片仿真工具，通过 `cannsim` 命令行工具提供以下能力：

- **精度仿真**：输出 bit 级精度结果，协助算子精度验证
- **性能仿真**：输出指令流水图，协助定位性能瓶颈


### cannsim 主命令

`cannsim` 是性能仿真分析的命令行入口，提供两个子命令：

| 子命令 | 功能 | 说明 |
|--------|------|------|
| `cannsim record` | 执行仿真 | 在仿真环境中运行用户程序，记录仿真数据 |
| `cannsim report` | 生成报告 | 基于仿真结果生成性能分析报告和流水线图 |

**使用方式**：`cannsim <子命令> [选项]`

## 适用场景

| 场景 | 说明 |
|------|------|
| 无 NPU 硬件环境 | 在没有真实 NPU 硬件的情况下进行算子开发 |
| 精度验证 | 需要 bit 级精度验证的场景 |
| 性能调优 | 需要分析指令流水、定位性能瓶颈 |
| 资源受限 | 芯片资源紧缺时的替代验证方案 |

### 使用约束

| 约束项 | 说明 |
|--------|------|
| 芯片限制 | 仅支持 Ascend 950 芯片架构 |
| 单卡场景 | 仅支持单卡，代码中只能设置为 0 卡 |
| 算子类型 | 仅支持 AI Core 计算类算子（不支持 MC2/HCCL） |
| 架构限制 | 支持 X86，ARM |

## 使用步骤

### 1. 执行仿真

```bash
# 精度仿真 + 性能仿真（生成报告）
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report

# 指定输出目录
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report -o ./sim_output

# 传递算子自定义参数
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report -u "--shape 1024,1024 --dtype float16"
```

### 2. 生成性能报告

```bash
# 从仿真结果生成流水线报告（默认当前目录）
cannsim report -e ./cannsim_Ascend950_*

# 指定输出目录
cannsim report -e ./cannsim_Ascend950_* -o ./report_output

# 指定查看的 Core ID
cannsim report -e ./cannsim_Ascend950_* -n 0         # 查看单个 core
cannsim report -e ./cannsim_Ascend950_* -n 0-2       # 查看 core 范围
cannsim report -e ./cannsim_Ascend950_* -n 0-2,5,12-14  # 混合格式
cannsim report -e ./cannsim_Ascend950_* -n all       # 查看所有 core
```

### 3. 查看输出文件

```
cannsim_output/
├── cannsim.log               # 仿真执行日志
└── report/
    ├── trace_core0.json      # 指令流水图文件
    └── ...
```

### 4. 性能瓶颈定位（Trace 空泡分析）

cannsim 生成的 `trace_core*.json` 是 Chrome Trace Format，包含每个 core 各 pipeline（SCALAR/VECTOR/MTE2/CUBE 等）的指令级周期事件。按 `references/pipeline-bubble-analysis.md` 的判定标准自主分析：

**分析目标**：定位主 pipeline（VECTOR/CUBE）idle 空泡的根因。

**核心方法**：
1. **空泡分类**：按 7 大类 / ~30 子类型标准判定等待原因（DATA_STALL、SCALAR_OVERHEAD、STRUCTURAL 等）
2. **因果归因**：检查"主 pipeline idle 时谁在 busy"，追踪具体阻塞源
3. **周期性模式检测**：判断空泡是系统性（每个 tile 重复）还是偶发
4. **重叠度判定**：验证双缓冲是否生效（MTE2 与 VECTOR/CUBE 时间重叠比例）

## 命令参考

### cannsim record - 执行仿真

在 AscendOps 仿真环境中运行用户程序，记录仿真数据。

**基本语法**：`cannsim record <user_app> -s <SOC_VERSION> [选项]`

| 参数 | 简写 | 必填 | 说明 |
|------|------|------|------|
| `user_app` | - | 是 | 用户编译后的可执行程序路径 |
| `--soc-version` | `-s` | 是 | 目标芯片版本（如 Ascend950） |
| `--gen-report` | `-g` | 否 | 仿真结束后生成性能报告 |
| `--output` | `-o` | 否 | 仿真结果输出目录 |
| `--user-option` | `-u` | 否 | 传递给用户程序的自定义参数 |

**示例**：
```bash
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report -o ./sim_output
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report -u "--shape 1024,1024"
```

### cannsim report - 生成性能报告

基于仿真结果生成可视化的性能分析报告和指令流水线图。

**基本语法**：`cannsim report -e <EXPORT_FOLDER> [选项]`

| 参数 | 简写 | 必填 | 说明 |
|------|------|------|------|
| `--export` | `-e` | 是 | 仿真结果文件夹路径（cannsim record 的输出） |
| `--output` | `-o` | 否 | 流水线图输出目录 |
| `--core-id` | `-n` | 否 | 指定 Core ID（支持 0、0-2、0-2,5、all 等格式） |

**示例**：
```bash
cannsim report -e ./cannsim_Ascend950_*
cannsim report -e ./cannsim_Ascend950_* -n 0-2
cannsim report -e ./cannsim_Ascend950_* -n all -o ./report_output
```

> 详细命令参数、输出目录结构、返回值说明见 [references/simulator-advanced.md](references/simulator-advanced.md)

## 常见问题

| 问题 | 解决方法 |
|------|---------|
| 仿真失败 | 确保代码中只设置 0 卡，仅使用 AI Core 计算算子 |
| 性能报告未生成 | 确保使用 `--gen-report` 参数 |
| 找不到仿真结果 | 使用 `-o` 指定输出目录，或检查当前目录下的 `cannsim_*` 文件夹 |

## 参考资料

| 文件 | 内容 | 何时查阅 |
|------|------|---------|
| `references/simulator-advanced.md` | 仿真进阶命令参考 | 需要高级参数或批量仿真时 |
| `references/troubleshooting.md` | 问题排查指南 | 仿真失败或报告未生成时 |
| `references/pipeline-bubble-analysis.md` | 指令流水空泡分类、因果归因、周期性模式检测 | 生成 trace 后，定位性能瓶颈根因时 |
| `scripts/trace_bubble_analyzer.py` | 自动化空泡分析脚本（兼容 msprof/cannsim 双格式） | 批量分析多核 trace 或获取预分析报告时 |
