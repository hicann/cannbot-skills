---
name: ascendc-npu-arch
description: Ascend NPU 架构知识查询技能。通过芯片型号映射、架构代际划分和 arch35 特性说明，帮助判断目标平台能力、特性支持与条件编译策略。当需要确认芯片型号、NpuArch/SocVersion、架构差异、特性支持或编译分支条件时使用。
---

# Ascend NPU 架构知识

## 架构代际概述

| 概念 | 说明 |
|-----|------|
| **NpuArch** | 芯片架构号，定义指令集和微架构，运行时通过 `GetCurNpuArch()` 获取 |
| **SocVersion** | 片上系统版本，软件命名标识，运行时通过 `GetSocVersion()` 获取 |
| **__NPU_ARCH__** | Device 侧编译宏，四位数值，用于条件编译 |

## 完整映射表

| NpuArch | __NPU_ARCH__ | SocVersion | 产品系列 | 芯片型号 |
|---------|-------------|------------|---------|---------|
| DAV_1001 | 1001 | ASCEND 910 | Atlas 训练系列 | Ascend 910 |
| DAV_2002 | 2002 | ASCEND 310P | Atlas 推理系列 | Ascend 310P1, Ascend 310P3 |
| DAV_2201 | 2201 | ASCEND 910B | Atlas A2 训练/推理系列 | Ascend 910B1~B4, Ascend 910B2C |
| DAV_2201 | 2201 | ASCEND 910B | Atlas A3 训练/推理系列 | Ascend 910_93 |
| DAV_3002 | 3002 | ASCEND 310B | Atlas 200I/500 A2 推理产品 | Ascend 310B1~B4 |
| DAV_3510 | 3510 | ASCEND 950 | Atlas A5 训练系列 | Ascend 950DT (Decode) |
| DAV_3510 | 3510 | ASCEND 950 | Atlas A5 推理系列 | Ascend 950PR (Prefill) |

> **注意**：`Ascend910_93` 的 SocVersion 字符串在运行时映射到 `SocVersion::ASCEND910B`（非独立枚举值），NpuArch 同为 DAV_2201。

## 获取当前架构

```cpp
#include "platform/soc_spec.h"
#include "utils/tiling/platform/platform_ascendc.h"

auto platformInfo = context->GetPlatformInfo();
auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
platform_ascendc::SocVersion socVer = ascendcPlatform.GetSocVersion();
```

- `GetCurNpuArch()` 返回 `NpuArch` 枚举，失败时返回 `NpuArch::DAV_RESV`
- `GetSocVersion()` 返回 `SocVersion` 枚举，失败时返回 `SocVersion::RESERVED_VERSION`

## Ascend 950 (DAV_3510) 独有特性

| 特性 | 说明 | 典型算子 |
|-----|------|---------|
| Regbase 编程 | 直接操作寄存器 | 量化算子 |
| SIMT 编程 | 线程级并行 | 随机数、排序 |
| FP8 格式 | 8-bit 浮点 | 量化、动态量化 |

## 详细文档

- [完整架构指南](references/npu-arch-guide.md)
