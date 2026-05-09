# NPU 架构代际说明

本文档说明 Ascend NPU 的架构代际划分及其对算子开发的影响。

---

## 目录

1. [架构代际概述](#架构代际概述)
2. [完整映射表](#完整映射表)
3. [开发指导](#开发指导)
4. [Ascend950 (DAV_3510) 特殊优化](#ascend950-dav_3510-特殊优化)
5. [架构兼容性检查清单](#架构兼容性检查清单)
6. [参考信息来源](#参考信息来源)

---

## 架构代际概述

### 核心概念

| 概念 | 说明 |
|-----|------|
| **NpuArch** | 芯片架构号，定义指令集和微架构，运行时通过 `GetCurNpuArch()` 获取 |
| **SocVersion** | 片上系统版本，软件命名标识，运行时通过 `GetSocVersion()` 获取 |
| **__NPU_ARCH__** | Device 侧编译宏，四位数值，用于条件编译 |
| **archXX** | 算子仓架构目录简写，取 DAV 编号前两位，如 arch22、arch35 |

### 架构目录简写（archXX）

算子仓中按架构划分的目录使用 `archXX` 命名，取 `DAV_XXXX` 前两位数字：

| 目录 | 对应 NpuArch | 芯片 |
|------|-------------|------|
| **arch22** | DAV_2201 | Ascend910B 系列、Ascend910_93 |
| **arch35** | DAV_3510 | Ascend950DT / Ascend950PR |

> 命名规则：`archXX` = `arch` + DAV 编号前两位。如 DAV_**22**01 → arch22，DAV_**35**10 → arch35。

### 架构代号别名

| 代号 | 对应 SocVersion | 对应 NpuArch | 说明 |
|-----|----------------|-------------|------|
| **A2** | ASCEND910B | DAV_2201 | Ascend910B1~B4, Ascend910B2C |
| **A3** | ASCEND910B (含 Ascend910_93) | DAV_2201 | 训练/推理芯片 |
| **A5** | ASCEND950 | DAV_3510 | Ascend950DT (Decode) / Ascend950PR (Prefill) |

**核心关系：一对多**

一个 NpuArch 可以对应多个 SocVersion。例如 `DAV_2201` 对应 Ascend910B1~B4、Ascend910B2C、Ascend910_93。

> **注意**：对 NPU 核内算子开发来说，通常不需要感知具体 SocVersion，使用 NpuArch 来区分芯片有利于易用性和可维护性。

### 关键细节

- `Ascend910_93` 的 SocVersion 字符串在运行时映射到 `SocVersion::ASCEND910B`（非独立枚举值），NpuArch 同为 DAV_2201
- 源码中存在 `SocVersion::ASCEND910_93` 枚举值（platform_ascendc.h），但 convertMap 中 `"Ascend910_93"` 映射到的是 `ASCEND910B`，该枚举仅在少数内部模块使用
- `DAV_RESV` 是 `GetCurNpuArch()` 的错误返回值：获取失败、字符串转换失败或值 <= 0 时返回
- `__DAV_C310__` 是代码内部命名，对应 NpuArch DAV_3510（Ascend950），不可按数值推断

---

## 完整映射表

| 产品系列 | SocVersion | NpuArch | __NPU_ARCH__ | 芯片型号 | 说明 |
|---------|-----------|---------|:---:|---------|------|
| Atlas A2 训练/推理 | ASCEND910B | DAV_2201 | 2201 | Ascend910B1~B4, Ascend910B2C | 主流训练芯片 |
| Atlas A3 训练/推理 | ASCEND910B | DAV_2201 | 2201 | Ascend910_93 | 训练/推理芯片 |
| Atlas A5 训练 | ASCEND950 | DAV_3510 | 3510 | Ascend950DT | 新一代 Decode |
| Atlas A5 推理 | ASCEND950 | DAV_3510 | 3510 | Ascend950PR | 新一代 Prefill |
| Atlas 训练系列 | ASCEND910 | DAV_1001 | 1001 | Ascend910 | 初代训练架构 |
| Atlas 推理系列 | ASCEND310P | DAV_2002 | 2002 | Ascend310P1, Ascend310P3 | 推理芯片 |
| Atlas 200I/500 A2 推理 | ASCEND310B | DAV_3002 | 3002 | Ascend310B1~B4 | 推理芯片 |

---

## 开发指导

### 算子目录结构

```
op_name/
├── op_kernel/
│   ├── *.cpp / *.h          # 通用实现（适用于所有架构）
│   ├── arch22/              # 特定于 DAV_2201 的优化（可选）
│   │   └── *_dag.h
│   └── arch35/              # Ascend950 专用优化
│       ├── *_dag.h          # 针对 950 优化的 DAG 定义
│       └── *_struct.h       # 针对 950 的模板参数声明
├── op_host/
│   ├── *tiling*.cpp         # 通用 Tiling 实现
│   └── arch35/              # Ascend950 专用 Tiling
│       └── *_tiling_arch35.cpp
└── examples/
    ├── test_*.cpp           # 通用测试
    └── arch35/              # 950 专用测试
        └── test_*.cpp
```

### 获取当前架构

**Kernel 端（Tiling 阶段）**：

```cpp
#include "platform/soc_spec.h"
#include "utils/tiling/platform/platform_ascendc.h"

auto platformInfo = context->GetPlatformInfo();
auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
platform_ascendc::SocVersion socVer = ascendcPlatform.GetSocVersion();
```

**Host 端（aclnn Tiling）**：

```cpp
#include "platform/soc_spec.h"
#include "utils/tiling/platform/platform_ascendc.h"

auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
bool IsArch35 = npuArch == NpuArch::DAV_3510;
```

### 架构条件编译

```cpp
template <bool isArch35>
__aicore__ inline void ProcessArchSpecific() {
    if constexpr (isArch35) {
        // Ascend950 特定优化
    } else {
        // 通用实现
    }
}
```

**Tiling 中选择实现**：

```cpp
if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
    tilingData.SetTilingKey(ARCH35_TILING_KEY);
} else {
    tilingData.SetTilingKey(GENERAL_TILING_KEY);
}
```

### 编译配置

```cmake
# Ascend910B (DAV_2201)
set(SOC_VERSION "Ascend910B")

# Ascend950 (DAV_3510)
set(SOC_VERSION "Ascend950")
```

### 文件命名约定

| 文件类型 | 通用 | arch35 专用 |
|---------|------|-------------|
| DAG 定义 | `*_dag.h` | `arch35/*_dag.h` |
| Tiling | `*_tiling.cpp` | `arch35/*_tiling_arch35.cpp` |
| Kernel 入口 | `*_apt.cpp` | 同一文件内通过 TilingKey 区分 |
| 测试 | `test_*.cpp` | `arch35/test_*.cpp` |

---

## Ascend950 (DAV_3510) 特殊优化

| 特性 | 说明 | 典型算子 |
|-----|------|---------|
| **Regbase 编程** | 直接操作寄存器，更高性能 | 量化算子 |
| **SIMT 编程** | 线程级并行编程模型 | 随机数、排序 |
| **FP8 格式** | 8-bit 浮点格式 | 量化、动态量化 |

### Regbase 编程

```cpp
// 仅在 Ascend950 (DAV_3510) 可用
template <typename T, typename T1, typename T2, typename U, ...>
class QuantizePerChannelRegbase : public QuantizeBase<T, T1, T2, U, ...> {
    __aicore__ inline void Init(...) {
        this->SetFloatOverflowModeForRegbase();  // 950 特有
    }
};
```

### SIMT 编程

```cpp
// 仅在 Ascend950 (DAV_3510) 可用
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void SimtCompute(...) {
    int64_t groupIndex = blockIdx.x * blockDim.x + threadIdx.x;
}

asc_vf_call<SimtCompute<Y_T, OFFSET_T>>(dim3{USED_THREAD}, ...);
```

**SIMT API**：`blockIdx.x`, `threadIdx.x`, `blockDim.x`, `asc_vf_call<Func>()`, `asc_atomic_add`, `asc_syncthreads`

### FP8/FP4 低精度格式

| 格式 | 类型名 | 说明 |
|-----|-------|------|
| **FP8 E5M2** | `fp8_e5m2_t` | 5位指数，2位尾数 |
| **FP8 E4M3FN** | `fp8_e4m3fn_t` | 4位指数，3位尾数 |
| **HiFloat8** | `hifloat8_t` | 华为自定义格式 |
| **INT4** | `int4b_t` | 4-bit 整数 |

---

## 架构兼容性检查清单

开发算子时，请确认：

- [ ] 通用实现在所有目标架构上测试通过
- [ ] 如有 arch35 特殊实现，已单独测试
- [ ] Tiling 逻辑正确识别架构并选择实现
- [ ] 性能在目标架构上达到基线要求

---

## 参考信息来源

### 源码文件

| 文件 | 内容 |
|------|------|
| `include/utils/tiling/platform/platform_ascendc.h` | `SocVersion` 枚举完整定义（24 个值） |
| `impl/utils/tiling/platform/platform_ascendc.cpp` | `convertMap`（字符串到 SocVersion 映射）、`GetCurNpuArch()` 实现 |
| `docs/api/context/GetCurNpuArch.md` | GetCurNpuArch API 文档，含产品-NpuArch 对应表 |

### 外部参考

| 资源 | 说明 |
|------|------|
| `include/platform/soc_spec.h`（CANN 安装路径） | NpuArch 枚举的完整定义 |
