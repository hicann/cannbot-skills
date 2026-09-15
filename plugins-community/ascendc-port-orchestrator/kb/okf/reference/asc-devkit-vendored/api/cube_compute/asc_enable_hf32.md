---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_enable_hf32"
description: "开启 Mmad 的 HF32 模式，开启后 L0A/L0B Buffer 中的 FP32 数据在参与 Mmad 计算前先被舍入为 HF32；无参数无约束，支持 Atlas A3 与 A2 训练/推理系列产品。"
tags: [cube_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/cube_compute/asc_enable_hf32.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_compute/asc_enable_hf32.md


# asc_enable_hf32

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

### 功能说明

用于设置Mmad计算开启HF32模式，开启该模式后L0A Buffer/L0B Buffer中的FP32数据将在参与Mmad计算之前被舍入为HF32。


### 函数原型


 ```cpp
__aicore__ inline void asc_enable_hf32()
```


### 返回值说明

无

### 流水类型

PIPE_S

### 约束说明

无

### 调用示例

```cpp
asc_enable_hf32();
```