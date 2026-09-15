---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_init"
description: "初始化 NPU 状态的接口 asc_init()，无入参无返回值、无约束说明；950PR/950DT 与 Atlas A3/A2 均支持。"
tags: [misc]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/misc/asc_init.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/misc/asc_init.md

# asc_init

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |

## 功能说明

初始化NPU状态。

## 函数原型

```cpp
__aicore__ inline void asc_init()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

无