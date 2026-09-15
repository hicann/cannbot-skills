---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_get_ctrl"
description: "读取 CTRL 控制寄存器特定比特位上的值，无参返回 int64_t，各比特位含义参见 asc_set_ctrl，流水类型 PIPE_S，支持 950PR/950DT 与 Atlas A3/A2。"
tags: [sys_var]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/sys_var/asc_get_ctrl.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_get_ctrl.md

# asc_get_ctrl

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |

## 功能说明

读取CTRL寄存器（控制寄存器）特定比特位上的值。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_ctrl()
```

## 参数说明

无

## 返回值说明

CTRL寄存器的值。具体含义请参考[asc_set_ctrl](asc_set_ctrl.md)中的描述。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

无