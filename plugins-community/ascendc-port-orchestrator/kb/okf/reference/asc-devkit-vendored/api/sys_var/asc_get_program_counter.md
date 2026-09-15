---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_get_program_counter"
description: "获取程序计数器指针（记录当前程序执行位置），无参返回 int64_t，流水类型 PIPE_TYPE_S，支持 950PR/950DT 与 Atlas A3/A2。"
tags: [sys_var]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/sys_var/asc_get_program_counter.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_get_program_counter.md

# asc_get_program_counter

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| <cann-filter npu_type="950"><term>Ascend 950PR/Ascend 950DT</term>  | √ </cann-filter>|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |

## 功能说明

获取程序计数器的指针，程序计数器用于记录当前程序执行的位置。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_program_counter()
```

## 参数说明

无

## 返回值说明

返回int64_t类型的程序计数器指针。

## 流水类型

PIPE_TYPE_S

## 约束说明

无

## 调用示例

```cpp
asc_get_program_counter();
```