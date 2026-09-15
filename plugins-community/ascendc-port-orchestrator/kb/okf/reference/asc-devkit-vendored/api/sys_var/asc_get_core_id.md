---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_get_core_id"
description: "获取当前核的编号，无参调用返回 int64_t 的核编号，流水类型 PIPE_S，支持 Atlas A3/A2 训练推理系列产品与 Ascend 950PR/950DT。"
tags: [sys_var]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/sys_var/asc_get_core_id.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_get_core_id.md

# asc_get_core_id

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √ </cann-filter>|

## 功能说明

获取当前核的编号。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_core_id()
```

## 参数说明

无

## 返回值说明

返回当前核的编号。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

无