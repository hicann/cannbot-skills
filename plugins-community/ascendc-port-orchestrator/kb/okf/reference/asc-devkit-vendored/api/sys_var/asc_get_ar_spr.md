---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_get_ar_spr"
description: "读取指定特殊寄存器的值，无参调用并返回 int64_t 数值，流水类型 PIPE_S，产品支持表仅列 Ascend 950PR/950DT。"
tags: [sys_var]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/sys_var/asc_get_ar_spr.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_get_ar_spr.md

# asc_get_ar_spr

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| Ascend 950PR/Ascend 950DT |    √    |

## 功能说明

读取指定特殊寄存器的值。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_ar_spr()
```

## 参数说明

无

## 返回值说明

返回int64_t类型的特殊寄存器中的数值。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

无
