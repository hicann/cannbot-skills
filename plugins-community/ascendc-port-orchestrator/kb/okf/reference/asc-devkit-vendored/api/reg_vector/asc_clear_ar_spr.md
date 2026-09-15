---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_clear_ar_spr"
description: "清理 AR 寄存器的接口 asc_clear_ar_spr()，无入参无返回值；AR 寄存器通常由 asc_squeeze 接口使用；仅 950PR/950DT 支持。"
tags: [reg_vector]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/reg/reg_vector/asc_clear_ar_spr.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_clear_ar_spr.md

# asc_clear_ar_spr

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

对AR寄存器进行清理，AR寄存器通常由[asc_squeeze](asc_squeeze.md)接口使用。

## 函数原型

```cpp
__simd_callee__ inline void asc_clear_ar_spr()
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

```cpp
asc_clear_ar_spr();
```