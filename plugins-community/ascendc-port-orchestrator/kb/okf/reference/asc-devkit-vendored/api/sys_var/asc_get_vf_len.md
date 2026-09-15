---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_get_vf_len"
description: "获取 Tensor 位宽 VL（Vector Length）的大小，无参返回 int64_t，流水类型 PIPE_S，产品支持表仅列 Ascend 950PR/950DT。"
tags: [sys_var]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/sys_var/asc_get_vf_len.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_get_vf_len.md

# asc_get_vf_len

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |

## 功能说明

获取Tensor位宽VL（Vector Length）的大小。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_vf_len()
```

## 参数说明

无

## 返回值说明

位宽VL（Vector Length）的大小。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
int64_t len = asc_get_vf_len();
```
