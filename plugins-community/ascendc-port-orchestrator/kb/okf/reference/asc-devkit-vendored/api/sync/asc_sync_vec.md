---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_sync_vec"
description: "对所有流水线执行同步操作的无参接口 asc_sync_vec()，无参数无返回值，流水类型 PIPE_TYPE_S，支持 Ascend 950PR/950DT 与 Atlas A3/A2 训练推理系列产品。"
tags: [sync]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/sync/asc_sync_vec.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/sync/asc_sync_vec.md

# asc_sync_vec

## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
| <cann-filter npu_type="950">Ascend 950PR/Ascend 950DT | √    </cann-filter>|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

针对所有流水线执行同步操作。

## 函数原型

```cpp
__aicore__ inline void asc_sync_vec()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

PIPE_TYPE_S

## 约束说明

无

## 调用示例

```cpp
asc_sync_vec();
```
