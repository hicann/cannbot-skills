---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_get_rsvd_count"
description: "返回 GatherMask 操作后剩余的元素数量（int64_t）；须与 asc_reduce 配合使用，并通过同步确认 asc_reduce 执行完成后再调用。"
tags: [vector_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/vector_compute/asc_get_rsvd_count.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_get_rsvd_count.md
# asc_get_rsvd_count

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

此接口用于获取执行GatherMask操作后剩余的元素数量。

## 函数原型

```cpp
  __aicore__ inline int64_t asc_get_rsvd_count()
```

## 参数说明

无

## 返回值说明

执行[asc_reduce](asc_reduce.md)操作后剩余的元素数量。

## 流水类型

PIPE_S

## 约束说明

- 需和[asc_reduce](asc_reduce.md)操作配合使用。
- 需通过同步操作确保[asc_reduce](asc_reduce.md)执行完成后再调用本接口获取结果。

## 调用示例

```cpp
// 初始化dst、src0、src1和total_length(参与计算的数据长度)
int total_length = 256;
asc_reduce(dst, src0, src1, total_length)
asc_sync_notify(PIPE_V, PIPE_S, 0); // 设置等待和同步信号
asc_sync_wait(PIPE_V, PIPE_S, 0);
int64_t result = asc_get_rsvd_count();  // 获取结果
```