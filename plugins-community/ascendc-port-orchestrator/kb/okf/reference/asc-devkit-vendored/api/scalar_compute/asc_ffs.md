---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_ffs"
description: "标量 FindFirstSet：从最低位向高位查 uint64_t 中第一个为 1 的位并返回其位置，找不到返回 -1，PIPE_S；A2/A3 与 950PR/950DT 均支持。"
tags: [scalar_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/scalar_compute/asc_ffs.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/scalar_compute/asc_ffs.md

# asc_ffs

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

FindFirstSet接口，输入数据的二进制表示中从最低位向最高位查找第一个值为1的位，并返回其位置，如果没找到则返回-1。‌

## 函数原型

```c++
__aicore__ inline int64_t asc_ffs(uint64_t value)
```

## 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| value     | 输入     | 输入数据。|

## 返回值说明

int64_t类型，输入数据中第一个1出现的位置。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```c++
uint64_t value = 10;
int64_t ret = asc_ffs(value);
```