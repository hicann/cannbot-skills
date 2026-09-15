---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_zero_bits_cnt"
description: "标量零比特计数：统计 uint64_t 二进制中 0 的个数并以 int64_t 返回，PIPE_S；Atlas A2/A3 与 Ascend 950PR/950DT 均支持。"
tags: [scalar_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/scalar_compute/asc_zero_bits_cnt.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/scalar_compute/asc_zero_bits_cnt.md

# asc_zero_bits_cnt

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
| Ascend 950PR/Ascend 950DT | √    |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

获取一个uint64_t类型数字的二进制中0的个数。

## 函数原型

```c++
__aicore__ inline int64_t asc_zero_bits_cnt(uint64_t value)
```

## 参数说明
表1 参数说明

|参数名|输入/输出|描述|
| :------ | :--- | :------------ |
|value   |输入   |被统计的二进制数字。

## 返回值说明

value中0的个数。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```c++
uint64_t value = 33;
asc_zero_bits_cnt(value);
// 输出数据count_zero为62
int64_t count_zero = asc_zero_bits_cnt(value);
```
