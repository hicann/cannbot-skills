---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_clear_nthbit"
description: "标量位操作：把 uint64_t 的第 idx 位清 0 并返回修改后的值，PIPE_S；Atlas A2/A3 与 Ascend 950PR/950DT 均支持。"
tags: [scalar_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/scalar_compute/asc_clear_nthbit.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/scalar_compute/asc_clear_nthbit.md

# asc_clear_nthbit

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √ </cann-filter>|

## 功能说明

位操作函数，用于将一个uint64_t整数bits的第idx位设置为0。

## 函数原型

```cpp
__aicore__ inline uint64_t asc_clear_nthbit(uint64_t bits, int64_t idx)
```

## 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| bits    | 输入     | 表示需要修改的值。   |
| idx     | 输入     | 位索引，表示需要设置为0的位的位置。|

## 返回值说明

修改后的uint64_t整数。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
uint64_t bits = 0x0;
int64_t  idx = 0x2;
uint64_t res = asc_clear_nthbit(bits, idx);
```