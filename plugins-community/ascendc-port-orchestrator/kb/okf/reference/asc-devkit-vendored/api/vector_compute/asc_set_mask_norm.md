---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_set_mask_norm"
description: "把 Mask 切回 Normal（系统默认）模式：该模式下 mask 参数按位控制高维切分计算单次迭代内参与计算的元素；需与 asc_set_vector_mask 配合使用。"
tags: [vector_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/vector_compute/asc_set_mask_norm.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_set_mask_norm.md
# asc_set_mask_norm

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

设置Mask模式为Normal模式，该模式为系统默认模式。该模式下，Mask参数可控制高维切分计算过程中单次迭代内参与计算的元素。

## 函数原型

```cpp
__aicore__ inline void asc_set_mask_norm()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

该API需和[asc_set_vector_mask](asc_set_vector_mask.md)配合使用。

## 调用示例

```cpp
asc_set_mask_norm();
asc_set_vector_mask(0xffffffffffffffff, 0xffffffffffffffff);    // 设置每个迭代内所有元素均参与计算。

// 进行高维切分计算
constexpr uint32_t total_length = 128; //total_length指参与计算的数据总长度
_ubuf__ half src0[total_length];
_ubuf__ half src1[total_length];
_ubuf__ half dst[total_length];
asc_add(dst, src0, src1, config);
```