---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_set_cmp_mask"
description: "设置作用于 asc_select 的选择 Mask 掩码，原型为 asc_set_cmp_mask(__ubuf__ void* sel_mask)，须与 asc_select 配合使用；标注支持 Atlas A3/A2。"
tags: [vector_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/vector_compute/asc_set_cmp_mask.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_set_cmp_mask.md
# asc_set_cmp_mask

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

为[asc_select](asc_select.md)操作设置作用于选择的Mask掩码。

## 函数原型

```cpp
__aicore__ inline void asc_set_cmp_mask(__ubuf__ void* sel_mask)
```

## 参数说明

|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
|sel_mask|输出|用于选择的Mask掩码的起始地址。|

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

需和[asc_select](asc_select.md)操作配合使用。

## 调用示例

```cpp
constexpr uint32_t total_length = 256;
constexpr uint32_t sel_length = 8;
__ubuf__ uint8_t sel_mask[sel_length];
__ubuf__ float dst[total_length];
__ubuf__ float src0[total_length];
__ubuf__ float src1[total_length];
asc_set_cmp_mask(sel_mask);
asc_sync_pipe(PIPE_V);
asc_select(dst, src0, src1, total_length);
```
