---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_set_ndim_loop_stride"
description: "为 asc_ndim_copy_gm2ub 设置各维度源/目的操作数元素间隔，asc_set_ndim_loop0~4_stride 最多 5 维，单位为元素个数，dst 范围 [0,2^20-1]、src 范围 [0,2^40-1]，PIPE_S。Ascend 950PR/950DT 支持。"
tags: [vector_datamove]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/vector_datamove/asc_set_ndim_loop_stride.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_datamove/asc_set_ndim_loop_stride.md
# asc_set_ndim_loop_stride

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| Ascend 950PR/Ascend 950DT | √    |

## 功能说明

用于设置[asc_ndim_copy_gm2ub](asc_ndim_copy_gm2ub.md)接口，每个维度内的源操作数与目的操作数的元素之间的间隔，最多设置5个维度。

## 函数原型

```cpp
__aicore__ inline void asc_set_ndim_loop0_stride(uint64_t dst_stride, uint64_t src_stride)
__aicore__ inline void asc_set_ndim_loop1_stride(uint64_t dst_stride, uint64_t src_stride)
__aicore__ inline void asc_set_ndim_loop2_stride(uint64_t dst_stride, uint64_t src_stride)
__aicore__ inline void asc_set_ndim_loop3_stride(uint64_t dst_stride, uint64_t src_stride)
__aicore__ inline void asc_set_ndim_loop4_stride(uint64_t dst_stride, uint64_t src_stride)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst_stride | 输入 | 目的操作数的元素之间的间隔，单位为元素个数，默认值：0。取值范围：[0, 2^20 - 1]。 |
| src_stride | 输入 | 源操作数的元素之间的间隔，单位为元素个数，默认值：0。取值范围：[0, 2^40 - 1]。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

需配合[asc_ndim_copy_gm2ub](asc_ndim_copy_gm2ub.md)使用。

## 调用示例

```cpp
uint64_t dst_stride = 8;
uint64_t src_stride = 8;
asc_set_ndim_loop0_stride(dst_stride, src_stride);
```