---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_min_scalar"
description: "reg 层矢量与标量取小 dst_i=min(src_i, value)，8 个重载覆盖 int8/uint8/int16/uint16/int32/uint32/half/float，未选中元素置零，PIPE_V。"
tags: [reg_vector]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/reg/reg_vector/asc_min_scalar.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_min_scalar.md

# asc_min_scalar

## 产品支持情况

| 产品                  | 是否支持  |
|:-------------------------| :------: |
| Ascend 950PR/Ascend 950DT | √    |

## 功能说明

源操作数矢量内每个元素与标量比，如果比标量大，则取标量值，比标量小，则取源操作数。计算公式如下：

$$
dst_i = min(src_i, scalar_i)
$$

## 函数原型

```cpp
__simd_callee__ inline void asc_min_scalar(vector_int32_t& dst, vector_int32_t src, int32_t value, vector_bool mask)
__simd_callee__ inline void asc_min_scalar(vector_uint32_t& dst, vector_uint32_t src, uint32_t value, vector_bool mask)
__simd_callee__ inline void asc_min_scalar(vector_int16_t& dst, vector_int16_t src, int16_t value, vector_bool mask)
__simd_callee__ inline void asc_min_scalar(vector_uint16_t& dst, vector_uint16_t src, uint16_t value, vector_bool mask)
__simd_callee__ inline void asc_min_scalar(vector_int8_t& dst, vector_int8_t src, int8_t value, vector_bool mask)
__simd_callee__ inline void asc_min_scalar(vector_uint8_t& dst, vector_uint8_t src, uint8_t value, vector_bool mask)
__simd_callee__ inline void asc_min_scalar(vector_float& dst, vector_float src, float value, vector_bool mask)
__simd_callee__ inline void asc_min_scalar(vector_half& dst, vector_half src, half value, vector_bool mask)
```

## 参数说明

| 参数名    | 输入/输出 | 描述                             |
|:-------| :--- |:-------------------------------|
| dst | 输出 | 目的操作数（矢量数据寄存器）。                 |
| src | 输入 | 源操作数（矢量数据寄存器）。                  |
| value | 输入 | 源操作数（标量）的起始地址。                  |
| mask   | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

矢量数据寄存器和掩码寄存器的详细说明请参见reg数据类型定义。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_half dst;
vector_half src;
half value;
vector_bool mask = asc_create_mask_b16(PAT_ALL);
asc_loadalign(src, src_addr); // src_addr是外部输入的UB内存空间地址。
asc_min_scalar(dst, src, value, mask);
```