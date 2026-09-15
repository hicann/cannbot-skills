---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_xor"
description: "reg 矢量按元素异或 dst=src0^src1，支持 int8/uint8/int16/uint16/int32/uint32/bool 寄存器形态，按 mask 筛选；支持 Ascend 950PR/950DT。"
tags: [reg_vector]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/reg/reg_vector/asc_xor.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_xor.md

# asc_xor

## 产品支持情况

| 产品                  | 是否支持  |
|:-------------------------| :------: |
| Ascend 950PR/Ascend 950DT | √    |

## 功能说明

根据mask对输入的src0、src1按元素异或（^）进行操作，将结果写入dst。

## 函数原型

```cpp
__simd_callee__ inline void asc_xor(vector_int32_t& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask)
__simd_callee__ inline void asc_xor(vector_uint32_t& dst, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask)
__simd_callee__ inline void asc_xor(vector_uint16_t& dst, vector_uint16_t src0, vector_uint16_t src1, vector_bool mask)
__simd_callee__ inline void asc_xor(vector_int16_t& dst, vector_int16_t src0, vector_int16_t src1, vector_bool mask)
__simd_callee__ inline void asc_xor(vector_uint8_t& dst, vector_uint8_t src0, vector_uint8_t src1, vector_bool mask)
__simd_callee__ inline void asc_xor(vector_int8_t& dst, vector_int8_t src0, vector_int8_t src1, vector_bool mask)
__simd_callee__ inline void asc_xor(vector_bool& dst, vector_bool src0, vector_bool src1, vector_bool mask)
```

## 参数说明

| 参数名      | 输入/输出 | 描述                             |
|:---------| :--- |:-------------------------------|
| dst      | 输出 | 目的操作数（矢量数据寄存器）。                |
| src0     | 输入 | 源操作数（矢量数据寄存器）。                                                       |
| src1     | 输入 | 源操作数（矢量数据寄存器）。                                                            |
| mask     | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

矢量数据寄存器和掩码寄存器的详细说明请参见reg数据类型定义

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_int8_t dst;
vector_int8_t src0;
vector_int8_t src1;
vector_bool mask = asc_create_mask_b8(PAT_ALL);
asc_loadalign(src0, src0_addr); // src0_addr是外部输入的UB内存空间地址。
asc_loadalign(src1, src1_addr); // src1_addr是外部输入的UB内存空间地址。
asc_xor(dst, src0, src1, mask);
```