---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_scatter"
description: "按目的地址偏移张量 index 把连续输入 src 分散写入 dst 的 SIMD 接口（vector_uint16_t index 加 vector_bool mask），覆盖 int8/uint8/int16/uint16 等约 9 个重载；仅 950PR/950DT 支持。"
tags: [reg_load]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/reg/reg_load/asc_scatter.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_load/asc_scatter.md

# asc_scatter

## 产品支持情况

| 产品                  | 是否支持  |
|:-------------------------| :------: |
| Ascend 950PR/Ascend 950DT | √    |

## 功能说明

给定一个连续的输入张量和一个目的地址偏移张量，Scatter指令根据偏移地址生成新的结果张量后将输入张量分散到结果张量中。

## 函数原型

```cpp
__simd_callee__ inline void asc_scatter(vector_int8_t& dst, __ubuf__ int8_t* src, vector_uint16_t index, vector_bool mask)
__simd_callee__ inline void asc_scatter(vector_uint8_t& dst, __ubuf__ uint8_t* src, vector_uint16_t index, vector_bool mask)
__simd_callee__ inline void asc_scatter(vector_int16_t& dst, __ubuf__ int16_t* src, vector_uint16_t index, vector_bool mask)
__simd_callee__ inline void asc_scatter(vector_uint16_t& dst, __ubuf__ uint16_t* src, vector_uint16_t index, vector_bool mask)
__simd_callee__ inline void asc_scatter(vector_int32_t& dst, __ubuf__ int32_t* src, vector_uint32_t index, vector_bool mask)
__simd_callee__ inline void asc_scatter(vector_uint32_t& dst, __ubuf__ uint32_t* src, vector_uint32_t index, vector_bool mask)
__simd_callee__ inline void asc_scatter(vector_bf16& dst, __ubuf__ bfloat16_t* src, vector_uint16_t index, vector_bool mask)
__simd_callee__ inline void asc_scatter(vector_half& dst, __ubuf__ half* src, vector_uint16_t index, vector_bool mask)
__simd_callee__ inline void asc_scatter(vector_float& dst, __ubuf__ float* src, vector_uint32_t index, vector_bool mask)
```

## 参数说明

| 参数名   | 输入/输出 | 描述                                                                      |
|:------| :--- |:------------------------------------------------------------------------|
| dst   | 输出 | 目的操作数（矢量数据寄存器）。                                                           |
| src   | 输入 | 源操作数（矢量）。                                                          |
| index | 输入 | 源操作数（标量）的起始地址。                                                          |
| mask  | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

掩码寄存器的详细说明请参见reg数据类型定义

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_half dst;
__ubuf__ half* src = (__ubuf__ half*)asc_get_phy_buf_addr(0);
vector_uint16_t index;
vector_bool mask = asc_create_mask_b16(PAT_ALL);
asc_loadalign(src, src_addr); // src_addr是外部输入的UB内存空间地址。
asc_scatter(dst, src, index, mask);
```