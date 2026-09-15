---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_shiftleft_scalar"
description: "reg 矢量按标量左移 dst_i=src_i<<value，所有类型均为逻辑左移、低位补 0，value 不支持负数；支持 8/16/32 位有无符号整型，按 mask 筛选。"
tags: [reg_vector]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/reg/reg_vector/asc_shiftleft_scalar.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_shiftleft_scalar.md

# asc_shiftleft_scalar

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

根据mask对源操作数src执行左移，左移的位数由输入参数value决定，将结果写入目的操作数dst。
对所有数据类型执行逻辑左移，去掉高位，低位填充为0。

计算公式如下：

$$
dst_i = src_i << value
$$

## 函数原型

```cpp
__simd_callee__ inline void asc_shiftleft_scalar(vector_uint8_t& dst, vector_uint8_t src, int16_t value, vector_bool mask)
__simd_callee__ inline void asc_shiftleft_scalar(vector_int8_t& dst, vector_int8_t src, int16_t value, vector_bool mask)
__simd_callee__ inline void asc_shiftleft_scalar(vector_uint16_t& dst, vector_uint16_t src, int16_t value, vector_bool mask)
__simd_callee__ inline void asc_shiftleft_scalar(vector_int16_t& dst, vector_int16_t src, int16_t value, vector_bool mask)
__simd_callee__ inline void asc_shiftleft_scalar(vector_uint32_t& dst, vector_uint32_t src, int16_t value, vector_bool mask)
__simd_callee__ inline void asc_shiftleft_scalar(vector_int32_t& dst, vector_int32_t src, int16_t value, vector_bool mask)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量数据寄存器）。 |
| src | 输入 | 源操作数（矢量数据寄存器）。 |
| value | 输入 | 源操作数（标量）。不支持设置为负数。 |
| mask | 输入 | 源操作数掩码（掩码寄存器）。用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

矢量数据寄存器和掩码寄存器的详细说明请参见reg数据类型定义.md。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- value不支持设置为负数。

## 调用示例

```cpp
vector_uint16_t dst;
vector_uint16_t src;
int16_t value = 1;
vector_bool mask = asc_create_mask_b16(PAT_ALL);
asc_loadalign(src, src_addr); // src_addr是外部输入的UB内存空间地址。
asc_shiftleft_scalar(dst, src, value, mask);
```