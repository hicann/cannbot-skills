---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_pack"
description: "reg 矢量位宽压缩：取源元素低 8/16/32 位，asc_pack 写入目的寄存器低半部分、asc_pack_v2 写高半部分；支持 16→8、32→16 位整型及 bool；支持 950PR/950DT。"
tags: [reg_vector]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/reg/reg_vector/asc_pack.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_pack.md

# asc_pack

## 产品支持情况

| 产品     | 是否支持 |
| ----------- | :----: |
| Ascend 950PR/Ascend 950DT | √    |

## 功能说明

将源操作数中的元素选取低8位（b16）、低16位（b32）、低32位（b64）写入目的操作数的低半部分或高半部分。

- asc_pack：将源操作数写入目的操作数的低半部分。

- asc_pack_v2：将源操作数写入目的操作数的高半部分。

## 函数原型

```cpp
__simd_callee__ inline void asc_pack(vector_uint8_t& dst, vector_uint16_t src)
__simd_callee__ inline void asc_pack(vector_uint8_t& dst, vector_int16_t src)
__simd_callee__ inline void asc_pack(vector_uint16_t& dst, vector_uint32_t src)
__simd_callee__ inline void asc_pack(vector_uint16_t& dst, vector_int32_t src)
__simd_callee__ inline void asc_pack(vector_bool& dst, vector_bool src)
__simd_callee__ inline void asc_pack_v2(vector_uint8_t& dst, vector_uint16_t src)
__simd_callee__ inline void asc_pack_v2(vector_uint8_t& dst, vector_int16_t src)
__simd_callee__ inline void asc_pack_v2(vector_uint16_t& dst, vector_uint32_t src)
__simd_callee__ inline void asc_pack_v2(vector_uint16_t& dst, vector_int32_t src)
__simd_callee__ inline void asc_pack_v2(vector_bool& dst, vector_bool src)
```

## 参数说明

| 参数名       | 输入/输出 | 描述               |
| --------- | ----- | ---------------- |
| dst       | 输出    | 目的操作数（矢量数据寄存器/掩码寄存器）。            |
| src | 输入    | 源操作数（矢量数据寄存器/掩码寄存器）。            |

矢量数据寄存器和掩码寄存器的详细说明请参见reg数据类型定义.md。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_uint8_t dst;
vector_uint16_t src;
asc_pack(dst, src);    // 将src的低8位写入dst的低半部分
asc_pack_v2(dst, src);    // 将src的低8位写入dst的高半部分
```