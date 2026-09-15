---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_float2int16"
description: "float 转 int16_t，舍入模式由后缀选择：_rn 四舍六入五成双、_rna 四舍五入、_rd 向负无穷、_ru 向正无穷、_rz 向零；含 count 形与 repeat 高维切分形。"
tags: [vector_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/vector_compute/asc_float2int16.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_float2int16.md
# asc_float2int16

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :-----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |   √   |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |   √   |

## 功能说明

将float类型数据转换为int16_t类型，支持多种舍入模式：

- RINT舍入模式：四舍六入五成双舍入。
- ROUND舍入模式：四舍五入舍入。
- FLOOR舍入模式：向负无穷舍入。
- CEIL舍入模式：向正无穷舍入。
- TRUNC舍入模式：向零舍入。

## 函数原型

- 前n个数据计算

  ```cpp
  // RINT舍入模式
  __aicore__ inline void asc_float2int16_rn(__ubuf__ int16_t* dst, __ubuf__ float* src, uint32_t count)
  // ROUND舍入模式
  __aicore__ inline void asc_float2int16_rna(__ubuf__ int16_t* dst, __ubuf__ float* src, uint32_t count)
  // FLOOR舍入模式
  __aicore__ inline void asc_float2int16_rd(__ubuf__ int16_t* dst, __ubuf__ float* src, uint32_t count)
  // CEIL舍入模式
  __aicore__ inline void asc_float2int16_ru(__ubuf__ int16_t* dst, __ubuf__ float* src, uint32_t count)
  // TRUNC舍入模式
  __aicore__ inline void asc_float2int16_rz(__ubuf__ int16_t* dst, __ubuf__ float* src, uint32_t count)
  ```

- 高维切分计算

  ```cpp
  // RINT舍入模式
  __aicore__ inline void asc_float2int16_rn(__ubuf__ int16_t* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  // ROUND舍入模式
  __aicore__ inline void asc_float2int16_rna(__ubuf__ int16_t* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  // FLOOR舍入模式
  __aicore__ inline void asc_float2int16_rd(__ubuf__ int16_t* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  // CEIL舍入模式
  __aicore__ inline void asc_float2int16_ru(__ubuf__ int16_t* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  // TRUNC舍入模式
  __aicore__ inline void asc_float2int16_rz(__ubuf__ int16_t* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  ```

- 同步计算

    ```cpp
  // RINT舍入模式
  __aicore__ inline void asc_float2int16_rn_sync(__ubuf__ int16_t* dst, __ubuf__ float* src, uint32_t count)
  // ROUND舍入模式
  __aicore__ inline void asc_float2int16_rna_sync(__ubuf__ int16_t* dst, __ubuf__ float* src, uint32_t count)
  // FLOOR舍入模式
  __aicore__ inline void asc_float2int16_rd_sync(__ubuf__ int16_t* dst, __ubuf__ float* src, uint32_t count)
  // CEIL舍入模式
  __aicore__ inline void asc_float2int16_ru_sync(__ubuf__ int16_t* dst, __ubuf__ float* src, uint32_t count)
  // TRUNC舍入模式
  __aicore__ inline void asc_float2int16_rz_sync(__ubuf__ int16_t* dst, __ubuf__ float* src, uint32_t count)
  ```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :----| :-----| :-----|
| dst | 输出 | 目的操作数（向量）的起始地址。 |
| src  | 输入 | 源操作数（向量）的起始地址|
| count | 输入 | 参与计算的元素个数。 |
| dst_block_stride |  输入 |目的操作数单次迭代内不同DataBlock间地址步长。 |
| src_block_stride |  输入 |源操作数单次迭代内不同DataBlock间地址步长。 |
| dst_repeat_stride | 输入 | 目的操作数相邻迭代间相同DataBlock的地址步长。 |
| src_repeat_stride | 输入 | 源操作数相邻迭代间相同DataBlock的地址步长。 |
| repeat | 输入 | 迭代次数。|

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- dst、src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考通用地址重叠约束。

## 调用示例

```cpp
// total_length指参与计算的数据长度
constexpr uint64_t total_length = 64
__ubuf__ float src[total_length];
__ubuf__ int16_t dst[total_length];
asc_float2int16_rn(dst, src, total_length);
```
