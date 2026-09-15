---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_axpy"
description: "标量乘加并累加到目的操作数 dst=src*value+dst，支持 half→half、half→float、float→float 三种组合；含 count 形、repeat 形与 _sync 形，地址需 32 字节对齐。"
tags: [vector_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/vector_compute/asc_axpy.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_axpy.md
# asc_axpy

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

源操作数src中每个元素与标量value求积后和目的操作数dst中的对应元素相加计算公式如下:

$$
dst = src * value + dst
$$

## 函数原型

- 连续数据计算

  ```cpp
  __aicore__ inline void asc_axpy(__ubuf__ half* dst, __ubuf__ half* src, half value, uint32_t count)
  __aicore__ inline void asc_axpy(__ubuf__ float* dst, __ubuf__ half* src, half value, uint32_t count)
  __aicore__ inline void asc_axpy(__ubuf__ float* dst, __ubuf__ float* src, float value, uint32_t count)
  ```

- 高维切分计算

  ```cpp
  __aicore__ inline void asc_axpy(__ubuf__ half* dst, __ubuf__ half* src, half value, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  __aicore__ inline void asc_axpy(__ubuf__ float* dst, __ubuf__ half* src, half value, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  __aicore__ inline void asc_axpy(__ubuf__ float* dst, __ubuf__ float* src, float value, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  ```

- 同步计算

  ```cpp
  __aicore__ inline void asc_axpy_sync(__ubuf__ half* dst, __ubuf__ half* src, half value, uint32_t count)
  __aicore__ inline void asc_axpy_sync(__ubuf__ float* dst, __ubuf__ half* src, half value, uint32_t count)
  __aicore__ inline void asc_axpy_sync(__ubuf__ float* dst, __ubuf__ float* src, float value, uint32_t count)
  ```

## 参数说明

表1 参数说明

| 参数名       | 输入/输出 | 描述               |
| --------- | ----- | ---------------- |
| dst       | 输出    | 目的操作数（矢量）的起始地址。            |
| src | 输入    | 源操作数（矢量）的起始地址。            |
| value | 输入    | 源操作数（标量）。       |
| count     | 输入    | 参与连续计算的元素个数。      |
| repeat |输入| 迭代次数。 |
| dst_block_stride |输入| 目的操作数单次迭代内不同DataBlock间地址步长。 |
| src_block_stride |输入| 源操作数单次迭代内不同DataBlock间地址步长。 |
| dst_repeat_stride |输入| 目的操作数相邻迭代间相同DataBlock的地址步长。 |
| src_repeat_stride |输入| 源操作数相邻迭代间相同DataBlock的地址步长。 |


## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址重叠约束请参考通用地址重叠约束。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```cpp
constexpr uint64_t total_length = 128;    // total_length指参与计算的数据长度
half value = 10;
__ubuf__ half src[total_length];
__ubuf__ half dst[total_length];
asc_axpy(dst, src, value, total_length);
```
