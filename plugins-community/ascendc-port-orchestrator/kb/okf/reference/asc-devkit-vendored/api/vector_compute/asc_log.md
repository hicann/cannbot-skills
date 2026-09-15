---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_log"
description: "按元素取自然对数 dst=ln(src)，支持 half/float，UB 指针形态，含 count 形与 repeat 高维切分形，dst/src 需 32 字节对齐；标注支持 Atlas A3/A2。"
tags: [vector_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/vector_compute/asc_log.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_log.md
# asc_log

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

按元素取自然对数，计算公式如下：
$$
dst_i = ln({src_i})
$$

## 函数原型

- 前n个数据计算
  ```cpp
  __aicore__ inline void asc_log(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count)
  __aicore__ inline void asc_log(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
  ```

- 高维切分计算
  ```cpp
  __aicore__ inline void asc_log(__ubuf__ half* dst, __ubuf__ half* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint8_t dst_repeat_stride, uint8_t src_repeat_stride)
  __aicore__ inline void asc_log(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint8_t dst_repeat_stride, uint8_t src_repeat_stride)
  ```

- 同步计算
  ```cpp
  __aicore__ inline void asc_log_sync(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count)
  __aicore__ inline void asc_log_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
  ```

## 参数说明

|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
|dst|输出|目的操作数（矢量）的起始地址。 |
|src|输入|源操作数（矢量）的起始地址。 |
|count|输入|参与计算的元素个数。|
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
__ubuf__ half dst[256];
__ubuf__ half src[256];
asc_log(dst, src, 256);
```