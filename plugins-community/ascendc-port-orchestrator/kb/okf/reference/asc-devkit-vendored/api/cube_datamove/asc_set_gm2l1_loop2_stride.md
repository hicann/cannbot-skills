---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_set_gm2l1_loop2_stride"
description: "GM→L1 Buffer 搬运的配置类接口：设置外层循环中相邻迭代数据块之间的间隔，参数 loop2_src_stride、loop2_dst_stride 分别作用于源与目的操作数；仅 Ascend 950PR/950DT 支持。"
tags: [cube_datamove]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/cube_datamove/asc_set_gm2l1_loop2_stride.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_set_gm2l1_loop2_stride.md

# asc_set_gm2l1_loop2_stride

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
| Ascend 950PR/Ascend 950DT | √    |

## 功能说明

将数据从Global Memory (GM) 搬运到L1 Buffer时，通过调用该接口设置外层循环中相邻迭代数据块间的间隔。

以源操作数搬运场景为例，如下图所示。

## 函数原型

```cpp
__aicore__ inline void asc_set_gm2l1_loop2_stride(uint64_t loop2_src_stride, uint64_t loop2_dst_stride)
```

## 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| loop2_src_stride     | 输入     | 外层循环中相邻迭代源操作数的数据块间的间隔，单位为Byte，取值范围为[0,2^40]。|
| loop2_dst_stride     | 输入     | 外层循环中相邻迭代目标操作数的数据块间的间隔，单位为Byte，取值范围为[0,2^21]，且必须32B对齐。|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
asc_set_gm2l1_loop_size(2, 2);
asc_set_gm2l1_loop1_stride(96, 128);
asc_set_gm2l1_loop2_stride(192, 288);
asc_copy_gm2l1_align(dst, src, size);
asc_set_gm2l1_loop_size(1, 1);
```
