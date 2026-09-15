---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_prelu"
description: "reg 矢量 PReLU：dst=(src0>0)?src0:src0*src1，负半轴按 src1 逐元素缩放，支持 half/float 寄存器形态并按 mask 筛选；PIPE_V，支持 950PR/950DT。"
tags: [reg_vector]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/reg/reg_vector/asc_prelu.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_prelu.md

# asc_prelu

## 产品支持情况

| 产品                  | 是否支持  |
|:-------------------------| :------: |
| Ascend 950PR/Ascend 950DT | √    |

## 功能说明

源操作数src0大于0的情况下直接将src0写入目的操作数dst，否则将src0 * src1的结果写入dst。计算公式如下：

$$
dst = (src0 > 0) ? src0 : src0 * src1
$$

## 函数原型

```cpp
__simd_callee__ inline void asc_prelu(vector_float& dst, vector_float src0, vector_float src1, vector_bool mask)
__simd_callee__ inline void asc_prelu(vector_half& dst, vector_half src0, vector_half src1, vector_bool mask)
```

## 参数说明

| 参数名  | 输入/输出 | 描述                                                                   |
|:-----| :--- |:---------------------------------------------------------------------|
| dst | 输出 | 目的操作数（矢量数据寄存器）。 |
| src0 | 输入 | 源操作数（矢量数据寄存器）。 |
| src1 | 输入 | 源操作数（矢量数据寄存器）。 |
| mask | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

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
vector_half src0;
vector_half src1;
vector_bool mask = asc_create_mask_b16(PAT_ALL);
asc_loadalign(src0, src0_addr); // src0_addr是外部输入的UB内存空间地址。
asc_loadalign(src1, src1_addr); // src1_addr是外部输入的UB内存空间地址。
asc_prelu(dst, src0, src1, mask);
```