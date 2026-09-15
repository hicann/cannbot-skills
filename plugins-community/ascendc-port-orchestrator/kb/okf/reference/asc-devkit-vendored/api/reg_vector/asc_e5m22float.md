---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_e5m22float"
description: "reg 层类型转换 fp8_e5m2_t→float：源按 256B 分四部分，asc_e5m22float/_v2/_v3/_v4 依次读第一到第四部分写入 vector_float，未选中元素置零，PIPE_V，950PR/950DT。"
tags: [reg_vector]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/reg/reg_vector/asc_e5m22float.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_e5m22float.md

# asc_e5m22float

## 产品支持情况

| 产品     | 是否支持 |
| ----------- | :----: |
| Ascend 950PR/Ascend 950DT | √    |

## 功能说明

将vector_fp8_e5m2_t类型的源操作数以256B为单位分为四部分，读取其中一部分元素，将其转换成vector_float类型并写入目的操作数。

- asc_e5m22float：读取第一部分。

- asc_e5m22float_v2：读取第二部分。

- asc_e5m22float_v3：读取第三部分。

- asc_e5m22float_v4：读取第四部分。

## 函数原型

```cpp
__simd_callee__ inline void asc_e5m22float(vector_float& dst, vector_fp8_e5m2_t src, vector_bool mask)
__simd_callee__ inline void asc_e5m22float_v2(vector_float& dst, vector_fp8_e5m2_t src, vector_bool mask)
__simd_callee__ inline void asc_e5m22float_v3(vector_float& dst, vector_fp8_e5m2_t src, vector_bool mask)
__simd_callee__ inline void asc_e5m22float_v4(vector_float& dst, vector_fp8_e5m2_t src, vector_bool mask)
```

## 参数说明

| 参数名       | 输入/输出 | 描述               |
| --------- | ----- | ---------------- |
| dst       | 输出    | 目的操作数（矢量数据寄存器）。            |
| src | 输入    | 源操作数（矢量数据寄存器）。            |
| mask     | 输入    | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。        |

矢量数据寄存器和掩码寄存器的详细说明请参见reg数据类型定义.md。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_float dst;
vector_fp8_e5m2_t src;
vector_bool mask;
asc_e5m22float(dst, src, mask);    // 将src的第一部分转换成vector_float类型并写入dst
asc_e5m22float_v2(dst, src, mask);    // 将src的第二部分转换成vector_float类型并写入dst
asc_e5m22float_v3(dst, src, mask);    // 将src的第三部分转换成vector_float类型并写入dst
asc_e5m22float_v4(dst, src, mask);    // 将src的第四部分转换成vector_float类型并写入dst
```