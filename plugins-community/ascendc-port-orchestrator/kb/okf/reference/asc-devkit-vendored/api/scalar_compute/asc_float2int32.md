---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_float2int32"
description: "标量 float→int32 转换，四种舍入模式各一个接口：_rn(四舍六入五成双)、_rna(四舍五入)、_rd(向负无穷)、_ru(向正无穷)；支持 950PR/950DT。"
tags: [scalar_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/scalar_compute/asc_float2int32.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/scalar_compute/asc_float2int32.md

# asc_float2int32
## 产品支持情况

| 产品     | 是否支持 |
| ----------- | :----: |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

将float类型转化为int32_t类型，并支持多种舍入模式。

舍入模式：
- RINT舍入模式：四舍六入五成双舍入
- ROUND舍入模式：四舍五入舍入
- FLOOR舍入模式：向负无穷舍入
- CEIL舍入模式：向正无穷舍入

## 函数原型

```cpp
// RINT舍入模式
__aicore__ inline int32_t asc_float2int32_rn(float value)
// ROUND舍入模式
__aicore__ inline int32_t asc_float2int32_rna(float value)
// FLOOR舍入模式
__aicore__ inline int32_t asc_float2int32_rd(float value)
// CEIL舍入模式
__aicore__ inline int32_t asc_float2int32_ru(float value)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :------ | :----- | :----------------- |
| value | 输入 | 源操作数（标量）。 |

## 返回值说明

目的操作数（标量），value精度转换成int32_t的结果。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
float value = 3.0;
int32_t dst = asc_float2int32_rn(value);
```
