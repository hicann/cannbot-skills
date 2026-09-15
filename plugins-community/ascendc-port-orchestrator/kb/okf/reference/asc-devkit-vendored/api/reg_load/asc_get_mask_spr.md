---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_get_mask_spr"
description: "从特殊寄存器 SPR{MASK1, MASK0}（各 64bit）读 mask 并展开为 vector_bool：b16 版读满 128bit 每 bit 复制 2 份，b32 版只读 MASK0 每 bit 复制 4 份；仅 950PR/950DT 支持。"
tags: [reg_load]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/reg/reg_load/asc_get_mask_spr.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_load/asc_get_mask_spr.md

# asc_get_mask_spr

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| Ascend 950PR/Ascend 950DT | √    |

## 功能说明

从特殊寄存器SPR{MASK1, MASK0}读取mask值并根据数据类型格式返回对应的mask数据，MASK0、MASK1均为64bit的寄存器。

- 对于b16类型，会读取完整的128bit{MASK1, MASK0}数据，并将每bit复制为2bit并返回。

- 对于b32类型，会读取64bit MASK0数据，并将每bit复制为4bit并返回。

## 函数原型

```cpp
__simd_callee__ inline vector_bool asc_get_mask_spr_b16()
__simd_callee__ inline vector_bool asc_get_mask_spr_b32()
```

## 参数说明

无

## 返回值说明

特殊寄存器中读取的mask值。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
vector_bool mask_b16 = asc_get_mask_spr_b16();
vector_bool mask_b32 = asc_get_mask_spr_b32();
```