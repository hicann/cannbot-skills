---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_set_store_atomic_config_v1"
description: "A2/A3 形态：以 type(0 无/1 float/2 half/3 int16/4 int32/5 int8/6 bf16) 和 op(0 求和) 直接设置搬出原子操作使能位与类型，PIPE_S。"
tags: [simd_atomic]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/simd_atomic/asc_set_store_atomic_config_v1.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/simd_atomic/asc_set_store_atomic_config_v1.md

# asc_set_store_atomic_config_v1

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

设置原子操作使能位与原子操作类型的值。

## 函数原型

```cpp
__aicore__ inline void asc_set_store_atomic_config_v1(uint16_t type, uint16_t op)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| type | 输入 | 原子操作使能位。<br>0：无原子操作。<br>1：使能原子操作，进行原子操作的数据类型为float。<br>2：使能原子操作，进行原子操作的数据类型为half。<br>3：使能原子操作，进行原子操作的数据类型为int16_t。<br>4：使能原子操作，进行原子操作的数据类型为int32_t。<br>5：使能原子操作，进行原子操作的数据类型为int8_t。<br>6：使能原子操作，进行原子操作的数据类型为bfloat16_t。<br>其余值无具体含义。 |
| op | 输入 | 原子操作类型。<br>0：求和操作。<br>其余值无具体含义。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
uint16_t type = 1;
uint16_t op = 0;
asc_set_store_atomic_config_v1(type, op);
```