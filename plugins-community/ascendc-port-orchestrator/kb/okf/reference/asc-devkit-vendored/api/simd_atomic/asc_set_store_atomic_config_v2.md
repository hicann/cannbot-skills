---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_set_store_atomic_config_v2"
description: "950PR/950DT 形态：以 type(1 int32/4 half/5 float/7 bf16/9 int16/10 int8) 和 op(2 求和) 设置搬出原子操作使能位与类型，编码与 v1 不同，PIPE_S。"
tags: [simd_atomic]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/simd_atomic/asc_set_store_atomic_config_v2.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/simd_atomic/asc_set_store_atomic_config_v2.md

# asc_set_store_atomic_config_v2

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

设置原子操作使能位与原子操作类型的值。

## 函数原型

```cpp
__aicore__ inline void asc_set_store_atomic_config_v2(uint16_t type, uint16_t op)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| type | 输入 | 原子操作使能位。<br>1：使能原子操作，进行原子操作的数据类型为int32_t。<br>4：使能原子操作，进行原子操作的数据类型为half。<br>5：使能原子操作，进行原子操作的数据类型为float。<br>7：使能原子操作，进行原子操作的数据类型为bfloat16_t。<br>9：使能原子操作，进行原子操作的数据类型为int16_t。<br>10：使能原子操作，进行原子操作的数据类型为int8_t。<br>其余值无具体含义。 |
| op | 输入 | 原子操作类型。<br>2：求和操作。<br>其余值无具体含义。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
uint16_t type = 1;
uint16_t op = 2;
asc_set_store_atomic_config_v2(type, op);
```