---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_get_cmp_mask"
description: "把 Compare 类指令写在比较寄存器中的结果读出到 dst（__ubuf__ void*）；必须与 asc_eq/asc_ge 等 Compare 操作配合使用。"
tags: [vector_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/vector_compute/asc_get_cmp_mask.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_get_cmp_mask.md
# asc_get_cmp_mask

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

此接口用于获取Compare操作的比较结果。

## 函数原型

```cpp
__aicore__ inline void asc_get_cmp_mask(__ubuf__ void* dst)
```

## 参数说明

|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
|dst|输出|存放比较操作结果的地址。|

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

需和Compare操作配合使用。

## 调用示例

```
__ubuf__ int8_t dst[total_length];
...     // 进行Compare操作
asc_get_cmp_mask(dst);
```