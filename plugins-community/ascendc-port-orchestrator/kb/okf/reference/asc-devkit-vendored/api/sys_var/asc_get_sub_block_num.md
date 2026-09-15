---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_get_sub_block_num"
description: "分离模式下获取一个 AI Core 上 Cube Core（AIC）或 Vector Core（AIV）的数量，返回值随 Kernel 类型（AIV_ONLY/AIC_ONLY/MIX_AIC_1_2 等）而不同，附返回值对照表。"
tags: [sys_var]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/sys_var/asc_get_sub_block_num.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_get_sub_block_num.md

# asc_get_sub_block_num

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |

## 功能说明

分离模式下，获取一个AI Core上Cube Core（AIC）或者Vector Core（AIV）的数量。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_sub_block_num()
```

## 参数说明

无

## 返回值说明


不同Kernel类型下，在AIC和AIV上调用该接口的返回值如下：

表1 返回值说明

|Kernel类型|KERNEL_TYPE_AIV_ONLY|KERNEL_TYPE_AIC_ONLY|KERNEL_TYPE_MIX__AIC_1_2|KERNEL_TYPE_MIX_AIC_1_1|KERNEL_TYPE_MIX_AIC_1_0|KERNEL_TYPE_MIX_AIV_1_0|
| :------ | :------------------ | :----------------- | :-------------------- | :--------------------- | :-------------------- | :-------------------- |
|AIV      |1                    |-                   |2                      |1                       |-                      |1                      |
|AIC      |-                    |1                   |1                      |1                       |1                      |-                      |

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

无