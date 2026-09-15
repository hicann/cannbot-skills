---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_get_l0c2gm_relu"
description: "在 L0C 到 GM 搬运的随路量化流程中，读取 ReLU 操作前矢量的起始地址，返回 uint64_t，无入参无约束；与 asc_set_l0c2gm_config 配套，支持 Atlas A3 与 A2 系列。"
tags: [cube_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/cube_compute/asc_get_l0c2gm_relu.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_compute/asc_get_l0c2gm_relu.md

# asc_get_l0c2gm_relu

## 产品支持情况

| 产品     | 是否支持 |
| :----------- |:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

数据搬运过程中进行随路量化时，通过调用该接口获取ReLU操作前矢量的起始地址。

## 函数原型

```cpp
__aicore__ inline uint64_t asc_get_l0c2gm_relu()
```

## 参数说明

无

## 返回值说明

ReLU操作前矢量的起始地址。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
uint64_t relu_addr = asc_get_l0c2gm_relu();
```
