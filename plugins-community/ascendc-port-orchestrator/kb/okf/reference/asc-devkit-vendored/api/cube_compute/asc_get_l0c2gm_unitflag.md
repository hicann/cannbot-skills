---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_get_l0c2gm_unitflag"
description: "在 L0C 到 GM 搬运的随路量化流程中，读取当前 unit_flag 设置，返回 uint64_t，无入参无约束；与 asc_set_l0c2gm_config 配套，支持 Atlas A3 与 A2 系列。"
tags: [cube_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/cube_compute/asc_get_l0c2gm_unitflag.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_compute/asc_get_l0c2gm_unitflag.md

# asc_get_l0c2gm_unitflag

## 产品支持情况

| 产品     | 是否支持 |
| :----------- |:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

数据搬运过程中进行随路量化时，通过调用该接口获取unit_flag设置。

## 函数原型

```cpp
__aicore__ inline uint64_t asc_get_l0c2gm_unitflag()
```

## 参数说明

无

## 返回值说明

unit_flag设置。unit_flag是一种矩阵计算指令和矩阵搬运指令细粒度的并行，使能该功能后，硬件每计算完一个分形，计算结果就会被搬出，该功能不适用于L0C Buffer累加的场景。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
uint64_t unitflag_value = asc_get_l0c2gm_unitflag();
```
