---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_set_mmad_direction_m"
description: "设置 Mmad 结果产生方向为先 N 后 M，即优先沿矩阵列方向再沿行方向出结果，无参数无约束；与 asc_set_mmad_direction_n 成对使用，支持 Atlas A3 与 A2 系列。"
tags: [cube_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/cube_compute/asc_set_mmad_direction_m.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_compute/asc_set_mmad_direction_m.md

# asc_set_mmad_direction_m

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

设置Mmad计算时优先通过M/N中的N方向，然后通过M方向产生结果，M为矩阵的行，N为矩阵的列。

## 函数原型

```c++
__aicore__ inline void asc_set_mmad_direction_m()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```c++
asc_set_mmad_direction_m();
```
