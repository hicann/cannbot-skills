---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_enable_hf32_trans"
description: "设置 HF32 模式的取整方式，须先调用 asc_enable_hf32 开启；mode 仅取 0（FP32 以最接近偶数方式四舍五入为 HF32）或 1（向零舍入），支持 Atlas A3 与 A2 系列。"
tags: [cube_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/cube_compute/asc_enable_hf32_trans.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_compute/asc_enable_hf32_trans.md

# asc_enable_hf32_trans

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

设置HF32模式取整方式，需要先使用[asc_enable_hf32](asc_enable_hf32.md)开启HF32取整模式。


## 函数原型

 ```cpp
   __aicore__ inline void asc_enable_hf32_trans(uint32_t mode);
```


## 参数说明

| 参数名 | 输入/输出 | 描述 |
|:-------|:----------|:------|
| mode | 输入 | HF32取整模式控制入参，uint32_t类型，支持如下2种取值：<br>0：FP32将以最接近偶数的方式四舍五入为HF32。<br>1：FP32将以向零靠近的方式四舍五入为HF32。 |


## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

mode仅支持如下2种取值：<br>0：FP32将以最接近偶数的方式四舍五入为HF32。<br>1：FP32将以向零靠近的方式四舍五入为HF32。

## 调用示例

```cpp
uint32_t mode = 0;
asc_enable_hf32_trans(mode);
```