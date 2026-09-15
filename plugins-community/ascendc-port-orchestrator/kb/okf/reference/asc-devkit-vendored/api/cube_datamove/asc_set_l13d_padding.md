---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_set_l13d_padding"
description: "设置 asc_copy_l12l0a 3D 格式搬运使用的 Pad 属性描述即填充数值，config 有 uint64_t、half、int16_t、uint16_t 等重载；950PR/950DT 与 Atlas A3/A2 支持。"
tags: [cube_datamove]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/cube_datamove/asc_set_l13d_padding.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_set_l13d_padding.md

# asc_set_l13d_padding

## AI处理器支持情况

|AI处理器类型   | 是否支持 |
| ------------|:----:|
| Ascend 950PR/Ascend 950DT |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

设置Pad属性描述，用于在调用[asc_copy_l12l0a](asc_copy_l12l0a_arch_2201.md)接口时配置填充数值。

## 函数原型

```c++
__aicore__ inline void asc_set_l13d_padding(uint64_t config)
__aicore__ inline void asc_set_l13d_padding(half config)
__aicore__ inline void asc_set_l13d_padding(int16_t config)
__aicore__ inline void asc_set_l13d_padding(uint16_t config)
```

## 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| config | 输入     | Pad填充值的数值。|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```c++
constexpr uint64_t config = 0;
asc_set_l13d_padding(config);
```