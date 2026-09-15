---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_set_l13d_rpt"
description: "设置 Load3Dv2 接口的 repeat 参数，入参为 asc_load3d_v2_config 结构体，需配合该结构体一起使用；950PR/950DT 与 Atlas A3/A2 支持。"
tags: [cube_datamove]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/cube_datamove/asc_set_l13d_rpt.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_set_l13d_rpt.md

# asc_set_l13d_rpt

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

用于设置Load3Dv2接口的repeat参数。

## 函数原型

```c++
__aicore__ inline void asc_set_l13d_rpt(asc_load3d_v2_config& config)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| config | 输入 | 用于设置Load3Dv2接口的repeat参数，详细说明请参考[asc_load3d_v2_config.md](../struct/asc_load3d_v2_config.md)。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

- 需配合[asc_load3d_v2_config.md](../struct/asc_load3d_v2_config.md)使用，用于设置Load3Dv2接口的repeat参数。

## 调用示例

```c++
asc_load3d_v2_config config;
asc_set_l13d_rpt(config); // 设置Load3D的repeat相关参数为默认值rpt_stride：0，rpt_time：1，rpt_mode：0
```