---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_set_l13d_fmatrix"
description: "设置 asc_copy_l12l0a/asc_copy_l12l0b 的 3D 格式搬运所用 Feature map 属性描述（asc_l13d_fmatrix_config）；仅当 f_matrix_ctrl 指示从左矩阵取属性时用本接口，否则用 asc_set_l13d_fmatrix_b。"
tags: [cube_datamove]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/cube_datamove/asc_set_l13d_fmatrix.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_set_l13d_fmatrix.md

# asc_set_l13d_fmatrix

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

设置Feature map属性描述，用于在调用[asc_copy_l12l0a](asc_copy_l12l0a_arch_2201.md)/[asc_copy_l12l0b](asc_copy_l12l0b_arch_2201.md)的3D格式搬运接口时配置填充数值。
仅当asc_copy_l12l0a/asc_copy_l12l0b接口的f_matrix_ctrl参数指示从左矩阵获取FeatureMap的属性时使用本接口设置Feature map属性描述，否则使用asc_set_l13d_fmatrix_b接口。

## 函数原型

```c++
__aicore__ inline void asc_set_l13d_fmatrix(asc_l13d_fmatrix_config& config)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| config | 输入 | 用于设置asc_copy_l12l0a/asc_copy_l12l0b的3D格式搬运接口的Feature map属性参数，详细说明请参考[asc_l13d_fmatrix_config](../struct/asc_l13d_fmatrix_config.md)。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```c++
asc_l13d_fmatrix_config config;
asc_set_l13d_fmatrix(config);
```