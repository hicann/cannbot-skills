---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_set_ndim_pad_count"
description: "为 asc_ndim_copy_gm2ub 设置各维度左右侧 padding 元素个数，参数为 asc_ndim_pad_count_config 结构体，PIPE_S，须配合 asc_ndim_copy_gm2ub 使用。Ascend 950PR/950DT 支持。"
tags: [vector_datamove]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/vector_datamove/asc_set_ndim_pad_count.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_datamove/asc_set_ndim_pad_count.md

# asc_set_ndim_pad_count

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

用于设置[asc_ndim_copy_gm2ub](asc_ndim_copy_gm2ub.md)接口的各个维度左右侧的padding元素个数。

## 函数原型

```c++
__aicore__ inline void asc_set_ndim_pad_count(asc_ndim_pad_count_config& config)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| config | 输入 | 用于设置asc_ndim_copy_gm2ub接口的各个维度左右侧的padding元素个数，详细说明请参考[asc_ndim_pad_count_config.md](../struct/asc_ndim_pad_count_config.md)。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

需配合[asc_ndim_copy_gm2ub](asc_ndim_copy_gm2ub.md)使用。

## 调用示例

```c++
asc_ndim_pad_count_config config;
asc_set_ndim_pad_count(config);
```