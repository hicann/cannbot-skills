---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_create_iter_reg"
description: "初始化地址寄存器 iter_reg，按位宽分 _b8/_b16/_b32 三个版本，入参为 uint32_t offset；生成的 iter_reg 用于循环中存储地址偏移量；仅 950PR/950DT 支持。"
tags: [reg_vector]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/reg/reg_vector/asc_create_iter_reg.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_create_iter_reg.md

# asc_create_iter_reg

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <cann-filter npu_type="950">Ascend 950PR/Ascend 950DT  | √ </cann-filter>|

## 功能说明

地址寄存器通过该接口初始化，然后在循环之中使用地址寄存器存储地址偏移量。

## 函数原型

  ```cpp
  __simd_callee__ inline iter_reg asc_create_iter_reg_b32(uint32_t offset)
  __simd_callee__ inline iter_reg asc_create_iter_reg_b16(uint32_t offset)
  __simd_callee__ inline iter_reg asc_create_iter_reg_b8(uint32_t offset)
  ```

## 参数说明

| 参数名       | 输入/输出 | 描述                |
| --------- | ----- | ----------------- |
| offset       | 输入 | 地址偏移量。 |

## 返回值说明

地址寄存器。

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

见地址寄存器调用示例。
