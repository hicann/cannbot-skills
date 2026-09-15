---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_set_l0c2gm_nz2nd"
description: "配置 L0C 到 GM 搬运时的随路 NZ 转 ND 格式转换参数：nd_num 取 1 到 65535、src_nd_stride 以 32B 分形为单位取 0 到 65535、dst_nd_stride 以元素为单位；仅 Ascend 950PR/950DT 支持。"
tags: [cube_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/cube_compute/asc_set_l0c2gm_nz2nd.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_compute/asc_set_l0c2gm_nz2nd.md


# asc_set_l0c2gm_nz2nd

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
| Ascend 950PR/Ascend 950DT | √    |

### 功能说明

数据搬运过程中进行随路格式转换（NZ格式转换为ND格式）时，通过调用该接口设置格式转换的相关配置。

### 函数原型

```cpp
__aicore__ inline void asc_set_l0c2gm_nz2nd(uint64_t nd_num, uint64_t src_nd_stride, uint64_t dst_nd_stride)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| nd_num | 输入 | ND矩阵的个数，取值范围为[1, 65535]。 |
| src_nd_stride | 输入 | 以分形大小为单位的源步长，源相邻NZ矩阵的偏移。取值范围为[0, 65535]，单位为32B。 |
| dst_nd_stride | 输入 | 目的相邻ND矩阵的偏移。取值范围为[1, $2^{32}$ - 1]，单位为元素。 |

### 返回值说明

无

### 流水类型

PIPE_S

### 约束说明

无

### 调用示例

```cpp
uint64_t nd_num = 2;
uint64_t src_nd_stride = 2;
uint64_t dst_nd_stride = 1;
asc_set_l0c2gm_nz2nd(nd_num, src_nd_stride, dst_nd_stride);
```
