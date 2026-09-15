---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_set_gm2l1_nz_para"
description: "设置 64bit 的 MTE2_NZ_PARA 寄存器，按位配置 ND/DN 矩阵数量、N 方向 block 间隔、C0 列间隔与 NZ 矩阵间隔，供 asc_copy_gm2l1_dn2nz/nd2nz 搬运前使用。"
tags: [sys_var]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/sys_var/asc_set_gm2l1_nz_para.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_set_gm2l1_nz_para.md

# asc_set_gm2l1_nz_para

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| Ascend 950PR/Ascend 950DT |    √     |

## 功能说明

对MTE2_NZ_PARA寄存器中的值进行设置，可以在调用[asc_copy_gm2l1_dn2nz](../cube_datamove/asc_copy_gm2l1_dn2nz.md)和[asc_copy_gm2l1_nd2nz](../cube_datamove/asc_copy_gm2l1_nd2nz.md)前设置相关参数。

MTE2_NZ_PARA是一个64bit的寄存器，其中各bit含义如下：

| bit范围 |        含义         |
|:------|:-----------------:|
| 15:0  |  需要搬运的ND/DN矩阵数量。  |
| 31:16 | N维度上相邻两个block块之间的间隔。 |
| 47:32 | 同一个NZ矩阵中2个C0列之间的间隔。 |
| 63:48 | 两个ND/DN矩阵中两个NZ矩阵之间的间隔。 |

## 函数原型

```cpp
__aicore__ inline void asc_set_gm2l1_nz_para(uint64_t config)
```

## 参数说明

|参数名|输入/输出| 描述        |
| :------ | :---  |:----------|
| config | 输入 | 待设置的寄存器值。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例
```cpp
uint64_t nzDstStride = 16;
uint64_t config = nzDstStride << 48; //[63:48]
uint64_t nDstStride = 32;
config |= nDstStride << 32; //[47:32]
uint64_t C0Stride = 32;
config |= C0Stride << 16; //[31:16]
uint64_t ndNum = 64;
config |= ndNum; //[15:0]

asc_set_gm2l1_nz_para(config);
```