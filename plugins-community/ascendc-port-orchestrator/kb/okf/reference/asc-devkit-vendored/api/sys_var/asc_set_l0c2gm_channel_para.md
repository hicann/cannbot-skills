---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_set_l0c2gm_channel_para"
description: "设置通道步长专用寄存器，CHANNEL_PARA[63:48] 表示 loop0 源步长（单位 C0_SIZE，不能为 0），仅在 asc_copy_l0c2l1/l0c2gm 开启 NZ2DN 模式时有效，须在搬运接口前调用。"
tags: [sys_var]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/sys_var/asc_set_l0c2gm_channel_para.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_set_l0c2gm_channel_para.md

# asc_set_l0c2gm_channel_para

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |

## 功能说明

对通道步长参数的专用寄存器的比特位进行设置。需要配合接口[asc_copy_l0c2l1](../cube_datamove/asc_copy_l0c2l1_arch_3510.md)或[asc_copy_l0c2gm](../cube_datamove/asc_copy_l0c2gm_arch_3510.md)使用。
仅当调用接口时开启NZ2DN模式时，此功能才有效。

## 函数原型

```cpp
__aicore__ inline void asc_set_l0c2gm_channel_para(uint64_t config)
```

## 参数说明

表1 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| config | 输入 | 设置的寄存器值。常用通道步长寄存器比特位说明参考表2。 |

表2 常用通道步长寄存器比特位说明
|CHANNEL_PARA比特位    |功能|
| :-------     | :---- |
| CHANNEL_PARA[48-0]    |  保留位，设置无效  |
| CHANNEL_PARA[63-48]    |  表示loop0源步长，单位为C0_SIZE。当开启NZ2DN模式时有效，且不能为0。  |


## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

需要在对应的数据搬运接口之前调用。

## 调用示例

```cpp
uint64_t config = 0;
asc_set_l0c2gm_channel_para(config);
```
