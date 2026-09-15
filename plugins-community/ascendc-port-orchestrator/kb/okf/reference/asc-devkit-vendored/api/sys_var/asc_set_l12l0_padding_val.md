---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_set_l12l0_padding_val"
description: "设置 64bit 的 PADDING_B 寄存器，位 31:0 存放搬运 padding 值（按 32/16/8/4 位宽有不同填法），位 33:32 选择固定 padding 或按通道 padding 模式，仅 950PR/950DT。"
tags: [sys_var]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/sys_var/asc_set_l12l0_padding_val.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_set_l12l0_padding_val.md

# asc_set_l12l0_padding_val

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| Ascend 950PR/Ascend 950DT |    √     |

## 功能说明

对PADDING_B寄存器中的值进行设置，这是一个64bit的寄存器，用于存储搬运过程中padding的值。
其中各bit含义如下：

| bit范围 |                                                                           含义                                                                           |
| :-----------|:------------------------------------------------------------------------------------------------------------------------------------------------------:|
| 31:0 | 存储进行padding时的值。<br>数据位宽为32时，直接使用31:0位；<br> 数据位宽为16时，使用15:0位，31:16位被忽略； <br> 数据位宽为8时，15:8位应与7:0位相同，31:16位被忽略；数据位宽为4时，15:12，11:8，7:4和3:0位应该相同，31:16位被忽略。 |
| 33:32 | 表示所使用的padding模式，当前支持三种模式。<br>2'b00：固定padding；<br>2'b01：按通道padding；<br>2'b010/2'b11：当前保留。 |

## 函数原型

```cpp
__aicore__ inline void asc_set_l12l0_padding_val(uint64_t config)
```

## 参数说明

|参数名|输入/输出| 描述        |
| :------ | :---  |:----------|
|config   |输入   | 待设置的寄存器值。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例
```cpp
uint64_t config = 0;
asc_set_l12l0_padding_val(config);
```
