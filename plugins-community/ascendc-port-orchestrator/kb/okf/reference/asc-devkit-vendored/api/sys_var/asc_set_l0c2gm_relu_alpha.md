---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_set_l0c2gm_relu_alpha"
description: "设置 64bit 的 RELU_ALPHA 寄存器，存放 fixpipe 或 cube 指令做 Scalar ReLU 的 alpha 值：31:13 为 ReLU_PRE 的 M2，63:45 为 ReLU_POST 的 M2。"
tags: [sys_var]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/sys_var/asc_set_l0c2gm_relu_alpha.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_set_l0c2gm_relu_alpha.md

# asc_set_l0c2gm_relu_alpha

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| Ascend 950PR/Ascend 950DT |    √     |

## 功能说明

对RELU_ALPHA寄存器中的值进行设置。这是一个64bit的寄存器，存储在fixpipe或cube指令中进行Scalar ReLU时使用的alpha值。

其中各bit含义如下：

| bit范围 |                                           含义                                           |
|:------|:--------------------------------------------------------------------------------------:|
| 31:13 | 表示ReLU_PRE中Scalar ReLU的M2值，只有Scalar ReLU时生效。硬件将以1位符号位，8位指数位和10位尾数位的格式用于计算，不能是INF/NAN。  |
| 63:45 | 表示ReLU_POST中Scalar ReLU的M2值，只有Scalar ReLU时生效。硬件将以1位符号位，8位指数位和10位尾数位的格式用于计算，不能是INF/NAN。 |

## 函数原型

```cpp
__aicore__ inline void asc_set_l0c2gm_relu_alpha(uint64_t config)
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
uint32_t pre_m2 = (0<<18)|(127<<10)|0;
uint32_t post_m2 = (0<<18)|(124<<10)|0;
uint64_t config = ((uint64_t)pre_m2<<13)|((uint64_t)post_m2<<45);
asc_set_l0c2gm_relu_alpha(config);
```