---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_set_l0c2gm_config"
description: "设置 L0C 到 GM 随路量化的矢量量化参数：relu_pre、quant_pre 两个前置矢量起始地址与 enable_unit_flag；unit_flag 让每算完一个分形即搬出，不适用于 L0C 累加场景，两个地址可只传其一、另一个填 0。"
tags: [cube_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/cube_compute/asc_set_l0c2gm_config.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_compute/asc_set_l0c2gm_config.md

# asc_set_l0c2gm_config
## AI处理器支持情况

|AI处理器类型   | 是否支持 |
| ------------|:----:|
| <cann-filter npu_type="950"><term>Ascend 950PR/Ascend 950DT</term>  | √ </cann-filter>|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

数据搬运过程中进行随路量化时，通过调用该接口设置量化流程中的矢量量化参数。

## 函数原型

```c++
__aicore__ inline void asc_set_l0c2gm_config(uint64_t relu_pre, uint64_t quant_pre, bool enable_unit_flag)
```

## 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| relu_pre | 输入     | ReLU操作前矢量的起始地址。|
| quant_pre | 输入     | 量化操作前矢量的起始地址。|
| enable_unit_flag | 输入     | 是否启用unit_flag。unit_flag是一种矩阵计算指令和矩阵搬运指令细粒度的并行，使能该功能后，硬件每计算完一个分形，计算结果就会被搬出，该功能不适用于L0C Buffer累加的场景。|
## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

支持以下三种传参形式：
- 同时设置relu_pre和quant_pre。
- 仅传入relu_pre，quant_pre传入0。
- 仅传入quant_pre，relu_pre传入0。

## 调用示例

```c++
constexpr uint64_t relu_pre = 0;
constexpr uint64_t quant_pre = 0x1000;// 假设量化操作有效地址为 0x1000
asc_set_l0c2gm_config(relu_pre, quant_pre, true);
```