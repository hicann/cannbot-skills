---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_datacache_preload"
description: "从指定 GM 地址按 offset 预加载数据到 Data Cache 的 C 接口，PIPE_S 流水，支持 Ascend 950PR/950DT 与 Atlas A3/A2；频繁调用会导致保留栈拥塞、指令被视为 NOP 并阻塞 Scalar 流水。"
tags: [cache_ctrl]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/cache_ctrl/asc_datacache_preload.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/cache_ctrl/asc_datacache_preload.md

# asc_datacache_preload

## 产品支持情况

|产品   | 是否支持 |
| :------------|:----:|
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

从源地址所在的特定GM地址预加载数据到Data Cache中。

## 函数原型

```cpp
__aicore__ inline void asc_datacache_preload(__gm__ uint64_t* address, int64_t offset)
```

## 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| address     | 输入     | 源操作数的起始地址。   |
| offset     | 输入     | 在源操作数上偏移offset大小开始加载数据，单位为Bytes。|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

频繁调用此接口可能会导致保留栈拥塞，此时，该接口将被是为NOP指令，阻塞Scalar流水。因此不建议频繁调用此接口。

## 调用示例

```cpp
__gm__ uint64_t* addr;
int64_t offset = 0;
asc_datacache_preload(addr, offset);
```
