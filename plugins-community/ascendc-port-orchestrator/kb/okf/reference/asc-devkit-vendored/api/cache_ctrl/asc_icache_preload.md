---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_icache_preload"
description: "从指令所在 DDR 地址按 prefetch_len 预加载到对应 cache line 的 ICache 预取接口，PIPE_S 流水，可配合 asc_get_program_counter 取 PC 使用；支持 950PR/950DT 与 A3/A2。"
tags: [cache_ctrl]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/cache_ctrl/asc_icache_preload.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/cache_ctrl/asc_icache_preload.md

# asc_icache_preload

## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
| <cann-filter npu_type="950"><term>Ascend 950PR/Ascend 950DT</term>  | √ </cann-filter>|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |


## 功能说明

从指令所在DDR地址预加载数据到对应的cacheline中。

## 函数原型

```cpp
__aicore__ inline void asc_icache_preload(const void* addr, int64_t prefetch_len)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| addr | 输入| 预加载数据的地址。 |
| prefetch_len | 输入 | 预加载数据的长度。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
int64_t prefetch_length = 32;
int64_t pc = asc_get_program_counter() & 0xFFFFFFFFFFFF;
asc_icache_preload(reinterpret_cast<void *>(pc), prefetch_length);
```
