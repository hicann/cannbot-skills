---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_get_icache_preload_status"
description: "查询 ICache 预加载状态的 C 接口，返回 int64_t，0 表示空闲、1 表示忙，PIPE_S 流水且无额外约束；支持 Ascend 950PR/950DT 与 Atlas A3/A2 系列产品。"
tags: [cache_ctrl]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/cache_ctrl/asc_get_icache_preload_status.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/cache_ctrl/asc_get_icache_preload_status.md

# asc_get_icache_preload_status

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

获取ICache的Preload的状态。

## 函数原型

```c++
__aicore__ inline int64_t asc_get_icache_preload_status()
```

## 参数说明

无

## 返回值说明

int64_t类型，0表示空闲，1表示忙。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```c++
int64_t status = asc_get_icache_preload_status();
```