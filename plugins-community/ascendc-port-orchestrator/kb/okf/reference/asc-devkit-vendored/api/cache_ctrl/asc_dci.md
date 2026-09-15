---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_dci"
description: "数据缓存整体失效指令，把所有 cache line 直接标记为无效且不回写脏数据，仅 Ascend 950PR/950DT 支持；调用前须先用 asc_sync_data_barrier 插入 DSB_ALL 等待访存结束。"
tags: [cache_ctrl]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/cache_ctrl/asc_dci.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/cache_ctrl/asc_dci.md

# asc_dci

## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

数据缓存失效，该指令用于使整个数据缓存无效化，且不会对处于“脏”状态的缓存行执行写回操作。换言之，所有缓存行将被直接标记为无效，其修改过但尚未同步至主存的数据将被丢弃。

## 函数原型

```cpp
__aicore__ inline void asc_dci()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

在调用asc_dci之前，需调用asc_sync_data_barrier插入DSB_ALL指令, 等待所有内存访问指令执行结束。

## 调用示例

```cpp
asc_sync_data_barrier(mem_dsb_t::DSB_ALL);
asc_dci();
```
