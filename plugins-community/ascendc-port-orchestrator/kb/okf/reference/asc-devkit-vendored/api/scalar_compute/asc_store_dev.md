---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_store_dev"
description: "绕过 DCache 直接向 GM 地址写单个标量值，覆盖 int8/uint8/int16/uint16/int32/uint32/int64/uint64；用于多核写非 Cache Line 对齐地址时规避按行读写造成的随机覆盖。"
tags: [scalar_compute]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/scalar_compute/asc_store_dev.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/scalar_compute/asc_store_dev.md

# asc_store_dev

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

不经过DCache向GM地址上写数据。‌
当多核操作GM地址时，如果数据无法对齐到Cache Line，经过DCache的方式下，由于按照Cache Line大小进行读写，会导致多核数据随机覆盖的问题。此时，可以采用不经过DCache直接读写GM地址的方式，从而避免上述随机覆盖的问题。

## 函数原型

```cpp
__aicore__ inline void asc_store_dev(__gm__ int8_t* addr, int8_t value)

__aicore__ inline void asc_store_dev(__gm__ uint8_t* addr, uint8_t value)

__aicore__ inline void asc_store_dev(__gm__ int16_t* addr, int16_t value)

__aicore__ inline void asc_store_dev(__gm__ uint16_t* addr, uint16_t value)

__aicore__ inline void asc_store_dev(__gm__ int32_t* addr, int32_t value)

__aicore__ inline void asc_store_dev(__gm__ uint32_t* addr, uint32_t value)

__aicore__ inline void asc_store_dev(__gm__ int64_t* addr, int64_t value)

__aicore__ inline void asc_store_dev(__gm__ uint64_t* addr, uint64_t value)
```

## 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| addr     | 输入     | 目标GM地址。|
| value     | 输入     | 待写入目标的数据。|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```c++
__gm__ int32_t* addr;
int32_t value = 2;
asc_store_dev(addr, value);
```