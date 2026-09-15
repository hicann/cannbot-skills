---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_store_atomic_config"
description: "搬出原子操作配置 union，供 asc_get_store_atomic_config 读回：atomic_type 占 3bit，编码使能位与原子操作数据类型；atomic_op 占 2bit，0 为求和。"
tags: [struct]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/struct/asc_store_atomic_config.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/struct/asc_store_atomic_config.md
# asc_store_atomic_config

asc_store_atomic_config参数包含原子操作使能位与原子操作类型，用于[asc_get_store_atomic_config](../simd_atomic/asc_get_store_atomic_config.md)接口使用。

## 结构体具体定义

```cpp
constexpr uint64_t ASC_STORE_ATOMIC_DEFAULT_VALUE = 0x0000000000000000;
union asc_store_atomic_config {
    uint64_t config = ASC_STORE_ATOMIC_DEFAULT_VALUE;
    struct {
        uint64_t atomic_type : 3;
        uint64_t atomic_op : 2;
        uint64_t reserved1 : 59;
    };
};
```

## 字段详解

|字段名|字段含义|
|----------|----------|
| atomic_type | 原子操作使能位，默认值：0。<br>0：无原子操作。<br>1：使能原子操作，进行原子操作的数据类型为float。<br>2：使能原子操作，进行原子操作的数据类型为half。<br>3：使能原子操作，进行原子操作的数据类型为int16_t。<br>4：使能原子操作，进行原子操作的数据类型为int32_t。<br>5：使能原子操作，进行原子操作的数据类型为int8_t。<br>6：使能原子操作，进行原子操作的数据类型为bfloat16_t。<br>其余值无具体含义。|
| atomic_op | 原子操作类型，默认值：0。<br>0：求和操作。<br>其余值无具体含义。 |
