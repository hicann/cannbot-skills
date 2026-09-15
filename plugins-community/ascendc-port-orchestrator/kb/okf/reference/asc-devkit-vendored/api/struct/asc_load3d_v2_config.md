---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_load3d_v2_config"
description: "Load3Dv2 的 repeat 参数 union：rpt_stride 迭代间起始地址距离 [0,65535]、rpt_time 迭代次数 [0,255] 默认 1、rpt_mode 选方向(0 沿 height、1 沿 width)。"
tags: [struct]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/struct/asc_load3d_v2_config.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/struct/asc_load3d_v2_config.md
# asc_load3d_v2_config

asc_load3d_v2_config用于设置Load3Dv2接口的repeat参数。

## 结构体具体定义

```cpp
constexpr uint64_t ASC_DEFAULT_LOAD3D_V2_CONFIG_VALUE = 0x0000000000010000;
union asc_load3d_v2_config {
    uint64_t config = ASC_DEFAULT_LOAD3D_V2_CONFIG_VALUE;
    struct {
        uint64_t rpt_stride : 16;
        uint64_t rpt_time : 8;
        uint64_t rpt_mode : 1;
        uint64_t reserved2 : 39;
    };
};
```

## 字段详解

|字段名|字段含义|
|----------|----------|
| rpt_stride | height/width方向上的前一个迭代与后一个迭代起始地址的距离，取值范围：n∈[0, 65535]，默认值为0。 |
| rpt_time | height/width方向上的迭代次数，取值范围：repeatTime ∈[0, 255] 。默认值为1。|
| rpt_mode | 控制repeat迭代的方向，取值范围：k∈[0, 1] 。默认值为0。<br> 0：迭代沿height方向；<br>1：迭代沿width方向 |
