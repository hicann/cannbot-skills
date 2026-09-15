---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_l13d_fmatrix_config"
description: "asc_copy_l12l0a/l12l0b 的 3D 搬运 Feature map 属性 union：l1_height、l1_width 取值 [1,32767]，上下左右四个 padding 大小各取 [0,255]。"
tags: [struct]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/struct/asc_l13d_fmatrix_config.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/struct/asc_l13d_fmatrix_config.md
# asc_l13d_fmatrix_config

asc_l13d_fmatrix_config用于设置asc_copy_l12l0a/asc_copy_l12l0b的3D格式搬运接口的Feature map属性参数。

## 结构体具体定义

```cpp
constexpr uint64_t ASC_DEFAULT_L13D_FMATRIX_CONFIG_VALUE = 0;
union asc_l13d_fmatrix_config {
    uint64_t config = ASC_DEFAULT_L13D_FMATRIX_CONFIG_VALUE;
    struct {
        uint16_t l1_height;
        uint16_t l1_width;
        uint8_t padding_left_size;
        uint8_t padding_right_size;
        uint8_t padding_top_size;
        uint8_t padding_bottom_size;
    };
};
```

## 字段详解

|字段名|字段含义|
|----------|----------|
| l1_height | Feature map的height，取值范围：l1_height∈[1, 32767]。 |
| l1_width | Feature map的width，取值范围：l1_width∈[1, 32767]。 |
| padding_left_size | 左侧填充的大小，取值范围：padding_left_size ∈[0, 255] 。|
| padding_right_size | 右侧填充的大小，取值范围：padding_right_size ∈[0, 255] 。|
| padding_top_size | 上侧填充的大小，取值范围：padding_top_size ∈[0, 255] 。|
| padding_bottom_size | 下侧填充的大小，取值范围：padding_bottom_size ∈[0, 255] 。|
