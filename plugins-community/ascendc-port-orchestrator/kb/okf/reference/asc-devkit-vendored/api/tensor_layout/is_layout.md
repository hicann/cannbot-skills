---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "is_layout"
description: "is_layout<T> 类型特征模板：通过成员常量 value 判断输入数据结构是否为 Layout 类型，value 为 true 即是 Layout，反之不是；用于编译期类型分支。"
tags: [tensor_layout]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/tensor_api/struct/layout/is_layout.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/is_layout.md
# is\_layout

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

判断输入的数据结构是否为Layout数据结构，可通过检查其成员常量value的值来判断。当value为true时，表示输入的数据结构是Layout类型；反之则为非Layout类型。

## 函数原型

```cpp
template <typename T> struct is_layout
```

## 参数说明

**表 1**  模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 根据输入的数据类型，判断是否为Layout数据结构。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
using namespace AscendC::Te;

// 初始化Layout数据结构并判断其类型
auto shape = MakeShape(10, 20, 30);
auto stride = MakeStride(1, 100, 200);

auto layoutMake = MakeLayout(shape, stride);
Layout<decltype(shape), decltype(stride)> layoutInit(shape, stride);

bool value = is_layout<decltype(shape)>::value; // value = false
value = is_layout<decltype(stride)>::value; // value = false

value = is_layout<decltype(layoutMake)>::value; // value = true
value = is_layout<decltype(layoutInit)>::value; // value = true
```
