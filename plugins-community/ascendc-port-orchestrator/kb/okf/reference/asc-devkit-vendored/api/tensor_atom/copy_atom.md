---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "CopyAtom"
description: "Tensor API 的搬运原子操作结构体 CopyAtom，以 std::tuple 封装搬运操作的全部参数，由 MakeCopy 构造并与 Copy 函数配合使用，仅 Ascend 950PR/950DT 支持。"
tags: [tensor_atom]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/tensor_api/struct/atom/CopyAtom.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/atom/CopyAtom.md
# CopyAtom

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

CopyAtom用于定义搬运原子操作，封装了搬运操作的所有信息。

## 结构体定义

```cpp
template <typename... Args>
struct CopyAtom {
    std::tuple<Args...> value;
};
```

## 字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| value | std::tuple<Args...> | 存储复制操作参数的元组。 |

## 约束说明

- Args的数量和类型必须与复制操作的要求匹配。
- CopyAtom通常与Copy函数配合使用。

## 调用示例

```cpp
// 创建CopyAtom
auto copyAtom = AscendC::Te::MakeCopy(arg1, arg2, arg3);

// 执行复制操作
AscendC::Te::Copy(copyAtom, dst, src);
```