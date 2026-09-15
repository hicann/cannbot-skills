---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "MakeStride"
description: "MakeStride(t...)：构造描述各维内存步长的 Stride 对象的 constexpr 可变参函数，各元素可为整型变量、Std::Int 整型常量或嵌套 Stride 子结构以构造层次化 Stride。"
tags: [tensor_layout]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/tensor_api/struct/layout/MakeStride.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/MakeStride.md
# MakeStride

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

构造Stride对象，用于描述各维度在内存中的步长。支持传入多个步长值，也支持传入嵌套的Stride子结构以构造层次化Stride。

## 函数原型

```cpp
template <typename... Ts>
__aicore__ inline constexpr Stride<Ts...> MakeStride(const Ts&... t)
```

## 参数说明

**表 1** 模板参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| Ts... | 输入 | Stride各元素的类型，可以是整型变量、`Std::Int`整型常量，或嵌套的Stride子结构类型。 |

**表 2** 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| t | 输入 | Stride的各维度步长或子结构，可变参数。 |

## 返回值说明

返回`Stride<Ts...>`对象。

## 约束说明

参数`t`需满足可用于构造`Stride<Ts...>`对象。

## 调用示例

```cpp
using namespace AscendC::Te;

// 构造普通Stride
auto stride = MakeStride(1, 100, 200);

// 构造层次化Stride
auto fractalStride = MakeStride(MakeStride(1, 16), MakeStride(32, 512));

auto stride0 = Std::get<0>(stride);                    // stride0 = 1
auto innerStride = Std::get<1>(Std::get<0>(fractalStride)); // innerStride = 16
```
