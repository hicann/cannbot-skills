---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "MakeCoord"
description: "构造 Coord 对象的 constexpr 可变参函数 MakeCoord(t...)，参数个数须与 Shape 维数一致、必须为非负整数且在维度范围内，支持 size_t/int 与 Std::Int。"
tags: [tensor_coord]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/tensor_api/struct/coord/MakeCoord.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/coord/MakeCoord.md
# MakeCoord

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

构造Coord对象，用于定义张量的坐标。

## 函数原型

```cpp
template <typename... Ts>
__aicore__ inline constexpr Coord<Ts...> MakeCoord(const Ts&... t)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| t | 输入 | 各维度的坐标，可变参数。 |

## 返回值说明

返回Coord<Ts...>对象。

## 约束说明

- 参数数量必须与对应的Shape维度数量一致。
- 各参数必须为非负整数。
- 坐标值必须在对应Shape维度的有效范围内。
- 支持的数据类型包括：size_t、int等整数类型或者Std::Int类型。

## 理论性能说明


## 调用示例

```cpp
// 使用整数类型创建一个3维张量的坐标
auto coord = AscendC::Te::MakeCoord(5, 10, 15);

// 获取各维度的坐标
auto coord0 = AscendC::Std::get<0>(coord); // coord0 = 5
auto coord1 = AscendC::Std::get<1>(coord); // coord1 = 10
auto coord2 = AscendC::Std::get<2>(coord); // coord2 = 15

// 使用Std::Int类型创建一个3维张量的坐标
auto coordInt = AscendC::Te::MakeCoord(AscendC::Std::Int<5>{}, AscendC::Std::Int<10>{}, AscendC::Std::Int<15>{});

// 获取各维度的坐标
auto coordInt0 = AscendC::Std::get<0>(coordInt); // coordInt0 = 5
auto coordInt1 = AscendC::Std::get<1>(coordInt); // coordInt1 = 10
auto coordInt2 = AscendC::Std::get<2>(coordInt); // coordInt2 = 15
```