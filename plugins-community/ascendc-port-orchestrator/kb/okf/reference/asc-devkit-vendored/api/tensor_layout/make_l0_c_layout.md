---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "MakeL0CLayout"
description: "MakeL0CLayout(row, column)：创建 L0C 格式布局（Nz 分形的一种），描述矩阵计算结果在 L0C Buffer 上的排布，返回对齐后的 Layout 类型或对象。"
tags: [tensor_layout]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/tensor_api/struct/layout/MakeL0CLayout.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/MakeL0CLayout.md
# MakeL0CLayout

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

创建L0C格式的布局，L0CLayout是Nz分形的一种，用于描述矩阵计算的结果在物理位置为L0C Buffer的布局格式。

## 函数原型

```cpp
template <typename U, typename S>
__aicore__ inline decltype(auto) MakeL0CLayout(U row, S column)
```

## 参数说明

**表 1** 模板参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| U | 输入 | 行数类型，size_t或Int整型常量。 |
| S | 输入 | 列数类型，size_t或Int整型常量。 |

**表 2** 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| row | 输入 | 矩阵的总行数。 |
| column | 输入 | 矩阵的总列数。 |

## 返回值说明

- 输入为编译时常量时，返回L0C格式的Layout类型。
- 输入为整型变量时，返回L0C格式的Layout对象。
- 返回对齐后的Layout，对齐方式及对应位置的参数大小说明详见Layout和层次化表述法。

## 约束说明

参数row和column需为size_t类型或Int整型常量。

## 调用示例

```cpp
// 创建L0C格式Layout
using namespace AscendC::Te;
// 编译时常量传参构造Layout
auto staticLayout = MakeL0CLayout(Std::Int<32>{}, Std::Int<32>{});

// 运行时变量传参构造Layout
auto layout = MakeL0CLayout(32, 32);
```
