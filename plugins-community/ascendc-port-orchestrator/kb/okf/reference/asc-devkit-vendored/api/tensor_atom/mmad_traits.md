---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "MmadTraits"
description: "MmadTraits 模板结构体，定义矩阵乘加操作的类型与参数（OperationType + args），用于声明 MmadAtom 并调用 Mmad，操作数分别位于 L0C/L0A/L0B。"
tags: [tensor_atom]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/tensor_api/struct/atom/MmadTraits.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/atom/MmadTraits.md
# MmadTraits

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

MmadTraits用于定义矩阵乘加操作的特性，包括矩阵乘加操作类型和相关参数。

## 结构体定义

```cpp
template <typename MadOperation, typename... MadOpArgs>
struct MmadTraits {
    using OperationType = MadOperation;
    std::tuple<MadOpArgs...> args;
};
```

## 字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| OperationType | MadOperation | 矩阵乘加操作类型。 |
| args | std::tuple<MadOpArgs...> | 矩阵乘加操作参数。 |

## 约束说明

- MadOperation必须是有效的矩阵乘加操作类型。
- MadOpArgs的数量和类型必须与MadOperation的要求匹配。

## 调用示例

```cpp
using namespace AscendC::Te;
// 创建MmadTraits
using mmadTraits = MmadTraits<MmadOperation, MmadTraitDefault>;
// 使用mmadTraits声明MmadAtom对象，调用Mmad接口实现MmadOperation，其中l0CTensor是位于L0C上的计算结果矩阵，l0ATensor是位于L0A上的左矩阵，l0BTensor是位于L0B上的右矩阵，para是Mmad计算运行时参数。
Mmad(MmadAtom<mmadTraits>{}, l0CTensor, l0ATensor, l0BTensor, para);
```
