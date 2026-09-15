---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "GetShape"
description: "GetShape(layout)：从 Layout 取出描述张量形状的 Shape 对象，模板参数 Is... 可选做多级索引递归选取；constexpr 接口，要求 layout 有效。"
tags: [tensor_layout]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/tensor_api/struct/layout/GetShape.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/GetShape.md
# GetShape

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

获取描述张量形状的Shape对象。

## 函数原型

```cpp
template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto GetShape(const Layout<ShapeType, StrideType>& layout)
```

## 参数说明

**表 1** 模板参数说明

| 参数名 | 输入/输出 | 描述 |
| -------- | ----------- | ------ |
| Is... | 输入 | 可选。多级索引递归选取。 |
| ShapeType | 输入 | Layout的shape类型。 |
| StrideType | 输入 | Layout的stride类型。 |

**表 2** 参数说明

| 参数名 | 输入/输出 | 描述 |
| -------- | ----------- | ------ |
| layout | 输入 | Layout对象。 |

## 返回值说明

描述张量形状的Shape对象。

## 约束说明

- layout必须是有效的Layout对象。
- 索引Is...必须在有效范围内。

## 调用示例

```cpp
using namespace AscendC::Te;

auto layout = MakeNDLayout(128,128);
// 无模板实参时Is... 为空，等价于layout.Shape() 得到整个shape元组
auto shapeTuple = GetShape(layout);

// 带索引时等价于layout.Shape<0>()，按递归取shape子结构或某一维
auto s0 = GetShape<0>(layout);
```
