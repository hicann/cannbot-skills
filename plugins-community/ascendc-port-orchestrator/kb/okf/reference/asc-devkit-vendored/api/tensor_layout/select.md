---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "Select"
description: "按编译期索引 Is... 从 Layout 的 shape 与 stride 中挑选指定维度，组合成新的子 Layout 对象返回；索引须在有效范围内，constexpr 求值。"
tags: [tensor_layout]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/tensor_api/struct/layout/Select.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/Select.md
# Select

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

选择Layout的shape和stride指定维度组成新的layout对象并返回。

## 函数原型

```cpp
template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Select(const Layout<Shape, Stride>& layout)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| layout | 输入 | Layout对象。 |

## 返回值说明

返回子Layout对象。

## 约束说明

- layout必须是有效的Layout对象。
- 索引Is...必须在有效范围内。

## 调用示例

```cpp
using namespace AscendC::Te;

auto shape = MakeShape(10, 20, 30);
auto layout = MakeLayout(shape);

// 选择第0和第1维度
auto subLayout = Select<0, 1>(layout);
```
