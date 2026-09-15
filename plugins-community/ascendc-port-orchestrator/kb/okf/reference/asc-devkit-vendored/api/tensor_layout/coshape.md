---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "Coshape"
description: "Coshape(layout)：返回 Layout 陪域（codomain）的形状，即有效逻辑坐标映射到的空间形状；模板可用索引序列递归选子结构，要求 shape 与 stride 维度相同。"
tags: [tensor_layout]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/tensor_api/struct/layout/Coshape.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/Coshape.md
# Coshape

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能概述

Coshape表示有效逻辑坐标映射的空间的形状，即陪域(codomain)的形状。

## 函数原型

```cpp
template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Coshape(const Layout<Shape, Stride>& layout);
```

### 参数说明

**表 1** 模板参数说明

  | 参数名 | 类型 | 描述 |
|--------|------|------|
| Shape | 输入 | 组成Layout的shape的类型，即元组（tuple）类型。 |
| Stride | 输入 | 组成Layout的stride的类型，即元组（tuple）类型。 |
| Is... | size_t... | 索引序列，用于编译时递归选择shape和stride的子结构。 |

**表 2** 参数说明

  | 参数名 | 类型 | 描述 |
|--------|------|------|
| layout | 输入 | Layout用于描述张量的布局。 |

### 返回值

返回Layout的陪域(codomain)的形状。

### 约束条件

shape和stride及其递归到某层的子结构，需满足二者的维度相同。

### 示例代码

   ```cpp
  using namespace AscendC::Te;
  auto layout = MakeLayout(MakeShape(10, 20), MakeStride(1, 100));
  auto coshape = Coshape(layout); //coshape = 1910
  ```
