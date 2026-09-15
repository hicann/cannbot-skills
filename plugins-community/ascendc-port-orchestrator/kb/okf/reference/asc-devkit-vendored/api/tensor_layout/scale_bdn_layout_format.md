---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "ScaleBDNLayoutFormat"
description: "定义 ScaleBDN 缩放布局（一种特殊的 DN 格式），按 row/column 生成对齐后的 Layout 类型或对象；数据类型仅支持 fp8_e8m0_t。"
tags: [tensor_layout]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/tensor_api/struct/layout/ScaleBDNLayoutFormat.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/ScaleBDNLayoutFormat.md
# ScaleBDNLayoutFormat

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

ScaleBDNLayoutFormat用于定义ScaleBDN格式的布局，ScaleBDN格式是一种支持缩放的布局，也是一种特殊的DN格式。

## 结构体定义

```cpp
template <typename T>
struct ScaleBDNLayoutFormat {
    template <size_t row, size_t column>
    using type = ScaleBDNFormatLayout<T, row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeScaleBDNLayout<T, U, S>(row, column);
    }
};
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| T | 输入 | 数据类型模板参数。<br>支持的数据类型为：fp8_e8m0_t。 |
| row | 输入 | 矩阵的总行数。 |
| column | 输入 | 矩阵的总列数。 |

## 返回值

- 输入为编译时常量时，返回ScaleBDN格式的Layout类型。
- 输入为整型变量时，返回ScaleBDN格式的Layout对象。
- 返回对齐后的Layout，对齐方式及对应位置的参数大小说明详见Layout和层次化表述法。

## 约束说明

参数row和column需为size_t类型或Int整型常量。

## 调用示例

```cpp
// 创建ScaleBDN格式Layout
using namespace AscendC::Te;
// 根据flag的值，选择ScaleBND格式或ScaleBDN格式的类型
constexpr bool flag = true;
using MyLayoutType = conditional_t<flag, ScaleBNDFormatLayout<fp8_e8m0_t>, ScaleBDNFormatLayout<fp8_e8m0_t>>;
size_t scaleK = 128;
size_t n = 128;
auto layoutAL1 = MyLayoutType{}(scaleK, n);

// 编译时常量传参构造Layout
using MyScaleBDNLayout = ScaleBDNLayoutFormat<fp8_e8m0_t>::type<Std::Int<64>, Std::Int<32>>;
auto staticLayout = MyScaleBDNLayout{};

// 运行时变量传参构造Layout
ScaleBDNLayoutFormat<fp8_e8m0_t> scaleBDNFormat;
auto layout = scaleBDNFormat(64, 32);
```
