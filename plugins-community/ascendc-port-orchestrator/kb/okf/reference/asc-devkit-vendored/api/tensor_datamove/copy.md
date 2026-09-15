---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "Copy"
description: "Tensor API 统一数据搬运接口 Copy，由 CopyAtom 选择通路：GM2L1、L12L0A、L12L0B、L12BT、L12FB、L0C2GM、L0C2UB；参数为目的张量、源张量与可选坐标偏移。"
tags: [tensor_datamove]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/tensor_api/data_move/Copy.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/data_move/Copy.md

# Copy

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

执行数据复制操作，通过CopyAtom选择不同存储空间之间的数据搬运。Copy API是Tensor API中的统一数据搬运接口，支持在不同存储空间之间进行数据传输。

**表 1** 支持的数据搬运通路

| 操作类型 | 说明 |
|---------|------|
| CopyGM2L1 | Global Memory到L1 Buffer |
| CopyL12L0A | L1 Buffer到L0A Buffer |
| CopyL12L0B | L1 Buffer到L0B Buffer|
| CopyL12BT |  L1 Buffer到BiasTable Buffer|
| CopyL12FB |  L1 Buffer到Fixpipe Buffer |
| CopyL0C2GM | L0C Buffer搬出到Global Memory |
| CopyL0C2UB | L0C Buffer搬出到Unified Buffer |


## 函数原型

```cpp
template <typename Tp, const Tp& traits, typename T, typename... Params>
__aicore__ inline void Copy(const CopyAtom<T>& atomCopy, const Params& ...params)

template <typename T, typename... Params>
__aicore__ inline void Copy(const CopyAtom<T>& atomCopy, const Params& ...params)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| atomCopy | 输入 | [CopyAtom](../tensor_atom/copy_atom.md)对象，封装了具体的复制操作和traits配置。 |
| params | 输入 | 复制操作的参数，可变参数。通常包括：<br>- 目的张量（dst tensor）<br>- 源张量（src tensor）<br>- 可选：坐标偏移（Coord） |

## 返回值说明

无

## 约束说明

- atomCopy必须是有效的CopyAtom对象；
- params的数量和类型必须与复制操作的要求匹配；
- 源操作数和目的操作数的内存空间必须支持对应的复制操作。

## 调用示例

```cpp
using namespace AscendC::Te;
// 以CopyL0C2GM为例
// 调用方式1
Copy(CopyAtom<CopyTraits<CopyL0C2GM, FixpipeTraitDefault>>{}, globalC, l0CTensor);
// 调用方式2
CopyAtom<CopyTraits<CopyL0C2GM, FixpipeTraitDefault>>{}.Call(globalC, l0CTensor);
// 调用方式3
auto atomCopyL0C2GM = MakeCopy(CopyL0C2GM{}, FixpipeTraitDefault{});
atomCopyL0C2GM.Call(globalC, l0CTensor);
```

## 相关参考

- [CopyAtom](../tensor_atom/copy_atom.md)
- [CopyTraits](../tensor_atom/copy_traits.md)
- [DataCopyGM2L1](data_copy_gm2_l1.md)
- [DataCopyFromL1](data_copy_from_l1.md)
- [LoadData](load_data.md)
- [Fixpipe](fixpipe.md)