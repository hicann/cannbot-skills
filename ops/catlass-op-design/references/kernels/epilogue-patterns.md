# Kernel 路由：Epilogue 后处理模式

> **占位文件**。本文档将系统性地描述 catlass Epilogue 的各种组合模式和自定义方法。

## Epilogue 概述

BlockEpilogue 在 BlockMmad 计算完成后对输出矩阵做逐元素后处理。catlass 提供了多种 EpilogueDispatchPolicy，每种对应固定的 Tile 槽序列。

## 常见 EpilogueDispatchPolicy

| Policy | 所在头文件 | 槽数 | 典型用途 |
|--------|-----------|------|---------|
| `EpilogueAtlasA2ElemWiseNoSource` | `block_epilogue_elemwise_no_source.hpp` | 1 | 纯激活（GELU/SILU/RELU） |
| `EpilogueAtlasA2ElemWiseOneSource` | `block_epilogue_elemwise_one_source.hpp` | 2 | Bias + 激活 |
| `EpilogueAtlasA2PerTokenDequant` | `block_epilogue_per_token_dequant.hpp` | 5 | Per-token 反量化 |
| `EpilogueAtlasA2ElemWiseOneSrcFixpipe` | ... | ... | 950 芯片 Epilogue |

## 现有 Epilogue 组件（catlass 内置 Tile）

| Tile 名 | 功能 | 使用场景 |
|---------|------|---------|
| `TileElemWiseGelu` | GELU 激活 | Matmul + GELU |
| `TileElemWiseSilu` | SILU 激活 | Matmul + SILU |
| `TileElemWiseRelu` | RELU 激活 | Matmul + RELU |
| `TileElemWiseClamp` | 数值裁剪 | 输出范围控制 |
| `TileCopy` | 数据搬运 | Bias 搬运 / 格式转换 |
| `EpilogueTileSwizzle` | Swizzle 写回 | 输出重排 |

## 自定义 Tile（粒度 A）流程

1. 打开目标 EpilogueDispatchPolicy 的特化头文件，确认目标槽位的**接口签名**（模板形参集合、`operator()` 入参、必要 typedef）
2. 编写自定义 Tile 头文件，严格对齐签名
3. 在 catlass 拼装头中用自定义 Tile 替换原槽位的 Tile

详见 `references/custom-epilogue.md`。

> 完整的后处理模式清单和组合指南待后续补充。
