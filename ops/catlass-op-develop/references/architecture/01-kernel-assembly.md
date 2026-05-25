# 01: Kernel Assembly — using 链标准模式

> **导航**：[00-overview.md](./00-overview.md) → 本文 → [02-device-calling.md](./02-device-calling.md)

本文将 DESIGN.md 的 catlass 组件选型表映射到 op_kernel 内的 `using` 链。两种场景：无 Epilogue（纯 matmul）和有 Epilogue（matmul + 后处理）。

---

## 场景 A：无 Epilogue（纯 matmul）

> 参考 example：`catlass/examples/00_basic_matmul/`

```cpp
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/layout/layout.hpp"

namespace NsMyOp {

using ArchTag = Catlass::Arch::AtlasA2;
using DispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<true>;
using L1TileShape = Catlass::GemmShape<128, 256, 256>;
using L0TileShape = Catlass::GemmShape<128, 256, 64>;
using AType = Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor>;
using BType = Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor>;
using CType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;

using BlockMmad = Catlass::Gemm::Block::BlockMmad<
    DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
using BlockEpilogue = void;  // ★ 无后处理
using BlockScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

using MatmulKernel = Catlass::Gemm::Kernel::BasicMatmul<
    BlockMmad, BlockEpilogue, BlockScheduler>;

} // namespace NsMyOp
```

**关键点**：
- `BlockEpilogue = void;` — 无后处理时唯一写法
- `Kernel = BasicMatmul` — 纯 matmul 只能用 BasicMatmul
- CType 推荐 `float` — MMAD 累加精度

---

## 场景 B：有 Epilogue（matmul + GELU 激活）

> 参考 example：`catlass/examples/27_matmul_gelu/`

```cpp
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_gelu.hpp"
#include "catlass/gemm/kernel/matmul_activation.hpp"

// BlockMmad 部分同场景 A，追加：

using EpiloguePolicy = Catlass::Epilogue::EpilogueAtlasA2ElemWiseNoSource;
using DType = Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor>;
constexpr uint32_t computeLength = 16384; // 从 L0TileShape 推导

using TileGelu = Catlass::Epilogue::Tile::TileElemWiseGelu<
    ArchTag, CType, computeLength>;
using TileCopy = Catlass::Epilogue::Tile::TileCopy<ArchTag, CType, DType>;

using BlockEpilogue = Catlass::Epilogue::Block::BlockEpilogue<
    EpiloguePolicy, CType, DType, TileGelu, TileCopy>;

// ★ Kernel 用 MatmulActivation，不是 BasicMatmul
using MatmulKernel = Catlass::Gemm::Kernel::MatmulActivation<
    BlockMmad, BlockEpilogue, BlockScheduler>;
```

**关键点**：
- 有 Epilogue 时 `Kernel = MatmulActivation`（不是 BasicMatmul）
- `computeLength` 从 L0TileShape 推导，具体值从参考 example 确认
- Epilogue 头文件在 `catlass/include/catlass/epilogue/`，不在 `gemm/` 目录

---

## 场景 C：matmul + Bias + 激活

> 参考 example：`catlass/examples/20_matmul_bias/`、`catlass/examples/03_matmul_add/`

```cpp
// BlockMmad 中追加 BiasType
using BiasType = Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor>;
using BlockMmad = Catlass::Gemm::Block::BlockMmad<
    DispatchPolicy, L1TileShape, L0TileShape,
    AType, BType, CType, BiasType>;  // ★

// BlockEpilogue 用 OneSource policy（两个 Tile 槽）
using EpiloguePolicy = Catlass::Epilogue::EpilogueAtlasA2ElemWiseOneSource;
using TileBiasCopy = Catlass::Epilogue::Tile::TileCopy<ArchTag, BiasType, CType>;
// Tile slot 2: BiasAdd + Activation fused

using BlockEpilogue = Catlass::Epilogue::Block::BlockEpilogue<
    EpiloguePolicy, CType, DType, TileBiasCopy, TileElemWiseEpilogue>;

using MatmulKernel = Catlass::Gemm::Kernel::MatmulActivation<
    BlockMmad, BlockEpilogue, BlockScheduler>;
```

---

## 场景 D：MatmulEpilogue 需要独立 X/D 缓冲

`MatmulEpilogue::ToUnderlyingArguments` 默认把 X 与 D 设为同一指针。需要 **Y = A·B − X** 且 `y` 与 `x3` 是不同的 GM 时，须**手动**构造 Params：

```cpp
typename BlockEpilogue::Params epilogueParams{ /* ..., ptrX = x3, ... */ };
typename MatmulKernel::Params params{
    problemShape, gmA, layoutA, gmB, layoutB,
    /*ptrD=*/ y, layoutD,
    userWs,
    epilogueParams
};
```

不要照搬只暴露 `Arguments` 的 host 适配器路径。

---

## compiled 分支实例化

当有多种 dtype / 转置 / Swizzle 组合时，每个分支内独立 `using Kernel = ...;`。详见 [patterns/branch-instantiation.md](../patterns/branch-instantiation.md)。

## 强制规则

| 规则 | 说明 |
|------|------|
| Δ3 | 每个分支内独立实例化 `using Kernel = ...` |
| Δ5 | op_kernel 不得 `#include` 自身 tiling 文件 |
| Δ8 | 固定 `COMPUTE_LENGTH` 与小 M/N 不兼容 |

详见 [rules.md](../rules.md)。
