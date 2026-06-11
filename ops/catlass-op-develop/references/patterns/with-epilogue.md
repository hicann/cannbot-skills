# Matmul + Epilogue — 代码骨架

> **导航**：[architecture/01-kernel-assembly.md](../architecture/01-kernel-assembly.md) 的场景 B/C 展开
> 参考 example：`catlass/examples/27_matmul_gelu/`、`20_matmul_bias/`

## 子场景 1：matmul + 激活（GELU / SILU / RELU）

```cpp
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_gelu.hpp"  // 按激活换头文件
#include "catlass/gemm/kernel/matmul_activation.hpp"

// ... BlockMmad 同纯 matmul ...

using EpiloguePolicy = Catlass::Epilogue::EpilogueAtlasA2ElemWiseNoSource;
using DType = Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor>;
constexpr uint32_t computeLength = 16384;

using TileGelu = Catlass::Epilogue::Tile::TileElemWiseGelu<
    ArchTag, CType, computeLength>;
using TileCopy = Catlass::Epilogue::Tile::TileCopy<ArchTag, CType, DType>;

using BlockEpilogue = Catlass::Epilogue::Block::BlockEpilogue<
    EpiloguePolicy, CType, DType, TileGelu, TileCopy>;

// ★ Kernel 用 MatmulActivation（示例级；性能优先见下方提示）
using MatmulKernel = Catlass::Gemm::Kernel::MatmulActivation<
    BlockMmad, BlockEpilogue, BlockScheduler>;
```

> **⚠ 性能提示**：`MatmulActivation` 把整块 `[M,N]` fp32 C 走 HBM 往返，大 N 时是瓶颈。**性能优先 / 大 N** 应改用「多级轮转 workspace + `MmadAtlasA2PreloadAsyncWithCallback`」：量化用 `QuantMatmulMultiStageWorkspace`；非量化 fp16 可在 op_kernel/ 下建其 fp16 类比（去 scale、epilogue 换逐元素激活）。详见 [mmad-epilogue 选型](../../../catlass-op-design/references/mmad-epilogue-selection.md) §1。

**可用激活 Tile**：

| 激活 | Tile 类 | 头文件 |
|------|--------|--------|
| GELU | `TileElemWiseGelu` | `tile_elemwise_gelu.hpp` |
| SILU | `TileElemWiseSilu` | `tile_elemwise_silu.hpp` |
| RELU | `TileElemWiseRelu` | `tile_elemwise_relu.hpp` |

## 子场景 2：matmul + Bias

```cpp
// BlockMmad 中追加 BiasType
using BiasType = Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor>;
using BlockMmad = Catlass::Gemm::Block::BlockMmad<
    DispatchPolicy, L1TileShape, L0TileShape,
    AType, BType, CType, BiasType>;  // ★ 第 7 个模板参数
```

## 子场景 3：matmul + Bias + 激活

> 参考 example：`catlass/examples/20_matmul_bias/`、`03_matmul_add/`

```cpp
// BlockEpilogue 用 OneSource policy
using EpiloguePolicy = Catlass::Epilogue::EpilogueAtlasA2ElemWiseOneSource;
using TileBiasCopy = Catlass::Epilogue::Tile::TileCopy<ArchTag, BiasType, CType>;
using TileActivation = Catlass::Epilogue::Tile::TileElemWiseGelu<
    ArchTag, CType, computeLength>;
using TileCopy = Catlass::Epilogue::Tile::TileCopy<ArchTag, CType, DType>;

using BlockEpilogue = Catlass::Epilogue::Block::BlockEpilogue<
    EpiloguePolicy, CType, DType,
    TileBiasCopy,       // 槽 1: BiasCopy
    TileActivation,     // 槽 2: BiasAdd + Activation
    TileCopy>;          // 槽 3: Copy out
```

## EpilogueDispatchPolicy 与 Tile 槽数

| Policy | Tile 槽数 | 典型用途 |
|--------|:---:|------|
| `EpilogueAtlasA2ElemWiseNoSource` | 1 个计算 Tile | 纯激活（GELU/SILU/RELU） |
| `EpilogueAtlasA2ElemWiseOneSource` | 2 个计算 Tile | Bias + 激活 |
| `EpilogueAtlasA2PerTokenDequant` | 5 个 Tile | 反量化 |

**槽位必须在 DESIGN.md 中逐槽确认**（见 `catlass-op-design`）。代码中的 `using Tile = ...` 顺序和类型必须与槽位清单一致。

## 常见陷阱

| 陷阱 | 表现 | 正确做法 |
|------|------|---------|
| 有 Epilogue 但用 BasicMatmul | 编译/运行期错误 | 用 MatmulActivation |
| Epilogue 模板参数顺序错 | 编译期模板不匹配 | 对照 `block_epilogue*.hpp` 确认顺序 |
| computeLength 随意填 | 运行期 AIV UB 越界 | 从 L0TileShape 正确推导 |
