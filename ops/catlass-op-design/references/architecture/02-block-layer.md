# Layer 2: Block 组件层

Block 是 catlass 的核心组装层。Kernel 由三个 Block 组件拼成：

```
Kernel = BlockMmad + BlockEpilogue + BlockScheduler
```

开发者在这一层做**所有选型决策**。

---

## 2.1 BlockMmad — 矩阵乘主循环

### 模板签名

```cpp
// catlass/include/catlass/gemm/block/block_mmad.hpp
template <
    class DispatchPolicy,    // ★ 调度策略
    class L1TileShape,       // GemmShape<M, N, K>
    class L0TileShape,       // GemmShape<M, N, K>
    class AType,             // GemmType<InputDtype, LayoutA>
    class BType,             // GemmType<InputDtype, LayoutB>
    class CType,             // GemmType<AccumDtype, LayoutC>  ← 累加类型
    class BiasType = void,   // GemmType<BiasDtype, Layout>   ← 可选
    class TileCopy = ...,    // 自动推导
    class TileMmad = ...     // 自动推导
>
struct BlockMmad;
```

通过 `DispatchPolicy` 的不同值，命中不同的 `block_mmad_*.hpp` 偏特化。

### DispatchPolicy — 四种调度策略

> 来自 `catlass/docs/zh/2_Design/01_kernel_design/03_dispatch_policies.md`。

| DispatchPolicy | 关键参数 | 承载代码 | 功能 |
|---------------|---------|---------|------|
| `MmadAtlasA2Pingpong` | `STAGES(2)`, `ENABLE_UNIT_FLAG(true)` | `block_mmad_pingpong.hpp` | L1/L0 pingpong buffer，最简 |
| `MmadAtlasA2Preload` | `STAGES(2)`, `ENABLE_UNIT_FLAG`, `ENABLE_SHUFFLE_K` | `block_mmad_preload.hpp` | + Block 间预加载 + ShuffleK |
| `MmadAtlasA2PreloadAsync` | `PRELOAD_STAGES`, `L1_STAGES`, `L0A_STAGES`, `L0B_STAGES`, `L0C_STAGES`, `ENABLE_UNIT_FLAG`, `ENABLE_SHUFFLE_K` | `block_mmad_preload_async.hpp` | + nBuffer + 异步 + group 预加载 |
| `MmadAtlasA2PreloadAsyncWithCallback` | 同 PreloadAsync | `block_mmad_preload_async_with_callback.hpp` | + AIC/AIV Callback 同步 |

### DispatchPolicy 与 Kernel 类型、典型 example 对照

| DispatchPolicy | 常用 Kernel | 典型 example |
|---------------|------------|-------------|
| `MmadAtlasA2Pingpong` | `BasicMatmul` | `00_basic_matmul`, `01_batched_matmul`, `03_matmul_add`, `04_padding_matmul`, `09_splitk_matmul` |
| `MmadAtlasA2Preload` | `BasicMatmulPreload` 等 | `06_optimized_matmul`, `21_basic_matmul_preload_zN` |
| `MmadAtlasA2PreloadAsync` | Grouped Matmul Kernel | `02_grouped_matmul_slice_m`, `05_grouped_matmul_slice_k`, `11_grouped_matmul_slice_k_per_token_dequant` |
| `MmadAtlasA2PreloadAsyncWithCallback` | `QuantMatmulMultiStageWorkspace` | `10_grouped_matmul_slice_m_per_token_dequant`, `12_quant_matmul` |

### 选型决策树

```
量化（有 scale / dequant）？
  ├─ 是 → MmadAtlasA2PreloadAsyncWithCallback + QuantMatmulMultiStageWorkspace Kernel
  └─ 否 →
      ├─ 是 Grouped Matmul？
      │   └─ MmadAtlasA2PreloadAsync + Grouped Matmul Kernel
      ├─ 需要 ShuffleK 或预加载优化？
      │   └─ MmadAtlasA2Preload
      └─ 常规场景 → MmadAtlasA2Pingpong（默认首选）
```

### TileShape — 分块大小

```cpp
L1TileShape = GemmShape<M, N, K>;  // L1 上的一次分块
L0TileShape = GemmShape<M, N, K>;  // L0 上的一次分块
```

**约束**（AtlasA2, fp16, Pingpong STAGES=2）：

```
L1: m1*k1*2*2 + n1*k1*2*2 ≤ 512KB
L0A: m0*k0*2*2 ≤ 64KB
L0B: n0*k0*2*2 ≤ 64KB
L0C: m0*n0*4*1 ≤ 128KB
m0 = m1, n0 = n1
```

**默认值**：`L1TileShape = <128, 256, 256>`, `L0TileShape = <128, 256, 64>`。性能调优时从此出发调整。

### GemmType — 数据类型包装

```cpp
// AType/BType = 输入数据
using AType = Gemm::GemmType<half, layout::RowMajor>;
using BType = Gemm::GemmType<half, layout::RowMajor>;
// CType = ★ 累加类型（推荐 fp32，保证精度）
using CType = Gemm::GemmType<float, layout::RowMajor>;
// DType = 输出类型（只 Epilogue 需要）
using DType = Gemm::GemmType<half, layout::RowMajor>;
```

**注意**：`CType` 是 MMAD 累加精度，`DType` 是写入 GM 的输出精度（仅 Epilogue 场景需要区分）。

### 标准实例化

```cpp
// 最简 matmul (910b, fp16, RowMajor)
using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
using L1TileShape    = GemmShape<128, 256, 256>;
using L0TileShape    = GemmShape<128, 256, 64>;
using AType = Gemm::GemmType<half, layout::RowMajor>;
using BType = Gemm::GemmType<half, layout::RowMajor>;
using CType = Gemm::GemmType<float, layout::RowMajor>;

using BlockMmad = Gemm::Block::BlockMmad<
    DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
```

---

## 2.2 BlockScheduler — Swizzle 策略

> 来自 `catlass/docs/zh/2_Design/01_kernel_design/02_swizzle.md`。

控制不同 AICore 处理 GM 数据块的遍历顺序。

```cpp
// catlass/include/catlass/gemm/block/block_swizzle.hpp
template <int Offset, int Direction>
struct GemmIdentityBlockSwizzle;
```

| 参数 | 含义 | 取值 |
|------|------|------|
| Offset | 起始错开量 | 1（默认）/ 3（推荐） |
| Direction | 遍历方向 | 0（沿 N）/ 1（沿 M） |

**选型规则**（来自 catlass 官方）：
- `M >= N` → `GemmIdentityBlockSwizzle<3, 0>`
- `M < N` → `GemmIdentityBlockSwizzle<3, 1>`

> 来自 `27_matmul_gelu` 示例：根据 `m > n` 判断选择 `<3,0>` 还是 `<3,1>`。

---

## 2.3 BlockEpilogue — 后处理

> 代码在 `catlass/include/catlass/epilogue/block/block_epilogue.hpp`（**不在 gemm 目录下，命名空间 `Epilogue::`**）。

BlockEpilogue 在 BlockMmad 完成后对输出做逐元素后处理。

### EpilogueDispatchPolicy

| Policy | 头文件 | 槽数 | 典型用途 |
|--------|-------|------|---------|
| `EpilogueAtlasA2ElemWiseNoSource` | `block_epilogue_elemwise_no_source.hpp` | 1 个 Tile 槽 | 纯激活（GELU/SILU/RELU） |
| `EpilogueAtlasA2ElemWiseOneSource` | `block_epilogue_elemwise_one_source.hpp` | 2 个 Tile 槽 | Bias + 激活 |
| `EpilogueAtlasA2PerTokenDequant` | `block_epilogue_per_token_dequant.hpp` | 5 个 Tile 槽 | 反量化 |

### 无后处理

```cpp
using BlockEpilogue = void;
```

### 有后处理（以 GELU 为例）

```cpp
// 来自 27_matmul_gelu
using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2ElemWiseNoSource;
constexpr uint32_t computeLength = 16384; // 64 * 128 * 2B
using TileElemWiseEpilogue = Epilogue::Tile::TileElemWiseGelu<ArchTag, CType, computeLength>;
using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, DType>;

using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    EpilogueDispatchPolicy,   // Policy
    CType, DType,             // 累加类型, 输出类型
    TileElemWiseEpilogue,     // Tile 槽 1
    EpilogueTileCopy          // Tile 槽 2
>;
```

**关键参数 `computeLength`**：一次 Vector 计算的元素数，来自 L0TileShape。例如 `L0TileShape<128, 256, 64>` → `computeLength = N * K_0 / sizeof(CType)` 或类似公式。具体值需从参考 example 确认。

**对应 Kernel**：有 Epilogue 时必须用 `MatmulActivation` 而不是 `BasicMatmul`（见 Layer 3）。
