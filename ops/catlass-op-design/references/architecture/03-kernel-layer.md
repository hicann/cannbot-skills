# Layer 3: Kernel 组装层

Kernel 把 BlockMmad + BlockEpilogue + BlockScheduler 组装成一个可执行的计算单元。

---

## 3.1 Kernel 类型（Matmul 系列）

> 来自 catlass example 实际使用类型。

| Kernel | 模板参数 | 使用场景 | 参考 example |
|--------|---------|---------|-------------|
| `BasicMatmul<MM, EP, BS>` | BlockMmad + BlockEpilogue + BlockScheduler | **纯矩阵乘**（EP=void） | `00_basic_matmul` |
| `MatmulActivation<MM, EP, BS>` | 同上 | **有 Epilogue**（激活/偏置） | `27_matmul_gelu`, `26_matmul_relu`, `28_matmul_silu` |
| `OptimizedMatmul<MM, EP, BS>` | 同上 | 带 Preload 优化的 matmul | `06_optimized_matmul`, `21_basic_matmul_preload_zN` |
| `SplitkMatmul<MM, EP, BS>` | 同上 | 多核切 K | `09_splitk_matmul` |
| `SingleCoreSlicekMatmul<MM, EP, BS>` | 同上 | 单核切 K | `34_single_core_splitk_matmul` |
| `SmallMatmul<MM, EP, BS>` | 同上 | 小 shape（任务块数 ≤ AIC 核数） | `31_small_matmul` |
| `MatmulFullLoadA<MM, EP, BS>` | 同上 + 特殊 Swizzle | A 全量驻留 L1 | `25_matmul_full_loadA` |
| `MatmulBias<MM, EP, BS>` | 同上（BlockMmad 含 BiasType） | Matmul + Bias | `20_matmul_bias` |
| `QuantMatmulMultiStageWorkspace<MM, EP, BS, STAGES>` | 同上 + workspaceStages | 量化 Matmul（AIC/AIV 协同） | `12_quant_matmul` |

### 选型决策

```
有 Epilogue（激活/偏置/反量化 等后处理）？
  ├─ 量化 → QuantMatmulMultiStageWorkspace
  ├─ 激活 → MatmulActivation
  ├─ Bias → MatmulBias（BlockMmad 传 BiasType）
  └─ 复杂 Epilogue（独立 X/D）→ MatmulEpilogue
无 Epilogue（纯矩阵乘）？
  ├─ 小 shape → SmallMatmul
  ├─ 大 K（需要切 K）→ SplitkMatmul / SingleCoreSlicekMatmul
  ├─ 全载 A → MatmulFullLoadA
  ├─ Preload 优化 → OptimizedMatmul
  └─ 默认 → BasicMatmul
```

---

## 3.2 标准组装（无 Epilogue）

```cpp
// 来自 00_basic_matmul / examples
using ArchTag = Arch::AtlasA2;
using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
using L1TileShape = GemmShape<128, 256, 256>;
using L0TileShape = GemmShape<128, 256, 64>;
using AType = Gemm::GemmType<half, layout::RowMajor>;
using BType = Gemm::GemmType<half, layout::RowMajor>;
using CType = Gemm::GemmType<float, layout::RowMajor>;

// Step 1: BlockMmad
using BlockMmad = Gemm::Block::BlockMmad<
    DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

// Step 2: BlockEpilogue = void（无后处理）
using BlockEpilogue = void;

// Step 3: BlockScheduler
using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

// Step 4: Kernel
using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
```

## 3.3 有 Epilogue 的组装（GELU 激活）

```cpp
// 来自 27_matmul_gelu / examples
using ArchTag = Arch::AtlasA2;
using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
// ... BlockMmad 同 3.2 ...

// Step 2: BlockEpilogue（用 MatmulActivation Kernel）
using EpiloguePolicy = Epilogue::EpilogueAtlasA2ElemWiseNoSource;
using DType = Gemm::GemmType<half, layout::RowMajor>;
constexpr uint32_t computeLength = 16384; // 需从 L0TileShape 计算
using TileGelu = Epilogue::Tile::TileElemWiseGelu<ArchTag, CType, computeLength>;
using TileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, DType>;
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    EpiloguePolicy, CType, DType, TileGelu, TileCopy>;

// Step 3: BlockScheduler（根据 M vs N 选方向）
using BlockScheduler = (M >= N)
    ? Gemm::Block::GemmIdentityBlockSwizzle<3, 0>
    : Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

// Step 4: Kernel（★ 用 MatmulActivation 而不是 BasicMatmul）
using MatmulKernel = Gemm::Kernel::MatmulActivation<BlockMmad, BlockEpilogue, BlockScheduler>;
```

---

## 3.4 Device 调用与 Host 调用

| 模式 | 适用位置 | 写法 |
|------|---------|------|
| **Host 调用** | catlass example 的 `main()` 中 | `DeviceGemm<Kernel>` + `Initialize()` + `operator()` |
| **Device 调用** | op_kernel 内 | `Kernel{}(params)` |

**op_kernel 只能用 Device 调用**。DeviceGemm 封装了 host 侧 workspace 分配和 stream 调度，在算子工程中由 op_host + CANN 框架管理。

### Device 调用（op_kernel 内）

```cpp
GM_ADDR userWs = const_cast<GM_ADDR>(AscendC::GetUserWorkspace(workspace));
Catlass::GemmCoord problemShape{m, n, k};

// BasicMatmul
typename Kernel::Params params{
    problemShape,
    gmA, layout::RowMajor{problemShape.m(), problemShape.k()},
    gmB, layout::RowMajor{problemShape.k(), problemShape.n()},
    gmC, layout::RowMajor{problemShape.m(), problemShape.n()},
    userWs
};
Kernel{}(params);
```

### Host 调用（catlass example 中，仅供理解架构）

```cpp
// Host 侧完整调用链（★ op_kernel 中禁止这样写）
using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
MatmulKernel::Arguments arguments{problemShape, deviceA, deviceB, deviceC};
MatmulAdapter matmulOp;
matmulOp.CanImplement(arguments);
size_t sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
matmulOp.Initialize(arguments, deviceWorkspace);
matmulOp(stream, aicCoreNum);
```

---

## 3.5 Kernel::Params 差异

### BasicMatmul::Params

```cpp
struct Params {
    GemmCoord problemShape;   // {M, N, K}
    GM_ADDR ptrA; LayoutA layoutA;
    GM_ADDR ptrB; LayoutB layoutB;
    GM_ADDR ptrC; LayoutC layoutC;
    GM_ADDR userWs;
};
```

### MatmulActivation::Arguments

```cpp
// 注意：MatmulActivation 需要 GM workspace 指针
struct Arguments {
    GemmCoord problemShape;
    uint32_t workspaceSize;    // ★ sizeof(float) 或其他
    GM_ADDR ptrA;
    GM_ADDR ptrB;
    GM_ADDR ptrD;              // 输出
};
```

### QuantMatmul::Params（增量字段）

```cpp
typename Kernel::Params params{
    problemShape,
    gmA, layoutA, gmB, layoutB,
    gmScale, layoutScale,        // ★ 量化 scale
    gmPerTokenScale, layoutPTS,  // ★ per-token scale（可选）
    gmD, layoutD,
    userWs                        // ★ 必须
};
```

---

## 3.6 op_kernel 内的分支实例化

当有多种 dtype / 转置 / Swizzle 组合时，在 op_kernel 入口用 `if constexpr` 分支：

```cpp
// op_kernel 入口
auto tilingKey = /* 从 tiling 数据获取当前 key */;

if (/* 条件1：dtype=half, transA=0, transB=0 */) {
    using Kernel = NsMyOp::KernelHalfNN;
    // 取 workspace, 构造 Params, Kernel{}(params)
} else if (/* 条件2 */) {
    using Kernel = NsMyOp::KernelHalfNT;
    // ...
}
```

关键要点：
1. 每个 `if constexpr` 分支内独立 `using Kernel = ...;` 实例化完整模板
2. `AscendC::GetUserWorkspace(workspace)` 取 workspace
3. 构造 `Kernel::Params` 或直接在 Params 构造函数中传参
4. `Kernel{}(params);` 执行
