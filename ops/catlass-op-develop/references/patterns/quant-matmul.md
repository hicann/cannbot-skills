# Quant Matmul — 量化矩阵乘

> **导航**：[architecture/02-device-calling.md](../architecture/02-device-calling.md) 的 QuantMatmul 部分展开
> 参考 example：`catlass/examples/12_quant_matmul/`

量化矩阵乘使用 AIC/AIV 协同模式，与常规 matmul 有本质差异。

## 核心差异

| 项目 | 常规 Matmul | Quant Matmul |
|------|-----------|-------------|
| Kernel | BasicMatmul / MatmulActivation | **QuantMatmulMultiStageWorkspace** |
| Epilogue 执行位置 | AIC 侧 | **AIV 侧**（反量化 + 可选激活） |
| DispatchPolicy | MmadAtlasA2Pingpong | **MmadAtlasA2PreloadAsyncWithCallback** |
| 输入 | A, B | A, B, **scale, perTokenScale** |
| Workspace | userWs | AIC→AIV 中间结果传递 |
| 编译选项 | 标准 | **+ `-DBUILD_CATLASS_MODULE=ON`** |

## 完整 using 链

```cpp
// DispatchPolicy 必须用 PreloadAsyncWithCallback
using DispatchPolicy = Catlass::Gemm::MmadAtlasA2PreloadAsyncWithCallback;
// ... BlockMmad 同常规 ...

// 量化 Epilogue（AIV 侧执行）
using EpiloguePolicy = Catlass::Epilogue::EpilogueAtlasA2PerTokenDequant;
// ... BlockEpilogue 槽位按 DESIGN.md 填 ...

// ★ Kernel 用 QuantMatmulMultiStageWorkspace
using MatmulKernel = Catlass::Gemm::Kernel::QuantMatmulMultiStageWorkspace<
    BlockMmad, BlockEpilogue, BlockScheduler, /*workspaceStages=*/2>;
```

## Device 调用

```cpp
GM_ADDR userWs = const_cast<GM_ADDR>(AscendC::GetUserWorkspace(workspace));
Catlass::GemmCoord problemShape{m, n, k};

typename MatmulKernel::Params params{
    problemShape,
    gmA, layoutA,
    gmB, layoutB,
    gmScale, layoutScale,                          // ★ scale 输入
    gmPerTokenScale, layoutPerTokenScale,           // ★ per-token scale
    gmD, layoutD,
    userWs
};
MatmulKernel{}(params);
```

## 量化 + 激活融合

`12_quant_matmul` + `27_matmul_gelu` 组合时，BlockEpilogue Tile 尾部追加 `TileElemWiseGelu`：

```cpp
// BlockEpilogue 的 Tile 序列末尾追加 activation tile
using TileActivation = Catlass::Epilogue::Tile::TileElemWiseGelu<
    ArchTag, CType, computeLength>;

using BlockEpilogue = Catlass::Epilogue::Block::BlockEpilogue<
    EpiloguePolicy, CType, DType,
    TileDequant1, TileDequant2, TileDequant3,
    TileActivation,     // ★ 激活在反量化之后
    TileCopy>;
```

**注意**：
- `computeLength` 与 EpilogueTileShape 匹配
- 激活 Tile 放在反量化 Tile 之后、Copy Tile 之前
- workspace 大小由 `Kernel::GetWorkspaceSize` 计算

## 编译选项

```cmake
target_compile_options(my_op_kernel PRIVATE
    -I${CATLASS_DIR}/include
    -DCATLASS_ARCH=2201
    -DBUILD_CATLASS_MODULE=ON      # ★ 量化必需
)
```

## 强制规则

| 规则 | 说明 |
|------|------|
| Δ7 | 必须用 `QuantMatmulMultiStageWorkspace`；含 scale 和 perTokenScale 输入 |
| Δ1 | BlockEpilogue 在 AIV 侧执行，不可与 AIC 侧混淆 |

> 量化 + **SwiGLU 等跨 N-half 门控**时，注意 Δ10：stock `QuantMatmulMultiStageWorkspace` 每 slot 单 tile 无法跨 N-block 取 `+H` 配对列，必须按输出形状 `[M, H]` 调度、每输出块产出左右两路 C tile。见 [rules.md](../rules.md) Δ10 与 [mmad-epilogue 选型](../../../catlass-op-design/references/mmad-epilogue-selection.md) §4.2。
