# Layer 1: Tile 原语层

Tile 是 catlass 的最小可组合单元。开发者**绝大多数情况不直接声明** Tile 类型——BlockMmad 的默认模板参数自动推导。

## 1.1 TileMmad

**功能**：完成 L0 buffer 上的矩阵乘累加（MMAD）。

**位置**：`catlass/include/catlass/gemm/tile/tile_mmad.hpp`

**自动推导**：BlockMmad 默认 `TileMmad = Gemm::Tile::TileMmad<ArchTag, AType, BType, BiasType>`

**手动指定场景**：仅在需要特殊 MMAD 实现时替换。常规开发不涉及。

## 1.2 TileCopy

**功能**：数据搬运。包含以下搬运组件：

| 组件 | 路径 | 用途 |
|------|------|------|
| `CopyGmToL1` | GM → L1 | 把 A/B 从全局内存搬到 L1 |
| `CopyL1ToL0` | L1 → L0A/L0B | 搬入计算 buffer |
| `CopyL0CToGm` | L0C → GM | 计算结果写出 |

**位置**：`catlass/include/catlass/gemm/tile/tile_copy.hpp`

**自动推导**：BlockMmad 默认 `TileCopy = Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType>`

**手动指定场景**：工程优化——替换特定搬运策略。

> 示例：小 M 场景（`M < 8`），使用 `CopyGmToL1IntervalDataCopy` 代替默认逐行搬运，来自 `06_optimized_matmul`。
>
> ```cpp
> struct TileCopyOpt : public Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType> {
>     using CopyGmToL1A = Gemm::Tile::CopyGmToL1IntervalDataCopy<ArchTag, AType>;
>     // ... 替换特定搬运组件，不改整体结构
> };
> ```

## 1.3 Epilogue Tile

Epilogue 有自己的 Tile 层，在 `catlass/include/catlass/epilogue/tile/`：

| Tile | 用途 |
|------|------|
| `TileCopy` | 搬运（GM→UB, UB→GM） |
| `TileElemWiseGelu` | GELU 激活 |
| `TileElemWiseSilu` | SILU 激活 |
| `TileElemWiseRelu` | RELU 激活 |
| `TileElemWiseClamp` | 数值裁剪 |

**签名模板**：`template <class ArchTag_, class ComputeType_, int COMPUTE_LENGTH_>`

> **原则**：99% 的场景不需要关心 Tile 层。只在自定义 Epilogue 或特殊搬运优化时才手动指定。
