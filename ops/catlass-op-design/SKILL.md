---
name: catlass-op-design
description: "Analyze operator requirements and select CATLASS components (ArchTag, DispatchPolicy, TileShape, BlockMmad, BlockEpilogue, BlockScheduler, Kernel type). Use when designing new CATLASS-based Ascend C operators, selecting DispatchPolicy, determining TileShape, choosing Kernel type, or picking Epilogue components. Output: component selection tables, epilogue slot analysis, branch instantiation conditions, workspace estimation."
---

# CATLASS Kernel Design

## Source Code Locations

CATLASS 源码位于工作区根目录 `./catlass/`。

```
catlass/
├── include/catlass/
│   ├── arch/arch.hpp              # ArchTag: AtlasA2, Ascend950
│   ├── gemm/
│   │   ├── dispatch_policy.hpp    # DispatchPolicy 定义
│   │   ├── block/block_mmad.hpp   # BlockMmad 模板
│   │   ├── block/block_swizzle/   # GemmIdentityBlockSwizzle
│   │   ├── kernel/                # BasicMatmul, MatmulActivation, SplitK 等
│   │   ├── tile/                  # TileMmad, TileCopy
│   │   └── gemm_type.hpp          # GemmType, GemmShape
│   └── epilogue/
│       ├── block/block_epilogue*.hpp  # BlockEpilogue 特化
│       ├── tile/tile_elemwise_*.hpp   # TileElemWise 激活
│       └── tile/tile_copy.hpp         # Epilogue TileCopy
├── examples/                      # 60+ 算子示例
│   ├── 00_basic_matmul/           到 42_*  # A2/A3 通用
│   ├── 43_ascend950_* 到 57_*     # Ascend950
│   └── advanced/basic_matmul_aclnn/  # aclnn 工程集成
└── docs/zh/
    ├── 2_Design/01_kernel_design/  # matmul 总结、dispatch、swizzle
    ├── 3_API/gemm_api.md           # GEMM 分层架构
    └── 3_API/include/catlass/      # 各组件 API 文档
```

## Search Strategy

优先用 `rg` 搜索局部目录，不要整仓加载。

```bash
# Arch & DispatchPolicy
rg "struct.*MmadAtlasA2|struct.*MmadAscend950" catlass/include/catlass/gemm/dispatch_policy.hpp
rg "ArchTag|AtlasA2|Ascend950" catlass/include/catlass/arch/

# Block 组件
rg "BlockMmad|DispatchPolicy|L1TileShape|L0TileShape" catlass/include/catlass/gemm/block/
rg "GemmIdentityBlockSwizzle|Swizzle" catlass/include/catlass/gemm/block/block_swizzle/

# Kernel 类型
ls catlass/include/catlass/gemm/kernel/  # 查看全部 kernel
rg "class BasicMatmul|class MatmulActivation|class.*Kernel" catlass/include/catlass/gemm/kernel/

# Epilogue
rg "BlockEpilogue|EpilogueAtlasA2|EpilogueAscend950" catlass/include/catlass/epilogue/block/
rg "TileElemWise|TileCopy" catlass/include/catlass/epilogue/tile/

# 参考 example
rg "using MatmulKernel|using BlockMmad|BlockEpilogue|DispatchPolicy" catlass/examples/
```

## When to Use Each Source

- 理解分层架构 → `catlass/docs/zh/3_API/gemm_api.md`
- DispatchPolicy 选型 → `catlass/docs/zh/2_Design/01_kernel_design/03_dispatch_policies.md`
- matmul 模板总览 → `catlass/docs/zh/2_Design/01_kernel_design/04_matmul_summary.md`
- Swizzle 策略 → `catlass/docs/zh/2_Design/01_kernel_design/02_swizzle.md`
- 硬件约束 → `catlass/docs/zh/2_Design/01_kernel_design/00_basics/atlasA2_hardware_info.md`
- ArchTag 源码 → `catlass/include/catlass/arch/arch.hpp`
- BlockEpilogue 特化 → `catlass/include/catlass/epilogue/block/block_epilogue_*.hpp`

## Component Selection Methodology

### Step 1: Identify Operator Type

```
量化（有 scale/dequant）？ → QuantMatmul 路径
分组（多组独立 A×B）？     → Grouped Matmul 路径
纯 matmul / matmul+激活？   → 标准 Matmul 路径
├── 纯 matmul  → 场景 A
├── + 激活     → 场景 B
├── + Bias+激活 → 场景 C
├── 小 shape    → 场景 D
├── 大 K/需切K  → 场景 E
└── 需 Preload  → 场景 F
```

### Step 2: Select DispatchPolicy

```
DispatchPolicy 选型：
│
├── 量化算子（AIC/AIV 协同）
│   └── MmadAtlasA2PreloadAsyncWithCallback
│
├── Grouped Matmul
│   └── MmadAtlasA2PreloadAsync
│
├── 常规 Matmul（A2 芯片）
│   ├── 需要 ShuffleK or 预加载？ → MmadAtlasA2Preload
│   └── 默认                       → MmadAtlasA2Pingpong
│
└── Ascend950
    └── 参考 43_ascend950_* 到 57_ascend950_* 的 DispatchPolicy
```

### Step 3: Select Kernel Type

| 条件 | Kernel | 参考 example |
|------|--------|-------------|
| 纯 matmul，无 Epilogue | `BasicMatmul` | 00_basic_matmul |
| matmul + 激活（GELU/SILU/RELU） | `MatmulActivation` | 27_matmul_gelu |
| matmul + Bias + 激活 | `MatmulActivation` | 20_matmul_bias, 27 |
| 小 shape（taskBlocks < AIC） | `SmallMatmul` | 31_small_matmul |
| 大 K，需切 K | `SplitkMatmul` | 09_splitk_matmul |
| 单核切 K | `SingleCoreSlicekMatmul` | 34_single_core_splitk |
| A 全量 L1 常驻 | `MatmulFullLoadA` | 25_matmul_full_loadA |
| Preload 优化 | `OptimizedMatmul` | 06_optimized_matmul |
| 量化 | `QuantMatmulMultiStageWorkspace` | 12_quant_matmul |

### Step 4: Select BlockScheduler

```
M >= N → GemmIdentityBlockSwizzle<3, 0>
M < N  → GemmIdentityBlockSwizzle<3, 1>
```

### Step 5: BlockEpilogue Slot Analysis

当 BlockEpilogue ≠ void，强制执行：

1. 打开对应 EpilogueDispatchPolicy 的特化头文件
2. 读出模板形参列表（每个 = 一个 Tile 槽）
3. 逐槽标记 ✅（现成）/ 🔧（自定义）/ ❌（槽不够）
4. 输出槽位清单表格

### Step 6: Branch Conditions

枚举所有改变 catlass 模板参数的条件：
- 输入 dtype → AType, BType, CType
- transA / transB → LayoutA, LayoutB
- Swizzle 方向 → BlockScheduler
- 激活类型 → BlockEpilogue.Tile

输出合法组合表。

## Complete Design Output Template

按 [references/design-document.md](references/design-document.md) 模板输出：

1. 参考 Example 与选型理由
2. Catlass 组件选型表（ArchTag, DispatchPolicy, TileShape, BlockMmad, BlockEpilogue, BlockScheduler, Kernel）
3. BlockEpilogue 槽位清单（BlockEpilogue ≠ void 时）
4. Kernel 适配方案（example main() → op_kernel Device 调用）
5. 分支实例化条件 + 合法组合
6. Workspace 来源
7. 自定义 Tile 契约（如有）

## Architecture Reference

本 skill 的 `references/` 目录提供分层架构知识：

| 文档 | 内容 |
|------|------|
| [architecture/00-hardware-arch.md](references/architecture/00-hardware-arch.md) | ArchTag 映射、资源约束、内存层级 |
| [architecture/01-tile-layer.md](references/architecture/01-tile-layer.md) | Tile 原语（自动推导） |
| [architecture/02-block-layer.md](references/architecture/02-block-layer.md) | DispatchPolicy 详解、Swizzle、Epilogue |
| [architecture/03-kernel-layer.md](references/architecture/03-kernel-layer.md) | Kernel 类型、组装、Params |

## Never / Always

**NEVER**:
- 臆测 DispatchPolicy 参数或 TileShape 值
- 跳过 Epilogue 头文件的槽位确认
- 输出 op_kernel 文件名、CMake、构建命令
- 把 catlass example 整份照抄

**ALWAYS**:
- 先查 catlass 官方文档再选型
- 选型用表格呈现
- BlockEpilogue ≠ void 时先列槽位清单
- 枚举每个分支条件的取值和合法组合
- 引用 catlass 仓库内具体路径
