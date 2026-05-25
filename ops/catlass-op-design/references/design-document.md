# Catlass 章节模板（嵌入 DESIGN.md 的 catlass 部分）

> **导航**：新架构文档已建立 → 设计时请先按 [kernels/matmul.md](kernels/matmul.md) 等 kernel 路由指南完成组件选型，再使用本模板填入选型结果。架构背景知识见 [architecture/](architecture/)。

> 本模板仅描述与 **catlass kernel 代码生成**直接相关的设计章节。算子需求、I/O 表、aclnn 接口、OpDef、Tiling 注册等通用算子设计章节由上游 agent 按 通用规范填写。本 skill 不规定它们的格式。

---

## 1. 参考 Example 与选型理由

| 项 | 内容 |
|----|------|
| 参考 example | `catlass/examples/00_basic_matmul`（举例） |
| 选型理由 | 功能最接近、可在此基础上增加 BlockEpilogue |
| 变通点 | 与 example 的差异，如：example 是 main() 直 launch，本算子需拆为上游 tiling + Device 调用 |

若无完全匹配，写明缺口与基于哪个 example 如何改造（不照搬代码）。

---

## 2. Catlass 组件选型表

### 2.1 硬件与架构

| 组件 | 选型 | 说明 |
|------|------|------|
| 目标芯片 | ascend910b（举例） | 对应 `Arch::AtlasA2` |

### 2.2 BlockMmad（块级矩阵乘）

| 组件 | 选型 | 说明 |
|------|------|------|
| DispatchPolicy | `MmadAtlasA2Pingpong<true>` | 流水调度，详见 [matmul-templates.md](./matmul-templates.md) |
| L1TileShape | `<128, 256, 256>` | (M, N, K) |
| L0TileShape | `<128, 256, 64>` | (M, N, K) |
| AType | `half`, `RowMajor` | |
| BType | `half`, `RowMajor` | |
| CType | `float`, `RowMajor` | 中间累加用 fp32 |

> **代码映射**：`using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;`

### 2.3 BlockEpilogue（后处理）

如算子无后处理：`using BlockEpilogue = void;`，跳过 §2.3.0。

否则**强制**先填 §2.3.0 槽位清单，再列 §2.3.1 Tile 流水线顺序。

#### 2.3.0 BlockEpilogue 槽位清单（强制）

> 选定 `EpilogueDispatchPolicy` 后，立即打开对应特化（`catlass/include/catlass/epilogue/block/block_epilogue_<policy>.hpp`），把模板形参列表抄成下表。每个槽决策为 ✅ / 🔧 / ❌。

| 槽 # | 模板形参 | 接口签名 | 必要 typedef | 决策 | 选用 / 自定义 Tile |
|------|---------|---------|------------|------|--------------------|
| 1 | `TileXxx_` | `operator()(ubOut, ubIn0, ubIn1)` | `TileShape` | ✅ / 🔧 / ❌ | … |
| 2 | … | … | … | … | … |

**决策图例**：✅ 现成 Tile；🔧 粒度 A 自定义 Tile（接口对齐，详见 §6）；❌ 槽位不够 → 粒度 B 重写 BlockEpilogue（详见 §6）

#### 2.3.1 Tile 流水线顺序

按 Tile 流水线顺序列出：

| 顺序 | 环节 | 组件 | 说明 |
|------|------|------|------|
| 1 | 数据搬入 | `TileCopy` | GM → UB |
| 2 | 计算 | `TileElemWiseGelu` | 激活 |
| 3 | 数据搬出 | `TileCopy` | UB → GM |

> **代码映射**：上表组件按顺序填入 `BlockEpilogue<DispatchPolicy, CType, DType, ...Tile组件>`。

#### 2.3.2 自定义 Tile（catlass 无现成组件时）

按 [custom-epilogue.md](./custom-epilogue.md) 写明粒度（A / B）、Tile 名称、目标槽位（粒度 A）/ 完整 Tile 槽序列（粒度 B）、接口签名、数学行为、computeLength、UB 占用变化。**不写可编译代码**——下游 `catlass-op-develop` 负责落盘头文件。

### 2.4 BlockScheduler

| 组件 | 选型 | 说明 |
|------|------|------|
| 调度器 | `GemmIdentityBlockSwizzle<3, 0>` | offset=3, direction=0 |

### 2.5 Kernel

| 组件 | 选型 | 说明 |
|------|------|------|
| Kernel 类型 | `Gemm::Kernel::BasicMatmul` / `MatmulEpilogue` / `MatmulActivation` / `QuantMatmulMultiStageWorkspace` 之一 | |

组装：

- 非量化：`using Kernel = <Kernel类型><BlockMmad, BlockEpilogue, BlockScheduler>;`
- 量化：`using Kernel = QuantMatmulMultiStageWorkspace<BlockMmad, BlockEpilogue, BlockScheduler, /*workspaceStages=*/2>;`

> **代码映射**：`catlass-op-develop` 在 op_kernel 入口分支内 `typename Kernel::Params params{...}; Kernel{}(params);`。

---

## 3. Kernel 适配方案（catlass example → op_kernel）

| catlass example 中 | 本算子拆分到 |
|--------------------|-------------|
| `main()` 内计算 problemShape / blockDim / TileShape 等常量 | 上游 host tiling（按通用 tiling 规范实现） |
| `main()` 内 workspace 分配 | 上游 host tiling 计算大小，框架在运行时分配 |
| `main()` 内 `DeviceGemm<Kernel>` host launch | op_kernel 入口直接 `typename Kernel::Params params{...}; Kernel{}(params);` |

**op_kernel 必须用 Device 调用**，不得使用 `DeviceGemm` 适配器（详见 `catlass-op-develop`）。

---

## 4. 需要分支实例化的条件

任何改变 catlass 模板参数的条件都要列出：

| 条件 | 取值集合 | 影响 catlass 模板 |
|------|----------|-------------------|
| 输入 dtype | `half` / `bfloat16` / `float` | `AType` / `BType` |
| `transA` | 0 / 1 | catlass `LayoutA` |
| `transB` | 0 / 1 | catlass `LayoutB` |
| Swizzle offset | 0 / 3 | `BlockScheduler` |

合法组合（笛卡尔积裁剪后）：

| 合法组合 | Kernel 实例化内容 |
|----------|------------------|
| `dtype=half, transA=0, transB=0, swizzle=3` | `BasicMatmul + Pingpong + Swizzle<3,0>` |
| `dtype=half, transA=0, transB=1, swizzle=3` | `BasicMatmul + Pingpong (转置 B) + Swizzle<3,0>` |
| … | … |

> **本 skill 只列条件与合法组合**。条件如何在 host 装配为 TilingKey、kernel 入口如何写分支选择，由调用方按 `ascendc-tiling-design` 规范实现。

---

## 5. Workspace 量级来源

| 项 | 内容 |
|----|------|
| 大小来源 | `Kernel::GetWorkspaceSize(arguments)` 或具体表达式（如 SplitK 的 `M·N·sizeof(float)`） |
| 用途 | AIC→AIV 中间结果、SplitK ReduceAdd 缓冲等 |

具体字节数计算与对齐由上游 host tiling 完成。

---

## 6. 自定义 Tile / BlockEpilogue 契约（如有）

> 决策树详见 [custom-epilogue.md §0](./custom-epilogue.md#0-决策树先选粒度)。本节先**显式标注粒度**（A / B），再按对应的契约模板填表。

### 6.1 粒度 A：替换 BlockEpilogue 现有 Tile 槽（推荐）

| 项 | 说明 |
|----|------|
| 目标槽位 | §2.3.0 槽位清单中标 🔧 的那一槽（如 `BlockEpilogue<EpilogueAtlasA2PerTokenDequant>` 的 `TileOneBlkColumnBroadcastMul_`） |
| 接口签名 | 抄 catlass 头文件的模板形参与 `operator()` 入参；**严格对齐**才能直接替换 |
| 必要 typedef | `TileShape` / `COMPUTE_LENGTH` / `ElementCompute` 等 BlockEpilogue 特化用 `static_assert` 检查的成员 |
| Tile 名称 | 如 `TileXxxYyy`（与 catlass 内置不重名） |
| 数学行为差异 | 相对原槽 Tile 的运算追加（如「在原 mul 之后追加 gelu」） |
| computeLength | 与原槽一致 |
| UB 占用变化 | 0 / 增加 N 字节（说明能否 in-place 复用 ubOut） |
| 参考 | catlass 内**目标槽位**对应的现成 Tile 仅作签名对照 |

### 6.2 粒度 B：写新 BlockEpilogue 特化（重，仅在粒度 A 不可行时）

| 项 | 说明 |
|----|------|
| 新 DispatchPolicy 类型 | 如 `EpilogueAtlasA2MyFusion<UB_STAGES>` |
| 完整 Tile 槽序列 | 列出新 BlockEpilogue 特化的所有模板形参 + 每个 Tile 的接口签名 |
| 主循环伪代码 | 用伪代码描述每个 Tile 的调用顺序、UB 流水、CrossCore 同步点 |
| CrossCoreFlag 用法 | 涉及 AIC/AIV 协同时必填 |
| 代码量预估 | 约 500–700 行 |
| catlass 升级耦合 | 列出依赖的 catlass 内部 API |

---

## 撰写原则

1. **只写选型，不写实现细节**：用表格呈现，不贴大段代码
2. **对外接口由调用方决定**：本 skill 不规定 OpDef、Tiling 字段、CMake、构建命令
3. **BlockEpilogue ≠ void 时强制填槽位清单**（§2.3.0），再决定每槽 ✅ / 🔧 / ❌
4. **自定义 Tile / BlockEpilogue 必须显式标粒度**（§6 A / B）
5. **每个分支条件都要列**：上游需要据此装配 TilingKey 与 kernel 入口分支
6. **变通必须说明**：example 不完全匹配时说明改造点
