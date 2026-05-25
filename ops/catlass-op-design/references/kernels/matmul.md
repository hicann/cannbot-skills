# Kernel 路由：Matmul 类算子

> 本指南是 catlass 知识体系的**顶层路由入口**。设计时按本文档的决策流程，从用户需求走到最终的组件选型和代码骨架。架构背景知识见 [architecture/](../architecture/)。

---

## 前置

**必读文档**（按顺序）：

| # | 文档 | 关键内容 |
|---|------|---------|
| 1 | [architecture/00-hardware-arch.md](../architecture/00-hardware-arch.md) | NpuArch 映射、ArchTag、资源约束 |
| 2 | [architecture/02-block-layer.md](../architecture/02-block-layer.md) | ★ DispatchPolicy 决策树、BlockEpilogue 组装 |
| 3 | [architecture/03-kernel-layer.md](../architecture/03-kernel-layer.md) | ★ Kernel 选型、Device 调用、Params 结构 |

[Tile 层](../architecture/01-tile-layer.md) 通常自动推导，仅了解即可。

---

## Step 1: 确认运算符类型

```
是量化算子（有 scale / dequant）？ → kernels/quant-matmul.md
是分组 matmul（多组独立 A_i×B_i）？ → kernels/grouped-matmul.md
有复杂后处理（自定义 Tile）？ → 先按本文 matmul 流程，再参考 kernels/epilogue-patterns.md
纯矩阵乘 / 矩阵乘+激活 → 继续本文
```

---

## Step 2: 提取关键决策参数

| 参数 | 示例值 | 影响 catlass 模板 |
|------|--------|-----------------|
| 目标芯片 | ascend910b | Arch::AtlasA2 |
| 输入 dtype | half / bfloat16 | AType, BType |
| 输入 layout | RowMajor / ColumnMajor | LayoutA, LayoutB |
| 输出 dtype | half / float | CType（累加）/ DType（写出） |
| transA / transB | 0 / 1 | 映射到 Layout 参数 |
| 有无 Bias | 有 / 无 | BlockMmad.BiasType, BlockEpilogue 选型 |
| 有无激活 | GELU / SILU / RELU / 无 | BlockEpilogue 选型 |
| 问题规模 | M, N, K 值 | DispatchPolicy / Kernel / TileShape |

---

## Step 3: 按场景选择组件

### 场景 A: 纯矩阵乘（无 Epilogue）

```
公式：C = A × B

BlockMmad:
  DispatchPolicy: MmadAtlasA2Pingpong<true>
  L1TileShape:    GemmShape<128, 256, 256>
  L0TileShape:    GemmShape<128, 256, 64>
  CType:          GemmType<float, Layout>   ← 累加精度

BlockEpilogue: void

BlockScheduler:
  M >= N → GemmIdentityBlockSwizzle<3, 0>
  M < N  → GemmIdentityBlockSwizzle<3, 1>

Kernel: BasicMatmul
```

### 场景 B: 矩阵乘 + 激活

```
公式：D = Activation(A × B)

BlockMmad: 同场景 A

BlockEpilogue:
  DispatchPolicy: EpilogueAtlasA2ElemWiseNoSource
  Tile:           根据激活函数选 TileElemWiseGelu / TileElemWiseSilu / TileElemWiseRelu
  computeLength:  从 L0TileShape 计算（参考 27_matmul_gelu: computeLength=16384）

Kernel: ★ MatmulActivation（不是 BasicMatmul）
```

**参考 example**: `27_matmul_gelu`, `26_matmul_relu`, `28_matmul_silu`

### 场景 C: 矩阵乘 + Bias + 激活

```
公式：D = Activation(A × B + Bias)

BlockMmad:
  BiasType: GemmType<BiasDtype, Layout>   ← ★

BlockEpilogue:
  DispatchPolicy: EpilogueAtlasA2ElemWiseOneSource
  Tile 槽1:        TileCopy（Bias 搬运）
  Tile 槽2:        TileElemWise（BiasAdd + Activation 融合）

Kernel: MatmulActivation
```

**参考 example**: `20_matmul_bias`, `03_matmul_add`

### 场景 D: 小 Shape（任务块数 ≤ AIC 核数）

```
条件：taskBlocks = CeilDiv(M, m1) * CeilDiv(N, n1) < aicCoreNum 且 K ≤ k1

Kernel: SmallMatmul

参考 example: 31_small_matmul
```

### 场景 E: 需 SplitK / Preload / 其他 Kernel

```
K 大、M*N 分核不足  → SplitkMatmul / SingleCoreSlicekMatmul（09/34）
需带宽优化           → DispatchPolicy 换 MmadAtlasA2Preload + Kernel 换 OptimizedMatmul（06）
A 全量入 L1          → MatmulFullLoadA（25）
```

---

## Step 4: 分支实例化条件

| 条件 | 影响 | 取值集合 |
|------|------|---------|
| 输入 dtype | AType, BType, CType | half / bfloat16 |
| transA | LayoutA | 0=RowMajor, 1=ColumnMajor |
| transB | LayoutB | 0=RowMajor, 1=ColumnMajor |
| Swizzle 方向 | BlockScheduler | M>=N → <3,0>; M<N → <3,1> |
| 激活类型 | BlockEpilogue.Tile | GELU / SILU / RELU |

列出合法组合：

| # | dtype | transA | transB | Swizzle | 激活 |
|---|-------|--------|--------|---------|------|
| 1 | half | 0 | 0 | <3,0> | 无 |
| 2 | half | 0 | 1 | <3,0> | 无 |
| 3 | half | 1 | 0 | <3,0> | 无 |
| 4 | half | 0 | 0 | <3,0> | GELU |
| ... | ... | ... | ... | ... | ... |

---

## Step 5: 输出设计章节

按 [design-document.md](../design-document.md) 模板填写：

1. **参考 Example** — 说明选择的 catlass example 和理由
2. **组件选型表** — Step 3 的选型结果，用表格呈现
3. **BlockEpilogue 槽位清单**（如有）— 从 catlass 头文件读出槽位，逐槽 ✅/🔧/❌
4. **适配方案** — example main() 如何拆分为 op_kernel Device 调用
5. **分支条件** — Step 4 的清单
6. **Workspace 来源** — 对应 Kernel 的 workspace 计算方式
7. **自定义 Tile 契约**（如有）

---

## 性能调优

1. 默认配置建立基线（Pingpong + 默认 TileShape）
2. 按 profiler 判断瓶颈（MTE2 / Cube / Vector）
3. 每次只改一个 `using`（DispatchPolicy / TileShape / Swizzle / Kernel）
4. 记录每次变更 (变量, 取值, 耗时, Δ%)

详见 `catlass-op-perf-tune` skill 和 `catlass/docs/zh/1_Practice/11_matmul_optimization.md`。

---

## 常见陷阱

| 陷阱 | 表现 | 正确做法 |
|------|------|---------|
| 有 Epilogue 但用 BasicMatmul | 编译/运行期错误 | 用 MatmulActivation |
| DeviceGemm 在 op_kernel 中 | 行为异常 | 用 Kernel{}(params) |
| CType 用 half | 精度差 | 累加用 float |
| 没读 Epilogue 头文件槽位 | 模板参数不匹配 | 打开 .hpp 确认签名 |
| computeLength 随意填 | 不正确 | 从 L0TileShape 推导 |
