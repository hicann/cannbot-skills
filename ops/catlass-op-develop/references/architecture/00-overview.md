# 00: Kernel 组装全景

> **导航**：本文件是 `references/architecture/` 系列的第 0 篇，给出来 catlass Kernel 代码生成的完整视图。实现时按 [01-kernel-assembly.md](./01-kernel-assembly.md) → [02-device-calling.md](./02-device-calling.md) → [03-compilation.md](./03-compilation.md) 顺序使用。

## Kernel = BlockMmad + BlockEpilogue + BlockScheduler

catlass Kernel 由三个 Block 组件拼装而成：

```
Kernel<BlockMmad, BlockEpilogue, BlockScheduler>
   │
   ├── BlockMmad (矩阵乘主循环)
   │   ├── DispatchPolicy   — 调度策略（Pingpong / Preload / PreloadAsync）
   │   ├── L1TileShape      — GemmShape<m1, n1, k1>
   │   ├── L0TileShape      — GemmShape<m0, n0, k0>
   │   ├── AType, BType     — 输入 GemmType<dtype, Layout>
   │   └── CType            — 累加 GemmType<float, Layout>
   │
   ├── BlockEpilogue (后处理) = void 或 BlockEpilogue<Policy, CType, DType, Tile1, Tile2, ...>
   │
   └── BlockScheduler (多核调度)
       └── GemmIdentityBlockSwizzle<Offset, Direction>
```

## 代码生成流程

```
DESIGN.md 选型表
      │
      ▼
op_kernel 内 using 链
      │
      ├── 1. ArchTag: Catlass::Arch::AtlasA2（或 Ascend950）
      ├── 2. DispatchPolicy: MmadAtlasA2Pingpong<true>（或 Preload/...）
      ├── 3. TileShape: L1TileShape<128,256,256>, L0TileShape<128,256,64>
      ├── 4. GemmType: AType, BType, CType（必要时 + DType）
      ├── 5. BlockMmad = Block::BlockMmad<DispatchPolicy, ...>
      ├── 6. BlockEpilogue = void 或 Epilogue::Block::BlockEpilogue<...>
      ├── 7. BlockScheduler = GemmIdentityBlockSwizzle<3, 0>
      ├── 8. Kernel = <KernelType><BlockMmad, BlockEpilogue, BlockScheduler>
      │
      ▼
op_kernel 入口分支内 Device 调用
      │
      ├── 9. AscendC::GetUserWorkspace(workspace) → userWs
      ├── 10. GemmCoord{m, n, k} → problemShape
      ├── 11. Kernel::Params params{...} → 构造参数
      └── 12. Kernel{}(params) → 执行
```

## 两种典型的 using 链

### 无 Epilogue（纯 matmul）

```
ArchTag → DispatchPolicy → L1/L0TileShape → AType/BType/CType
    → BlockMmad (void Epilogue) → BlockScheduler → BasicMatmul Kernel
```

详见 [patterns/basic-matmul.md](../patterns/basic-matmul.md)。

### 有 Epilogue（激活/偏置/反量化）

```
ArchTag → DispatchPolicy → L1/L0TileShape → AType/BType/CType
    → BlockMmad
    → EpiloguePolicy → CType/DType → Tile1 → Tile2 → ... → BlockEpilogue
    → BlockScheduler → MatmulActivation / QuantMatmul 等 Kernel
```

详见 [patterns/with-epilogue.md](../patterns/with-epilogue.md)。

## 与前序设计的关联

| 设计文档产出 | 代码生成对应 |
|-------------|-------------|
| DESIGN.md §2.2 BlockMmad 选型表 | using 链 step 1-5 |
| DESIGN.md §2.3 BlockEpilogue 槽位清单 | using 链 step 6 |
| DESIGN.md §2.4 BlockScheduler 选型 | using 链 step 7 |
| DESIGN.md §2.5 Kernel 选型 | using 链 step 8 |
| DESIGN.md §3 Kernel 适配方案 | Device 调用 step 9-12 |
| DESIGN.md §4 分支实例化条件 | 分支骨架（详见 [patterns/branch-instantiation.md](../patterns/branch-instantiation.md)） |
| DESIGN.md §5 Workspace 来源 | userWs 计算 |
| DESIGN.md §6 自定义 Tile 契约 | [custom-epilogue.md](../custom-epilogue.md) |

## 强制规则速查

所有强制性规则集中在 [rules.md](../rules.md)，其中最重要的关卡规则：

| 规则 | 含义 |
|------|------|
| Δ1 | 仅用 catlass Kernel / Block* / Tile*，禁手写矩阵乘/逐元素/拷贝循环 |
| Δ2 | 必须用 Device 调用 `Kernel{}(params)`，禁用 `DeviceGemm` |
| Δ3 | 每个分支正确实例化 `using Kernel = ...` + `Kernel::Params` |
| Δ4 | `AscendC::GetUserWorkspace(workspace)`，禁 `SetSysWorkspaceForce` |
