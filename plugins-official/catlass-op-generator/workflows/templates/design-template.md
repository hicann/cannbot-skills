# {operator_name} Catlass 算子设计文档模板

> ⚠️ **Architect 生成此文档时必须替换以下占位符**：
> - `{operator_name}` → 实际算子名称（snake_case，**必须**含 `catlass` 子串，如 `catlass_matmul_add`）
> - `{OperatorName}` → CamelCase 类名（如 `CatlassMatmulAdd`）
> - `{arch}` → catlass 架构号（如 220 / 100）

---

## 0. 概述

### 0.0 catlass 命名校验

| 项目 | 内容 |
|-----|------|
| `operator_name` | {operator_name} |
| 含 `catlass` 子串 | ✅ / ❌（不通过禁止继续） |
| 一致映射 CamelCase | {OperatorName} |

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | {operator_name} |
| 算子类别 | GEMM / Matmul + Epilogue / Quant Matmul / 其他 catlass 表达的融合算子 |
| 需求类型 | 特定用例（M/N/K=..., dtype=...） / 通用 |
| 支持数据类型 | A/B/C 各自的 dtype |
| 目标 SoC | Atlas A2 / A3 / A5 |
| 是否量化 | 是 / 否 |
| 转置约定 | A: {NN/TN/NT/TT} |
| 特殊约束 | M/N/K 上下界、对齐、是否带 bias 等 |

---

## 1. 算子设计

### 1.1 数学公式

```
// 输入输出定义
A: shape=[M, K], dtype=...
B: shape=[K, N], dtype=...
C: shape=[M, N], dtype=...

// 数学公式
C = epilogue(A @ B, ...)
```

### 1.2 Catlass 组件选型表

> ⚠️ 所有 `using` 必须打开 `catlass/include/catlass/*` 对应 header 验证形参后再写入，禁止凭印象。

| 组件 | 选型 | 头文件路径 | 选型理由 |
|------|------|-----------|---------|
| ArchTag | `Catlass::Arch::AtlasA2` | `catlass/include/catlass/arch/arch.hpp` | 目标 SoC 为 A2 |
| DispatchPolicy | `Catlass::Gemm::MmadAtlasA2Pingpong<true>` | `catlass/include/catlass/gemm/dispatch_policy.hpp` | A/B 均 Pingpong，最大化流水重叠 |
| L1TileShape | `Catlass::GemmShape<128, 256, 256>` | — | M=128 K=256 是 A2 上典型最优 |
| L0TileShape | `Catlass::GemmShape<128, 256, 64>` | — | 与 L1 配套 |
| AType | `Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor>` | `catlass/include/catlass/gemm/gemm_type.hpp` | A 为 fp16 行主序 |
| BType | `Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor>` | 同上 | B 为 fp16 行主序 |
| CType | `Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>` | 同上 | C 累加为 fp32 |
| BlockMmad | `Catlass::Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>` | `catlass/include/catlass/gemm/block/block_mmad.hpp` | 上述参数组合 |
| BlockEpilogue | `void`（纯 GEMM）/ `Catlass::Epilogue::Block::BlockEpilogue{Policy}<...>` | `catlass/include/catlass/epilogue/block/block_epilogue_*.hpp` | 算子需要的后处理 |
| BlockScheduler | `Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>` | `catlass/include/catlass/gemm/block/gemm_block_swizzle.hpp` | 3D Swizzle 缓解 L2 cache 冲突 |
| Kernel | `Catlass::Gemm::Kernel::BasicMatmulKernel<BlockMmad, BlockEpilogue, BlockScheduler>` | `catlass/include/catlass/gemm/kernel/*` | 与 BlockMmad/Scheduler 匹配 |

### 1.3 参考 Example

| 项目 | 内容 |
|-----|------|
| Example 路径 | `catlass/examples/00_basic_matmul/00_basic_matmul.cpp` |
| 选型理由 | 与本算子形态最接近（GEMM + 可选 Epilogue） |
| 抄结构范围 | 仅抄 catlass 拼装类 + Device 调用结构，**不抄** main()（因 main 仅 example 用） |

### 1.4 Kernel 适配方案

| Example main() 阶段 | 本算子归属 | 实现位置 |
|---------------------|-----------|---------|
| ACL 初始化 / device 内存分配 | host | `op_host/{operator_name}.asc` |
| Tiling 计算（输出 TilingData + workspaceSize + tilingKey） | host | `op_host/{operator_name}.asc` 中的 Tiling 函数 |
| `<<<usedNumBlocks, ...>>>` Kernel 启动 | host main | `op_host/{operator_name}.asc` |
| catlass 拼装 `using` + `Kernel{}(params)` | device | `op_kernel/{operator_name}.asc` |
| `DeviceGemm` 适配器（example 用） | **去除** | 本算子直接 `Kernel{}(params)` |
| 结果搬回 / verify | host | `op_host/{operator_name}.asc` |

### 1.5 BlockEpilogue 槽位清单（如 BlockEpilogue ≠ void）

> 打开 `catlass/include/catlass/epilogue/block/block_epilogue_<policy>.hpp` 读出每个槽位的模板形参与签名后填写。

| 槽位 | 形参签名（来自 header） | 来源 | 现成 / 粒度 A 自定义 / 粒度 B 新特化 |
|------|----------------------|------|---------------------------------|
| TileEpilogue | `template <class CType, class DType, ...>` | `block_epilogue_<policy>.hpp` 第 N 行 | 现成 |
| TileBroadcastOneBlk | `template <class XType, ...>` | 同上 | 现成 |
| TileMul | `template <class XType, class YType, ...>` | 同上 | 自定义（见 §1.6） |

### 1.6 自定义 Tile 契约（如有）

> 按 `/catlass-op-design` references/custom-epilogue.md 写头文件骨架与契约。

```cpp
// op_kernel/tiles/{tile_name}.hpp
template <class XType, class YType, class ZType /* 必须与槽位期望严格一致 */>
struct {TileName} {
    static constexpr DispatchCategory DISPATCH_CATEGORY = DispatchCategory::{...};

    CATLASS_DEVICE void operator()(
        XType const& x, YType const& y, ZType& z, /* ... */) {
        // 契约：操作粒度 = ...
        // 契约：UB 占用 = ...
    }
};
```

| 契约项 | 内容 |
|-------|------|
| DispatchPolicy 类别 | `DispatchCategory::{...}` |
| 操作粒度 | per-Tile / per-Block / per-Element |
| UB 占用 | 多少字节 |
| 同步要求 | 是否需要 SyncBlock |

---

## 2. 架构设计

### 2.1 TilingKey 分支条件与合法组合

| 分支 | dtype 组合 | 转置组合 | Swizzle | DispatchPolicy | Kernel 实例化 |
|------|-----------|---------|---------|----------------|--------------|
| 0 | A=fp16, B=fp16, C=fp32 | NN | Identity<3,0> | Pingpong<true> | `BasicMatmulKernel<...>` |
| 1 | A=bf16, B=bf16, C=fp32 | NN | Identity<3,0> | Pingpong<true> | `BasicMatmulKernel<...>` |
| 2 | A=fp16, B=fp16, C=fp16 | TN | Identity<3,0> | Preload<true> | `PreloadMatmulKernel<...>` |
| ... | ... | ... | ... | ... | ... |

### 2.2 Workspace 量级

| 项目 | 内容 |
|-----|------|
| 来源 API | `AscendC::GetUserWorkspace(workspace)` |
| 计算依据 | catlass `Kernel::GetWorkspaceSize(...)` 或 example 推导公式 |
| 量级 | 约 N KB / MB |
| host 端处理 | host Tiling 时累加 workspaceSize 返回给框架 |

### 2.3 实现约束

| 约束 | 说明 |
|------|------|
| C3 编译选项 | CMakeLists.txt 必须注入 `-I<catlass>/include` + `-DCATLASS_ARCH={arch}` |
| C4 Device 调用 | 直接 `Kernel{}(params)`；**禁用** `DeviceGemm` |
| C5 自实现禁项 | op_kernel 不得自实现矩阵乘 / 逐元素 / 拷贝循环 |
| C6 Workspace | 必须 `AscendC::GetUserWorkspace(workspace)`；**禁用** `SetSysWorkspaceForce` |
| C7 头文件边界 | op_kernel **禁** include 算子自身的 tiling 实现文件 |

---

## 3. 确认清单

- [ ] `{operator_name}` 含 `catlass` 子串，CamelCase 一致映射
- [ ] catlass 选型表（§1.2）每个 `using` 已打开对应 header 验证
- [ ] 参考 example 路径已锁定（§1.3）
- [ ] Kernel 适配方案已说明（§1.4）
- [ ] BlockEpilogue 槽位清单（§1.5）已与 header 对齐（如适用）
- [ ] 自定义 Tile 契约（§1.6）已写出（如适用）
- [ ] TilingKey 分支条件 / 合法组合已穷举（§2.1）
- [ ] Workspace 量级与计算依据已记录（§2.2）
- [ ] §2.3 实现约束已声明
