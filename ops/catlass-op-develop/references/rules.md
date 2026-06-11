# Catlass Kernel 强制规则 Δ1–Δ10

> **导航**：本文档集中所有强制性规则。每条 Δ 对应一个独立约束，标注违规后果和正确做法。

---

## Δ1：仅使用 catlass 提供的实现

**要求**：op_kernel 中**只能**使用 catlass 提供的 `Kernel` / `Block*` / `Tile*` 组件。

**禁止**：
- 手写矩阵乘 / 逐元素 / 拷贝循环（AICore 标量循环极慢）
- 绕过 catlass 直接用 AscendC 向量 API 实现计算路径（自定义 Tile 除外，见 [custom-epilogue.md](./custom-epilogue.md)）

**判断标准**：`using Kernel = Catlass::Gemm::Kernel::*` 是唯一合法的 Kernel 来源。

---

## Δ2：必须使用 Device 调用

**要求**：op_kernel **只能**用 Device 调用 `Kernel{}(params)`。

**禁止**：`Gemm::Device::DeviceGemm<Kernel>` 适配器（host 侧 API）。

```cpp
// ✅ 正确
typename Kernel::Params params{...};
Kernel{}(params);

// ❌ 禁止
using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
MatmulAdapter matmulOp;
matmulOp.Initialize(arguments, deviceWorkspace);
matmulOp(stream, aicCoreNum);
```

**理由**：`DeviceGemm` 包装 host 侧 workspace 分配 / stream 调度，在算子工程中由 op_host + CANN 框架管理。

---

## Δ3：分支实例化

**要求**：任何会改变 catlass 模板参数的条件（dtype / 转置 / Swizzle / 激活变体），在 op_kernel 入口对应分支内**实例化对应的 `using Kernel = ...`** 与 `Kernel::Params`。

**要求**：每个分支实例化与 DESIGN.md §4 的合法组合表一一对应。

```cpp
// ✅ 正确：每个分支独立 using
if /* dtype=half, transA=0, transB=0 */ {
    using Kernel = NsMyOp::KernelHalfNN;
    // ... Kernel::Params + Kernel{}(params)
} else if /* dtype=half, transA=1, transB=0 */ {
    using Kernel = NsMyOp::KernelHalfTN;
    // ...
}
```

---

## Δ4：Workspace 必须用 GetUserWorkspace

**要求**：`AscendC::GetUserWorkspace(workspace)` 取 user workspace。

**禁止**：`SetSysWorkspaceForce(workspace)`。

```cpp
// ✅ 正确
GM_ADDR userWs = const_cast<GM_ADDR>(AscendC::GetUserWorkspace(workspace));

// ❌ 禁止
SetSysWorkspaceForce(workspace);
```

**注意**：必须使用带命名空间的 `AscendC::GetUserWorkspace`。

---

## Δ5：op_kernel 不得 include 自身 tiling 文件

**要求**：op_kernel 不得 `#include` 算子自身的 op_host tiling 实现文件。

**允许**：TilingData POD struct 放在 host / kernel 共享头中，op_kernel 仅 include 共享头。取 tiling 数据用通用宏（如 `GET_TILING_DATA_WITH_STRUCT`）。

**理由**：op_kernel 不应依赖 op_host 的实现细节。编译时分离编译单元各自独立。

---

## Δ6：MatmulEpilogue 独立输出缓冲

**要求**：需要 Y = A·B − X 且 `y` 与 `x3` 是不同 GM 时，须**手动**构造 `Kernel::Params`，不依赖 `ToUnderlyingArguments` 默认行为。

```cpp
typename BlockEpilogue::Params epilogueParams{ /* ..., ptrX = x3, ... */ };
typename MatmulKernel::Params params{
    problemShape, gmA, layoutA, gmB, layoutB,
    /*ptrD=*/ y, layoutD,
    userWs,
    epilogueParams
};
Kernel{}(params);
```

**禁止**：把 `__gm__` 指针 `reinterpret_cast` 成 `uint8_t *` 传给 `ToUnderlyingArguments`。

---

## Δ7：Quant Matmul 的 AIC/AIV 协同

**要求**：量化矩阵乘必须使用 `QuantMatmulMultiStageWorkspace` Kernel。

```cpp
using Kernel = Gemm::Kernel::QuantMatmulMultiStageWorkspace<
    BlockMmad, BlockEpilogue, BlockScheduler, /*workspaceStages=*/2>;
```

**差异点**：
- DispatchPolicy 必须是 `MmadAtlasA2PreloadAsyncWithCallback`
- BlockEpilogue 在 AIV 侧执行
- 多出 `gmScale`、`gmPerTokenScale` 输入
- workspace 大小由 `Kernel::GetWorkspaceSize` 计算
- `workspaceStages` 通常取 2

详见 [patterns/quant-matmul.md](./patterns/quant-matmul.md)。

---

## Δ8：Epilogue 向量长与矩阵规模

**要求**：固定模板 `COMPUTE_LENGTH` 的 `TileElemWise*` 与过小矩阵 / 尾块组合时，运行期可能 AIV UB 越界。

**约束**：
- 不宜用过小 M/N（如个位数）
- 测试 shape 宜选 L1 分块 M/N 的整数倍

详见 [shape-constraints.md](./shape-constraints.md)。

---

## Δ9：精度验证脚本对齐官方标准（影响精度判定）

**要求**：`gen_data.py` / `golden.py` / `verify_result.py` 三件套必须满足：

1. **verify 判据 = `ops-precision-standard` 选出的官方标准**（fp16/bf16 → 社区浮点标准 MERE/MARE Threshold，实践等价形式 `atol=1e-3, rtol=1e-3, error_ratio≤1e-3`）。
2. **golden 必须真生成**（不得注释 / 跳过写入）；verify 必须覆盖**基础 + 实网 shape**。
3. **golden 镜像内核数值路径**：fp32 累加、epilogue fp32 计算末尾才 cast、激活公式/常量一致、量化反量化顺序一致、输出 dtype 与形状一致。
4. **int8 GEMM golden 用 fp32 BLAS**（`|Cint|<2²⁴` 时精确），不用慢的 numpy 整数矩阵乘。

**禁止**：
- 自创**零容忍小值域门限**（如「`|golden|<2⁻¹¹ 且 abs>2⁻¹⁶` 的元素数必须为 0」）——比官方严，过零激活必误判。
- 用**全体元素 MARE-max** 作硬门限（近零 golden 让相对误差爆表）。

**自检**：内核 MERE 很小却 verify FAIL → 先查 verify 门限，而非改内核。详见 [precision-verification.md](./precision-verification.md)。

---

## Δ10：跨 N-half 门控 epilogue 的调度（影响精度，大 N 才暴露）

**要求**：epilogue 操作数**跨 N-block**时（SwiGLU=`silu(C[:, :H])·C[:, H:]`、GeGLU、ReGLU，两操作数相距 `H=N/2`），**必须**按输出形状 `[M, H]` 调度，每个输出块产出左/右两个 N-tile（列 `c` 与 `c+H`）供 epilogue 门控。

**禁止**：用「每 slot 单 `[L1.M, L1.N]` tile」的 per-block 轮转 workspace 直接做门控——`+H` 配对列落在另一 N-block，**仅 `N ≤ L1TileShape::N` 时碰巧正确，大 N 整片输出错误**（曾现 N=18432 时 ~44% 元素错、最大绝对误差 171.5，且基础 512³ 用例照过）。

**自检**：门控类算子必须在 N>L1TileShape::N 的实网 shape 上验精度。设计依据见 [catlass-op-design/references/mmad-epilogue-selection.md](../../catlass-op-design/references/mmad-epilogue-selection.md) §4.2。
