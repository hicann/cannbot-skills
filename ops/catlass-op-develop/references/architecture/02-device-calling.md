# 02: Device Calling

> **导航**：[01-kernel-assembly.md](./01-kernel-assembly.md) → 本文 → [03-compilation.md](./03-compilation.md)

catlass example 中存在两种调用模式，op_kernel **只能**使用 Device 调用。

## 两种调用模式

| 模式 | 适用位置 | 写法 | op_kernel 可用？ |
|------|---------|------|:---:|
| **Device 调用** | op_kernel | 直接 `Kernel{}(params)` | ✅ |
| **Host 调用** | example `main()` | `DeviceGemm<Kernel>` 适配器 | ❌ 禁用 |

## Device 调用（op_kernel 内）

```cpp
// op_kernel 入口分支内
GM_ADDR userWs = const_cast<GM_ADDR>(AscendC::GetUserWorkspace(workspace));
Catlass::GemmCoord problemShape{m, n, k};

typename Kernel::Params params{
    problemShape,
    gmA, Catlass::layout::RowMajor{problemShape.m(), problemShape.k()},
    gmB, Catlass::layout::RowMajor{problemShape.k(), problemShape.n()},
    gmC, Catlass::layout::RowMajor{problemShape.m(), problemShape.n()},
    userWs
};
Kernel{}(params);
```

**为什么禁用 Host 调用**：`DeviceGemm` 包装 host 侧 workspace 分配 / stream 调度，在算子工程中由 op_host tiling 与 CANN 框架管理，不应出现在 device 侧代码中。

## Workspace 获取

```cpp
// ✅ 正确
GM_ADDR userWs = const_cast<GM_ADDR>(AscendC::GetUserWorkspace(workspace));

// ❌ 禁止
SetSysWorkspaceForce(workspace);
```

## Kernel::Params 差异

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
struct Arguments {
    GemmCoord problemShape;
    uint32_t workspaceSize;    // sizeof(float) 或其他
    GM_ADDR ptrA;
    GM_ADDR ptrB;
    GM_ADDR ptrD;              // 输出（不是 ptrC）
};
```

### QuantMatmulMultiStageWorkspace::Params（增量字段）

```cpp
typename Kernel::Params params{
    problemShape,
    gmA, layoutA, gmB, layoutB,
    gmScale, layoutScale,                              // ★ 量化 scale
    gmPerTokenScale, layoutPerTokenScale,               // ★ per-token scale
    gmD, layoutD,
    userWs
};
```

## 完整调用骨架（带分支）

```cpp
// op_kernel 入口
auto key = GET_TILING_KEY();  // 从 tiling 获取

if /* 分支条件1 */ {
    using Kernel = NsMyOp::KernelVariant1;
    GM_ADDR userWs = const_cast<GM_ADDR>(AscendC::GetUserWorkspace(workspace));
    Catlass::GemmCoord problemShape{m, n, k};
    typename Kernel::Params params{problemShape, gmA, layoutA, gmB, layoutB,
                                   gmC, layoutC, userWs};
    Kernel{}(params);
} else if /* 分支条件2 */ {
    using Kernel = NsMyOp::KernelVariant2;
    // ... 同上但用不同的 Params
}
```

**注意**：
- 每个 `if` 块内独立 `AscendC::GetUserWorkspace` —— 不能提到块外
- `GemmCoord` 是 int 类型，直接传 m/n/k
- 分支条件怎么写由调用方决定，本 skill 只要求每个分支内正确实例化

## 强制规则

| 规则 | 说明 |
|------|------|
| Δ2 | op_kernel 只能用 Device 调用 `Kernel{}(params)` |
| Δ4 | 必须用 `AscendC::GetUserWorkspace(workspace)` |
| Δ6 | MatmulEpilogue 需独立 X/D 时手写 Params |
| Δ7 | Quant Matmul 走 `QuantMatmulMultiStageWorkspace` |

详见 [rules.md](../rules.md)。
