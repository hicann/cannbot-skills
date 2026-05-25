# Branch Instantiation — 多分支实例化

> **导航**：[architecture/02-device-calling.md](../architecture/02-device-calling.md) 的"完整调用骨架"展开

任何会改变 catlass 模板参数的条件（dtype / 转置 / Swizzle / 激活变体）都要在 op_kernel 入口对应分支内实例化对应的 `using Kernel = ...` 与 `Kernel::Params`。

## 分支条件来源

分支条件的取值集合来自 DESIGN.md §4 的合法组合表：

| 条件 | 影响 catlass 模板 |
|------|-------------------|
| 输入 dtype | AType, BType, CType |
| transA | LayoutA = RowMajor 或 ColumnMajor |
| transB | LayoutB = RowMajor 或 ColumnMajor |
| Swizzle 方向 | BlockScheduler `<3,0>` 或 `<3,1>` |
| 激活类型 | BlockEpilogue 的 Tile 选择 |

## 分支骨架

```cpp
// op_kernel 入口
auto tilingKey = /* 从 tiling 获取字段 */;

// 分支 1: half, transA=0, transB=0
if /* 条件 */ {
    using Kernel = NsMyOp::CatlassMatmul<half, false, false>;

    GM_ADDR userWs = const_cast<GM_ADDR>(AscendC::GetUserWorkspace(workspace));
    Catlass::GemmCoord problemShape{m, n, k};
    Catlass::layout::RowMajor layoutA{m, k};
    Catlass::layout::RowMajor layoutB{k, n};
    Catlass::layout::RowMajor layoutC{m, n};

    typename Kernel::Params params{problemShape,
                                   gmA, layoutA, gmB, layoutB,
                                   gmC, layoutC, userWs};
    Kernel{}(params);
}
// 分支 2: half, transA=1, transB=0
else if /* 条件 */ {
    using Kernel = NsMyOp::CatlassMatmul<half, true, false>;
    // Layout 换成 ColumnMajor
    Catlass::layout::ColumnMajor layoutA{m, k};
    // ... 同上
}
// ... 更多分支
```

## 命名空间组织

每个分支的 Kernel 类型放在独立的命名空间或模板特化中：

```cpp
namespace NsMyOp {
    // half, NN
    using KernelHalfNN = Catlass::Gemm::Kernel::BasicMatmul<
        BlockMmadHalfNN, BlockEpilogue, BlockSchedulerNN>;

    // half, TN
    using KernelHalfTN = Catlass::Gemm::Kernel::BasicMatmul<
        BlockMmadHalfTN, BlockEpilogue, BlockSchedulerTN>;
}
```

**注意**：命名约定（KernelHalfNN / KernelHalfTN 等）仅为示例。实际命名由工程模板和开发习惯决定。

## 最多实例化数量

典型 combo 数量：

| dtype 数 | 转置组合 | Swizzle 方向 | 激活类型 | 最大实例化数 |
|:---:|:---:|:---:|:---:|:---:|
| 2 | 4 (NN, NT, TN, TT) | 2 | 1 | 16 |
| 1 | 1 | 1 | 1 | 1 |

**实际只实例化"合法组合"**——笛卡尔积中满足约束的子集。约简后的组合远少于全排列。

## 强制规则

| 规则 | 说明 |
|------|------|
| Δ3 | 每个分支内独立 `using Kernel = ...` + `Kernel::Params` |
| Δ3 补充 | 每个分支与 DESIGN.md 合法组合表一一对应 |
| Δ5 | op_kernel 不得 include 自身 tiling 文件 |
| Δ4 | 每个分支内独立 `AscendC::GetUserWorkspace` |

详见 [rules.md](../rules.md)。
