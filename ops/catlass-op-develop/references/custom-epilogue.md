# 自定义 Tile Epilogue 实现骨架

> **前端依赖**：`catlass-op-design` 的 `references/custom-epilogue.md`（设计阶段已确认无现成 Tile 并写出契约）
> **本文件聚焦**：在设计契约基础上落盘自定义 Tile 头文件，并拼装进 BlockEpilogue。

## 前置：确认粒度

由 `catlass-op-design` 设计阶段确定粒度（A：替换现有槽位 / B：重写 BlockEpilogue）。**本 skill 只处理粒度 A**；粒度 B 的完整 BlockEpilogue 重写不在此范围。

---

## 1. 自定义 Tile 骨架（固定写法）

```cpp
#pragma once
#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Tile {

template <class ArchTag_, class ComputeType_, uint32_t COMPUTE_LENGTH_>
struct TileMyCustom {
    using ArchTag        = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;
    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    CATLASS_DEVICE void operator()(
        AscendC::LocalTensor<ElementCompute> const &ubOut,
        AscendC::LocalTensor<ElementCompute> const &ubIn0,
        AscendC::LocalTensor<ElementCompute> const &ubIn1)
    {
        // 仅用 AscendC 向量 API：Add / Mul / Muls / Div / Exp / ...
        // 与 COMPUTE_LENGTH 对齐
    }
};

} // namespace Catlass::Epilogue::Tile
```

**关键约定**：

| 项 | 约定 |
|----|------|
| 模板形参 | `<ArchTag_, ComputeType_, COMPUTE_LENGTH_>` — 与 catlass 内置 Tile 一致 |
| 必要 typedef | `ArchTag`, `ElementCompute`, `COMPUTE_LENGTH` — BlockEpilogue `static_assert` 会检查 |
| `operator()` 签名 | 必须与目标槽位的接口签名对齐（参数量和类型） |
| 命名空间 | `Catlass::Epilogue::Tile` — 与 catlass 保持一致 |
| Guard | `#pragma once` |

**NoSource policy 的 Tile** 若签名不同，对照 catlass 同 policy 的现有 Tile（如 `epilogue/tile/tile_elemwise_*.hpp`）。

---

## 2. 在 catlass 拼装中引用

```cpp
#include "/* 调用方决定的自定义 Tile 头路径 */"

using TileElemWiseEpilogue = Catlass::Epilogue::Tile::TileMyCustom<
    ArchTag, ComputeType, /*COMPUTE_LENGTH=*/computeLength>;

using BlockEpilogue = Catlass::Epilogue::Block::BlockEpilogue<
    EpilogueDispatchPolicy, CType, DType, TileCopy, TileElemWiseEpilogue>;
```

**要点**：
- `using` 与设计文档中的契约（DispatchPolicy 类别、computeLength、组装顺序）一一对应
- 自定义 Tile 替换对应槽位的 `using`，其余槽位不变
- 头文件**具体放在哪**由工程模板决定（如 `op_kernel/custom_epilogue/`），本 skill 不规定具体路径

---

## 3. op_kernel 入口分支

入口分支按 [rules.md](./rules.md) 的 Δ2 / Δ4 写法实例化 `Kernel` + `Kernel::Params`，调用 `Kernel{}(params)`。

---

## 4. 检查清单

- [ ] 自定义头文件的 `template <ArchTag_, ComputeType_, COMPUTE_LENGTH_>` 与 `operator()` 签名与设计契约一致
- [ ] `operator()` 内仅用 AscendC 向量 API
- [ ] 未修改 catlass 上游源码
- [ ] catlass 拼装类中 `using TileXxx = TileMyCustom<...>;` 并组装进 `BlockEpilogue` / `MatmulEpilogue`
- [ ] op_kernel 未使用 `DeviceGemm` 适配器
