# 自定义 Tile Epilogue（设计阶段）

**侧重点**：在定稿组件选型前，判断 catlass 仓库是否已有所需 Tile / 流水线；若无，先在设计文档中**定义**自定义 Tile 的数学与接口契约。具体头文件落盘、kernel 引用方式由 `catlass-op-develop` 负责。

---

## 0. 决策树（先选粒度）

> **强制前置**：在写"自定义 Tile"前，先按 [SKILL.md Step 2.5](../SKILL.md#step-25blockepilogue-槽位清单关键前置) 列出 BlockEpilogue 特化的**槽位清单**。再据此决定走粒度 A 还是粒度 B。

```
所选 BlockEpilogue 特化的 Tile 槽中，是否有一个槽
   接口签名能严格容纳所需运算？
   │
   ├── 是  → ✅ 粒度 A（推荐）
   │         在该槽内写自定义 Tile，复用现有 BlockEpilogue
   │         主体（不复制 700+ 行 Block 代码）
   │
   └── 否  → ❌ 粒度 B（重）
             写新 BlockEpilogue 特化（DispatchPolicy + 完整 Tile 列表）
             代码量 500–700 行，需仔细评估是否值得
```

| 维度 | 粒度 A：替换 Tile 槽 | 粒度 B：新 BlockEpilogue 特化 |
|------|---------------------|----------------------------|
| 触发条件 | 现有 BlockEpilogue 特化中**至少一个槽**的接口签名能容纳所需运算 | 所有现有特化都不匹配（运算位置 / Tile 数量根本不对） |
| 工作量 | 一个 Tile 头文件（~80 行） | 一份 BlockEpilogue 特化（~500–700 行） + 可选新 DispatchPolicy 类 |
| 维护风险 | 低——只跟 Tile 接口耦合 | 高——跟 catlass BlockEpilogue 主循环全部代码耦合，catlass 升级需同步 |
| 性能影响 | 与原 Tile 等价（在同一 UB / 流水节点上做计算） | 自由——可以完全重排 Tile 调度顺序与 UB 复用 |
| 典型案例 | dequant + GELU 融合（GELU 折进 `TileOneBlkColumnBroadcastMul` 槽） | 新流水形态（如 attn 类，需要不同 UB 调度模式） |

**强烈建议优先粒度 A**——大多数算子融合都能用粒度 A 实现。粒度 B 仅在数学上**无法**与任何现有 Tile 槽接口对齐时才考虑。

---

## 1. 何时走「自定义 Tile」

1. 查 [epilogue-components.md](./epilogue-components.md) 与 `catlass/include/catlass/epilogue/tile/`、`catlass/examples/`（如 `03_matmul_add`）。
2. **已有**与需求一致的 Tile（如 `TileElemWiseAdd`）→ **不**自定义；选型表中写清用哪个**现成**符号即可。
3. **没有**现成 Tile（新逐元素公式、新融合方式）→ 进入本节先设计、后实现。

---

## 2. 粒度 A 契约（推荐，详见 §0 决策树）

在「Catlass 组件选型 → BlockEpilogue → 槽位清单」中**单列一小节**，至少包含：

| 项 | 说明 |
|----|------|
| **目标槽位** | 替换现有 BlockEpilogue 特化的哪个 Tile 槽（如 `TileOneBlkColumnBroadcastMul_`） |
| **目标槽位的接口签名** | 直接抄 catlass 头文件中的模板形参列表与 `operator()` 入参（如 `template <ArchTag_, ComputeType_, TileShape_>`，`operator()(ubOut, ubIn0, ubIn1)`），自定义 Tile 必须**严格对齐** |
| **必要 typedef / 常量** | BlockEpilogue 特化会用 `static_assert` 检查的成员（如 `TileShape`、`COMPUTE_LENGTH`、`ElementCompute`） |
| **Tile 名称** | 如 `TileOneBlkColumnBroadcastMulGelu`，与 catlass 内置不重名 |
| **数学 / 行为差异** | 相对原槽 Tile 的差异——例如「在原 mul 之后追加 gelu」 |
| **computeLength 量级** | 与原槽 Tile 一致（一般 = `TileShape::COUNT`） |
| **UB 占用变化** | 是否需要额外临时 buffer，能否 in-place 复用 ubOut |

设计阶段**不写**可编译 C++，只写清上表。

### 粒度 A 示例（来自实际算子）

> 算子：`catlass_quant_matmul_gelu`——把 GELU 接到 W8A8 + per-token + per-channel dequant 流水末尾。

| 项 | 内容 |
|----|------|
| 目标槽位 | `BlockEpilogue<EpilogueAtlasA2PerTokenDequant>` 的第 3 个槽 `TileOneBlkColumnBroadcastMul_` |
| 接口签名 | `template <class ArchTag_, class ComputeType_, class TileShape_> struct TileXxx; void operator()(ubOut, ubIn0, ubIn1);` |
| 必要 typedef | `using ArchTag = ArchTag_; using ElementCompute = typename ComputeType_::Element; using TileShape = TileShape_;` |
| 名称 | `TileOneBlkColumnBroadcastMulGelu` |
| 数学 | 原槽：`ubOut = ubIn0 ⊙ broadcast_col(ubIn1)`<br>新行为：`tmp = ubIn0 ⊙ broadcast_col(ubIn1); ubOut = gelu(tmp)` |
| computeLength | `TileShape::COUNT`（与原槽一致） |
| UB 占用变化 | 0（in-place 复用 ubOut，先 mul 再 gelu in-place） |

---

## 3. 粒度 B 契约（重，仅在粒度 A 不可行时）

至少包含：

| 项 | 说明 |
|----|------|
| **新 DispatchPolicy 类型** | 如 `EpilogueAtlasA2MyFusion<UB_STAGES_>`，与现有 dispatch_policy.hpp 中类风格一致 |
| **完整 Tile 槽序列** | 列出新 BlockEpilogue 特化的所有模板形参 + 每个 Tile 的接口签名 |
| **主循环伪代码** | 用伪代码描述每个 Tile 的调用顺序、UB 流水、CrossCore 同步点 |
| **CrossCoreFlag 用法** | 如涉及 AIC/AIV 协同，说明 flag 数量与 wait/set 顺序 |
| **代码量预估** | 约 500–700 行（参照 `block_epilogue_per_token_dequant.hpp`） |
| **与 catlass 升级的耦合** | 说明哪些 catlass 内部 API 被依赖（一旦 catlass 升级可能需要同步改） |

**慎选粒度 B**——大多数情况下都能改用粒度 A 解决。

---

## 4. 与下游实现的衔接

`catlass-op-develop` 按上述契约实现自定义 Tile / BlockEpilogue。**接口签名一致**就能直接拼装到原 BlockEpilogue 特化（粒度 A）或独立编译（粒度 B）。

---

## 5. 检查清单（设计）

### 通用

- [ ] 已检索 catlass `epilogue/tile`、`epilogue/block`、[epilogue-components.md](./epilogue-components.md)，确认「无现成组件」
- [ ] 已按 [SKILL.md Step 2.5](../SKILL.md#step-25blockepilogue-槽位清单关键前置) 列出 BlockEpilogue 特化的槽位清单
- [ ] 已显式标注粒度（A / B）

### 粒度 A 特有

- [ ] 目标槽位的接口签名（模板形参 / `operator()` 入参 / 必要 typedef）已**逐项**对齐
- [ ] 数学行为差异、computeLength、UB 占用变化已写明

### 粒度 B 特有

- [ ] 新 DispatchPolicy 类型 / 完整 Tile 槽序列 / 主循环伪代码 / CrossCoreFlag 用法 已写明
- [ ] 已评估代码量与维护成本，确认粒度 A 不可行
