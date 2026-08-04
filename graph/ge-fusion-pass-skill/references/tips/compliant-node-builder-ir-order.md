# tip: 显式手建已注册节点，IR 输入/属性严格按 op_proto REG_OP 顺序

> 📎 导航落点：`references/fusion-troubleshooting.md` §5（replacement 是否成功）、`interface-catalog.md` §三。本文件仍是 IR 顺序硬性做法的**唯一权威定义源**，正文未改动。迁移映射见 `tips/MIGRATION.md`。

**读者**：② 开发 / ③ 分析。

## 症状

用 Graph API / `CompliantNodeBuilder` 显式落某个 op type 后，ATC 报 `attribute order has changed` / `Failed to recover ir definitions`。

## 根因

显式手建已注册节点时，IR 输入与属性的**顺序/必填性**必须严格匹配该 op 的 op_proto `REG_OP` 定义。顺序错了，GE 无法把手建节点映射回已注册的 IR 定义。

## 何时才走这条路

**仅当 `es_all` 确实未暴露你需要的 wrapper**（决策树见 `es-all-no-version-rename.md`）。若 es_all 已有 wrapper（如 `es::GEMM`），**禁止**为"换版本名/少建常量节点"绕开它去自行构建。

## 硬性做法

用 `ge::es::CompliantNodeBuilder` 显式 `OpType("<op type>")`，`IrDefInputsV2/IrDefOutputsV2/IrDefAttrsV2` **严格按 op_proto `REG_OP` 顺序**填。有序配方以 op_proto `REG_OP`（CANN OPP / GE 文档中该算子定义）为准；见到报错**不要猜属性顺序**，回到 `REG_OP` 定义按序补齐。

Conv2D 标准骨架（配方序 strides→pads→dilations→groups→data_format→offset_x）：

```cpp
#include "compliant_node_builder.h"
#include "es_c_graph_builder.h"
using CNB = ge::es::CompliantNodeBuilder;
auto &builder = x->GetOwnerBuilder();           // x: 已在图上的输入 TensorHolder
auto ge_graph = builder.GetGraph();
auto node = CNB(ge_graph).OpType("Conv2D")       // ← 显式落已注册 op type，而非 es::Conv2DV2
    .Name(builder.GenerateNodeName("Conv2D").GetString())
    .IrDefInputsV2({{"x", CNB::kEsIrInputRequired, ""}, {"filter", CNB::kEsIrInputRequired, ""},
                    {"bias", CNB::kEsIrInputOptional, ""}, {"offset_w", CNB::kEsIrInputOptional, ""}})
    .IrDefOutputsV2({{"y", CNB::kEsIrOutputRequired, ""}})
    .IrDefAttrsV2({                              // 顺序/必填性严格照配方
        {"strides",     CNB::kEsAttrRequired, "ListInt", ge::es::CreateFrom(strides)},   // std::vector<int64_t>
        {"pads",        CNB::kEsAttrRequired, "ListInt", ge::es::CreateFrom(pads)},
        {"dilations",   CNB::kEsAttrOptional, "ListInt", ge::es::CreateFrom(dilations)},
        {"groups",      CNB::kEsAttrOptional, "Int",     ge::es::CreateFrom(static_cast<int64_t>(groups))},
        {"data_format", CNB::kEsAttrOptional, "String",  ge::es::CreateFrom(ge::AscendString("NCHW"))},
        {"offset_x",    CNB::kEsAttrOptional, "Int",     ge::es::CreateFrom(static_cast<int64_t>(0))},
    }).Build();
ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, x->GetProducer(), x->GetOutIndex(), node, 0);       // 逐输入连边
ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, filter->GetProducer(), filter->GetOutIndex(), node, 1);
// bias→2、offset_w→3 仅在非空时连；按需对输入 SetFormat/UpdateInputDesc 设 NCHW（见 format-sensitive-nchw.md）
auto y = builder.GetTensorHolderFromNode(std::move(node), 0);   // 取回输出 TensorHolder
```

此骨架已内嵌于本 tip，直接可用；需要更多参考时，`examples/` 下的样例实现源码也可以读。

## 自查

- IrDefInputs/IrDefAttrs 的顺序是否逐项对齐了该 op 的 `REG_OP` 定义？
- 是不是在 es_all 已有 wrapper 的情况下才误走了这条显式路径？若是，退回用 wrapper。
- format-sensitive 算子（Conv/Pool）是否还按 `format-sensitive-nchw.md` 设了 NCHW？
