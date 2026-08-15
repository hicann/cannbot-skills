# Attention 子场景：GDN / KDA

> 本文件只覆盖 GDN / gated delta rule / KDA / KDA dAv backward stage 的专项设计规则。先由 [linear-attention.md](linear-attention.md) 完成 Attention 大类路由、primary reference 判定和通用 stage 设计，再读取本文件。

---

## KDA `dAv` Backward Stage 快速判定

当需求包含 `chunk_kda_bwd_dAv`、`catlass_chunk_kda_bwd_dav` 或 KDA backward 的 `dA/dV` 输出时，按 stage operator 设计，不把它误判为完整 backward：

| 项 | 设计要求 |
|---|---|
| 同语义 baseline | 仅用于评测口径、shape 和报告字段；优先读取 FLA Triton `chunk_bwd.py` 与 `chunk_kda_bwd_kernel_dAv_npu` backend，再对照 NPU AscendC KDA 公开/curated 经验；实现 primary reference 仍按 `reference_source` 判定 |
| 主要计算 | `dA = dO @ V^T` 后按 causal/gate/scale 处理；`dV = A^T @ dO`，其中 `A` 可来自前向/中间量契约 |
| stage 契约 | 明确 `A`、`dO`、`V`、`cu_seqlens`/`chunk_indices`、GVA `HV/HK` 映射、输出 `dA/dV` 的 layout |
| varlen/partial | `nt` 与 task 数来自真实 chunk 索引；设计区分“物理 chunk 计算尺寸”和“有效行数写回/掩码” |
| shape 覆盖 | 按 develop 侧 [shape-constraints.md Δ5](../../../../catlass-op-develop/references/shape-constraints.md#Δ5linear-attention--gdn-类-shape-覆盖按算法维度生成) 覆盖 noGVA/GVA、`V=128/256`、fixed/varlen、`BT=64/128`、小/大 `B*chunk`、GQA/GVA 和数值边界 |
| 精度标准 | 使用 `ops-precision-standard` mixed tolerance，报告 `case_name`、shape、dtype、输出名、atol、rtol、matched_ratio、max_abs、pass/fail，不自创小值域规则 |
| 性能报告 | 同一 report 中记录 custom、Triton/开源 baseline、baseline unsupported/MISSING 状态，避免把不可运行 baseline 当 FAIL |

设计阶段不要承诺通过 `actualShape=validRows` 一类“尾块收敛”优化提速。对当前 Catlass TLA 路径，KDA dAv varlen 长 case 已出现该做法触发分钟级长跑的经验，应在实现/性能阶段单独 gated 和 profile。

---

## GDN/KDA Shape 覆盖

GDN/KDA 不维护单独 case 数据文件，统一使用 Linear Attention shape 覆盖规则。最小 representative 子集必须至少 8 例，并覆盖：

| 维度 | 最小覆盖 |
|---|---|
| noGVA/GVA | noGVA 至少 1 例，GVA 至少 1 例 |
| `V=128/256` | 两类都至少 1 例；`V=256` 覆盖 split accumulation |
| 定长/varlen | 定长至少 1 例，varlen 至少 1 例 |
| `BT=64/128` | 两类都至少 1 例 |
| `B*chunk` | 小规模和大规模各至少 1 例 |

若少于 8 例或缺少任一覆盖维度，不得声称为 representative 子集；只能标记为 smoke。

---

## GDN/KDA 报告口径

- 精度报告按 mixed tolerance 口径输出：`case_name`、shape、dtype、输出名、atol、rtol、matched_ratio、max_abs、pass/fail。
- Triton/开源 baseline 只作为 evaluation baseline 辅助记录；某些 shape 不支持时，应记录 `baseline_status=UNSUPPORTED|MISSING|FAIL`，不把 baseline 不可运行等同于 custom FAIL。
- smoke case 只用于环境和基本功能门禁，不代表完整 shape 覆盖。
- 性能报告必须区分 custom Task Duration、baseline Task Duration、launch count、workspace peak 和 baseline_status。
