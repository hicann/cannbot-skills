---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "重构/升级防 regression 方法论(3-way oracle)"
description: "重构/升级后防止 regression 的方法论 位置: src/skills/aog-regression-check/references/methodology.md。由 /aog-regression-check skill 条件性载入——日常 port 流程不读本文档，只有在检测/诊断 regression 时才进入上下文。此设计（user 2026-04-21）保持主 KB lean。"
confidence: single_run
original_id: doc/shared/REGRESSION_METHODOLOGY.md
timestamp_inferred: true
tags: [regression, methodology, oracle, verification, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
# 重构/升级后防止 regression 的方法论

> **位置**: `src/skills/aog-regression-check/references/methodology.md`。由 `/aog-regression-check` skill **条件性载入**——日常 port 流程不读本文档，只有在检测/诊断 regression 时才进入上下文。此设计（user 2026-04-21）保持主 KB lean。

## 一句话结论（2026-04-21）

**13 个已 DONE 的 L2 op 全部通过了 3-way oracle 复核（pytorch_oracle_reverify.py），无任何 kernel 回归**。工具链升级（case_gen schema、tolerance-mode、pytorch-native oracle 三件套）证实有效，可继续用于后续 port。

---

## 为什么需要这个方法论

- KB/skill/scaffold 持续演进时，我们需要确认**老 op 没有因为新规则被改坏**
- 单次 benchmark 通过 ≠ spec 对齐。op#10 OL-83 证明，`torch_npu.*` 作为 reference 本身可能漂移，因此仅凭 benchmark PASS 不够
- 并行 NPU 运行会产生**假阳性 drift**（见 PB-15）——必须串行验证才算数
- 每次重构（如 classifier 逻辑调整、tolerance 调整）都可能产生 silent breakage，只有系统性复核才能抓到

---

## 工具栈（2026-04-21 完成的三件套）

| 工具 | 作用 | 何时用 |
|---|---|---|
| `src/scripts/pytorch_oracle_reverify.py` | 三路对比 kernel(NPU) / torch_ref(NPU) / pytorch-native(CPU) 的分类器，输出 `all_bit_exact` / `torch_npu_drift` / `kernel_drift` / `spec_ambiguous` | 每次 scaffold 或关键 KB 更新后，对所有 DONE op 跑一遍 |
| `src/scripts/batch_oracle_reverify.sh` | 串行批量 driver，逐 op 部署+build+跑+归档 | 批量回归测试 |
| `src/scripts/drift_probe.py` | 单 op 单 case 诊断器，dump shape/dtype/top-k-diff/bit-hex | reverify 标出问题时用于隔离、确认是否假阳性 |
| `src/scripts/aggregate_oracle_reverify.py` | 扫 `<op>/ol83_reverify.md` 输出 aggregate 表 + 行动项 | 收尾汇总 |

---

## Regression 防止 workflow（重构后必走）

1. **冻结基线** — 记录当前 REPORT.md 状态（已 DONE 的 op 列表、精度/性能数字）
2. **跑全量 reverify** — 串行：`bash src/scripts/batch_oracle_reverify.sh /tmp/l2_pass_ops.txt --max-cases 10`
3. **读 aggregate** — `python3 src/scripts/aggregate_oracle_reverify.py --out docs/regression_check_<date>.md`
4. **分类处理**：
   - `kernel_drift` → **必须** 用 drift_probe.py 串行复查；若仍有 drift，冻结 op DONE 状态 + 开 DEBT
   - `torch_npu_drift` ≥ 3 → 升级方法学为 P-P58.2 子规则
   - `spec_ambiguous` → 通常是 fp16 归约顺序或 topk tie-break，记录但不阻塞
   - `all_bit_exact` → 干净
5. **串行优先**（PB-15 约束）——同一 docker 容器内**禁止并行 launch** NPU op；若需并行，必须 1 容器/NPU 隔离
6. **commit + push** — aggregate doc 入库作为 regression 记录

---

## 2026-04-21 regression 复核结果（基线）

Full aggregate: `docs/analysis/P0_oracle_reverify_aggregate_20260421.md`

| # | Op | Oracle 分类 | 说明 |
|:--:|---|:---:|---|
| 2 | GroupNormSwish | ✅ all_bit_exact | 干净 |
| 4 | MoeComputeExpertTokens | ✅ all_bit_exact | 干净 |
| 7 | MoeGatingTopKSoftmax | 🟡 spec_ambiguous (9/10) | topk tie-break；kernel 与 torch_ref 位等 |
| 8 | QuantScatter | ✅ all_bit_exact (7/10 + 3 errs) | oracle Python 循环慢，3 case 超时；非 kernel 问题 |
| 9 | TopKTopP | 🟡 spec_ambiguous | 9 all_bit_exact + 1 spec_amb；tie 位的 tie-break |
| 10 | SwigluQuant | 🟡 spec_ambiguous | 9 all_bit_exact + 1 case 命中 ±0.5 边界（OL-83 类） |
| 14 | AdaptiveInstanceNormalization2DBackward | 🟡 spec_ambiguous (1/10) | fp16 归约顺序 |
| 18 | FusedAddRmsnorm | ✅ all_bit_exact | 干净 |
| 19 | FusedResidualRmsNormBackward | 🔀 torch_npu_drift (1/10) | 1 case 5.72e-06 bf16 ULP（torch_npu 融合路径） |
| 21 | GaussianTopkSparseActivation | ✅ all_bit_exact | 干净 |
| 26 | MoeGroupScoreAggregationAndMasking | 🟡 spec_ambiguous (10/10) | **已确认** topk tie-break（masked_fill -inf 位置） |
| 29 | TanhGatedResidualAddBackward | ✅ all_bit_exact | 干净 |
| 30 | TimeDecayExponentialStabilization | 🟡 spec_ambiguous (9/10) | fp16 递归累加顺序 |

**聚合**：6 all_bit_exact + 6 spec_ambiguous + 1 torch_npu_drift + **0 kernel_drift**。**无 op 需要冻结 DONE**。

---

## 发现与教训

### PB-15: 并行 NPU 假阳性 drift
- **现象**: 2026-04-21 首次 4 路并行 NPU(0-3) 跑 op#9/19/21 重现 drift 研究，返回 "kernel_drift 3e+38"
- **串行复查**: 全部假阳性，`ko_max ≤ 0.008`（bf16 ULP 级）
- **根因**: 同 docker 容器内并行 NPU launch → cross-kernel 运行时状态污染
- **约束**: `batch_oracle_reverify.sh` 与 `drift_probe.py` 均强制串行；若需并行必须 1 容器/NPU

### 分类器假阳性修复（inf_position_drift）
- **现象**: op#26 MoeGroupScoreAggregationAndMasking kernel 与 torch_ref 均使用 topk 选 top-k group，tie 破损不同 → masked_fill(-inf) 位置不同
- **原 classifier**: 把这种"位置差异"归类为 kernel_drift（假阳性）
- **修复**: 新 `inf_position_drift` 字段 — 当 NaN/Inf mask 不同但 mutual-finite 位置值相等时标记；classifier 把此类降级为 spec_ambiguous
- **验证**: op#26 重测 kernel_drift 10/10 → spec_ambiguous 10/10 ✅

### 工具 vs 方法学分离
- `pytorch_oracle_reverify.py` 做**分类**（是否 drift、什么类）
- `drift_probe.py` 做**诊断**（具体数值是什么、在哪里）
- 这样分离的好处：scaffold bug 在 drift_probe 里会明显（原始数值不会因 classifier 逻辑变化）
- 反面教训：若只有 classifier，scaffold bug 会被误认为 kernel bug（如 2026-04-21 最初判 op#19 有 3e+38 drift）

---

## 下次重构前 checklist

- [ ] 跑 batch_oracle_reverify.sh，aggregate 结果与本文档 baseline 一致吗？
- [ ] 如果有变化，drift_probe.py 能复现吗（串行，单 NPU）？
- [ ] 变化是 **真 regression**（kernel 代码或硬件变化）还是 **scaffold bug**（classifier 逻辑变化误报）？
- [ ] 无论哪种，commit message 必须引用本文档 baseline 或 override 原因
- [ ] 如果是 kernel regression，冻结 op DONE 状态 + 打 DEBT + 停止新 port

---

## 维护记录

- 2026-04-21 首次建立（commit `2b0cad4` 后重构，commit `<本 commit>` 抽离方法学）
- 后续每次重构后追加 `## YYYY-MM-DD regression check` 小节，列聚合与 delta

<!-- 迁移自 porter kb/shared/REGRESSION_METHODOLOGY.md(整档忠实搬运,convert_docs_to_okf.py)。跨 op 参考/方法论知识,非机械家族。 -->
