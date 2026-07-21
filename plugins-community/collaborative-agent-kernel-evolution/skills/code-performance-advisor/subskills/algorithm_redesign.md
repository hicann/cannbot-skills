---
name: algorithm_redesign (Scalar-Locked Kernel Architect)
description: Algorithmic-level reformulation for kernels where rule-based suggestions are exhausted but scalar_ratio remains high. Proposes reformulations with rigorous equivalence classification and bounded error analysis.
---

## What I do

I address performance bottlenecks that **cannot be resolved by existing rules**. When `deep_research` or `suggest` correctly applies all available patterns yet `scalar_ratio` stays above threshold, the bottleneck is structural — it lives in the **algorithm itself**, not in implementation patterns.

This subskill transforms the bottleneck algorithm into a form with fewer scalar dependencies, fewer `GetValue` calls, and fewer conditional branches in the hot path — but **only after verifying the mathematical relationship between old and new formulas with explicit error bounds**.

---

## Trigger Conditions

The ROUTE phase sets this path automatically when **all** of the following hold:

### AND 条件（全部满足才触发）

| # | 条件 | 说明 |
|---|------|------|
| 1 | **迭代门槛**：`optimization_rounds ≥ 1` | 至少经历过一次完整 APPLY→EVALUATE 循环。第一轮不做算法重设计，规则库必须先被用完。 |
| 2 | **规则穷尽**：`max_rule_score < 0.40` 且至少有 1 条规则被评分 | 区分"规则没匹配"（深度不足，应回 deep_research）和"规则全试过但分低"（已穷举，才触发本 subskill）。 |
| 3 | **Scalar 持续偏高**（硬件自适应） | AIV 路径（910B vector 核）：`aiv_scalar_ratio > 0.35`；AIC 路径（cube 核）：`aic_scalar_ratio > 0.30`（cube 算子有 ~20% 正常 scalar 基线，阈值更保守）。 |
| 4 | **瓶颈可归因**：profiling 数据存在 | 无 profiling 数据时无法判断瓶颈类型，不触发。 |

### OR 排除条件（任意一项成立则不触发）

| 排除情形 | 处置建议 |
|----------|----------|
| 最近一次 EVALUATE 有正确性失败（三路比较任意指标超标） | 先修复正确性，不在有 bug 的代码上做算法重设计 |
| 最近两次评估间 scalar_ratio 改善 ≥ 5% | 优化仍在生效，继续规则路径 |
| APPLY 前后代码 hash 相同（未真正修改内核） | APPLY 没有发生，重新检查建议执行情况 |
| 硬件信息缺失（task_type = UNKNOWN） | 需要先重新 profiling 确认硬件类型 |

> **human override**：`deep_research` 结论明确标注 `bottleneck=algorithmic` 时，可跳过迭代门槛直接触发。

---

## Equivalence Taxonomy（必读，不得跳过）

在提出任何变换前，**必须**将其归类为以下四类之一，并在输出中明确标注。

| 类别 | 定义 | 可直接替换？ |
|------|------|-------------|
| **EXACT** | 在所有有限输入上位模式相同；NaN/Inf 传播行为一致 | ✅ 无条件可替换 |
| **WITHIN_TOLERANCE** | `max_re < dtype_machine_epsilon`（bfloat16: 2^-7；float16: 2^-10；float32: 2^-23） | ✅ 可替换，需标注适用 dtype |
| **APPROXIMATE** | 误差有界：`max_re ≤ E`，但 `E > dtype_epsilon`；需标注有效输入范围 [a, b] | ⚠️ 条件可替换：E 和 [a,b] 必须显式给出并由三路评测验证 |
| **MONOTONE_ONLY** | 符号/排序相同，但幅度不同；`max_re` 无上界 | ❌ **不得**声称可直接替换。这是模型架构变更，需模型 owner 审批，超出本 subskill 职责范围 |

**关键区分**：`MONOTONE_ONLY` ≠ 数值等价。两个函数在 x=0.01 处可能差 10 倍，即使它们的零点相同、单调性相同。历史上曾以"单调等价"为由替换 sigmoid，导致精度指标超标 3x，教训在案。

---

## Workflow: 4-Step Redesign Protocol

### Step 1: Scalar Dependency Chain Extraction

Map every scalar stall in the hot loop:

```
For each iteration of the main loop:
  List: GetValue calls → scalar arithmetic → Duplicate/Muls
  Mark: dependency edges (which scalar feeds which vector op)
  Output: DAG of scalar stalls
```

**Output format**:
```
Scalar chain #1 (RMSNorm):
  ReduceSum → [GetValue] → *invD → +eps → [Duplicate] → Sqrt → [GetValue] → 1.0f/x → Muls
  Stall count: 2 GetValue + 1 scalar_div
  Hot loop: YES (called 3x per tile iteration)
  Vector-replaceable: YES → Muls(1) + Adds(1) + Sqrt(1) + Reciprocal(1)

Scalar chain #2 (Gate sigmoid):
  ReduceSum → [GetValue] → *0.03125 → abs() → branch → sqrt(abs) → [GetValue] → sign
           → Duplicate → Exp(1) → [GetValue] → 1/(1+x)
  Stall count: 3 GetValue + 6 scalar ops + 2 branches
  Hot loop: YES (1x per tile iteration, but dominates because of branch cost on AIV)
  Vector-replaceable: PARTIAL — abs/sign branches block full vectorization
```

### Step 2: Algorithmic Equivalence Search

For each non-replaceable scalar chain, search for equivalent formulations.

**Transformation catalogue** — equivalence class is mandatory for each row:

| Original | Alternative | Equivalence Class | Required Condition |
|----------|-------------|-------------------|--------------------|
| `sqrt(abs(x)) * sign(x)` | `x / (sqrt(abs(x)) + ε)` | **APPROXIMATE** — E = O(ε/\|x\|) for \|x\|≫ε | abs(x) > ε across input distribution; ε = 1e-6 typical |
| `1.0f / (1 + exp(-x))` | `0.5f + 0.5f * tanhf(x * 0.5f)` | **WITHIN_TOLERANCE** for \|x\| ≤ 8 (bfloat16 saturates anyway) | Verify no inputs outside ±20 (overflow risk) |
| `1.0f / (1 + exp(-x))` | `expf(x) / (expf(x) + 1)` | **EXACT** (mathematically); implementation may differ by 1 ULP | Only for x ≥ 0 to avoid exp overflow |
| `sign(x)` scalar branch | `(x > 0.0f) * 2.0f - 1.0f` via Muls | **EXACT** for x ≠ 0 (sign undefined at 0); **APPROXIMATE** if 0 possible | Confirm x=0 cannot occur, or treat as +1 |
| `ReduceSum → GetValue → Duplicate` | `ReduceSum → Muls(1) → Adds(1) → ...` count=1 | **EXACT** | count=1 vector ops are single-element operations |
| `GetValue + 1.0f/x` | `Reciprocal(dst, src, 1)` | **WITHIN_TOLERANCE** (hardware fma precision) | CANN ≥ 8.x, verify `AscendC::Reciprocal` exists in api ref |
| `sigmoid(sqrt(abs(x)) * sign(x))` | `0.5f + 0.5f * tanhf(x * scale)` | **MONOTONE_ONLY** — shape is fundamentally different | **Not a replacement**; requires model-level approval |

> **On the Gate sigmoid chain specifically**: there is no known WITHIN_TOLERANCE replacement that eliminates all branches. The best achievable is APPROXIMATE with documented error bounds on the activation input distribution. Any proposal must go through numerical validation (Step 3) before code change.

### Step 3: Numerical Equivalence Validation

Before writing any code, quantify the error introduced by the reformulation.

**Required analysis for each APPROXIMATE transformation**:

```
Formula pair:
  Original:    f(x) = sigmoid(sqrt(abs(x)) * sign(x))
  Proposed:    g(x) = sigmoid(x / (sqrt(abs(x)) + 1e-6))

Equivalence class: APPROXIMATE
Operating range:   x ∈ [-4, 4]  (bfloat16 activations after normalization, 3σ estimate)
Error analysis:
  At x = 1.0: f(1) = sigmoid(1) = 0.731, g(1) = sigmoid(1/1.0000005) ≈ 0.731  Δ ≈ 0
  At x = 0.01: f(0.01) = sigmoid(0.1) = 0.525, g(0.01) = sigmoid(0.01/1.00001) ≈ 0.502  Δ ≈ 2.3%
  At x → 0:  f(0) = sigmoid(0) = 0.5, g(0) = sigmoid(0) = 0.5  (continuous)
  Max relative error vs original: ~2.3% at x near 0 (within bfloat16 2^-7 ≈ 0.78%)
  ⚠️  max_re > bfloat16 machine epsilon → class is APPROXIMATE, not WITHIN_TOLERANCE

Out-of-range behavior:
  x outside [-10, 10]: sigmoid saturates, error negligible
  x = 0 exactly: g(0) = sigmoid(0/1e-6) = sigmoid(0) = 0.5 — safe

Three-way validation prediction:
  Predicted svec ratio: ~1.3 (within threshold 2.0)
  Predicted mean_re ratio: ~1.1 (within threshold 2.0)
  Verdict: likely PASS — but must run evaluate.py to confirm
```

**Hard rule**: Do not write code for an APPROXIMATE transformation without completing this analysis. Do not proceed if predicted mean_re ratio > 1.8 (too close to the 2.0 gate).

### Step 4: Implementation Sketch + Risk Assessment

Produce a concrete AscendC code diff with:
- Before/after scalar chain count
- Buffer reuse analysis (does the new formula need extra scratch space?)
- Correctness risk (HIGH/MEDIUM/LOW) with justification
- Suggested test

---

## Output Format

```
Algorithm Redesign: [Op Name] — [Bottleneck Name]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Scalar Chain (current):
  [chain diagram with GetValue/branch counts, hot-loop marker]

Proposed Reformulation:
  Original:            f(x) = ...
  Proposed:            g(x) = ...
  Equivalence class:   APPROXIMATE  ← 必须填写
  Operating range:     x ∈ [a, b]   ← 必须填写
  Max absolute error:  < E          ← 必须量化
  Max relative error vs original: < E_rel  ← 必须量化
  Out-of-range behavior: [describe]

Code Change:
  [before/after diff, 10-20 lines]
  Buffer impact: [extra scratch? reuse existing?]

Expected Impact:
  GetValue eliminated: N
  Branches eliminated: N
  Estimated aiv_scalar_ratio: {before}% → {after}%  (Δ ≈ -{delta}%)
  Correctness risk: LOW/MEDIUM/HIGH — [reason tied to equivalence class]

Validation Required:
  Run: python3 evaluate.py {op_name} --advanced-perf --task-type vector
  Pass criteria: max_re ≤ 10.0, mean_re ≤ 2.0, svec ≤ 2.0  (three-way comparison)
  If FAIL: revert immediately, do not chain with next change
```

---

## Constraints

- **Equivalence class first**: Every proposed transformation must be classified before code is written. "It looks right" is not a class.
- **APPROXIMATE requires input range**: Never propose an APPROXIMATE transformation without specifying the operating range of the relevant variable. Claim without range is unfalsifiable.
- **MONOTONE_ONLY is not a replacement**: If the best achievable class is MONOTONE_ONLY, report that the scalar chain cannot be eliminated without changing the model's numerical output. Escalate to the user.
- **AscendC API check**: Before proposing any new API, verify it exists in `references/standards/ascendc_api_validation_reference.md`. The error `no member named 'X' in namespace 'AscendC'` has cost multiple compilation cycles — always verify first.
- **Single change per cycle**: Propose one algorithmic change per APPLY→EVALUATE cycle to isolate correctness impact.
- **Correctness gate**: If the three-way evaluation fails (any ratio > 2.0), revert immediately; do not chain with the next proposed change.

---

## 参考文档

- `references/standards/ascendc_api_validation_reference.md`：AscendC 合法 API 列表（含 `Reciprocal`、`Sqrt`、`Muls`、`Adds`）
- `notes/compilation_issues_and_fixes.md`：历史编译失败案例（含 `AscendC::Rec` 不存在、`AscendC::Reciprocal` 正确写法，DataCopy 替换标量循环精度失败案例）
- `subskills/deep_research.md`：诊断阶段输出（本 subskill 的输入来源）
