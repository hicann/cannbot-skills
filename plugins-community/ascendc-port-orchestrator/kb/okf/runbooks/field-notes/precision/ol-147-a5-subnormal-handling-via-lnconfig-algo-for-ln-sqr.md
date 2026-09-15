---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A5 Subnormal handling via `LnConfig{algo}` for Ln/Sqrt/Rsqrt/Div/Reciprocal/Exp [V351, precision-config]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=elementwise-math,norm-with-eps"
phenomenon: precision_issue
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=elementwise-math,norm-with-eps"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-147
timestamp_inferred: true
tags: [ascendc, ol-147]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=elementwise-math,norm-with-eps`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 references/migration/基础API迁移指导.md §section7364115741514; PR 103 SKILL.md §1185`

**Rule**: On A5, hardware **does not natively support subnormal (denormalized) floating-point values**. Subnormals are treated as zero (flush-to-zero, FTZ) by default. For ops whose **input domain includes very small values near float min** AND whose **computation is sensitive to subnormal vs zero behavior**, the kernel MUST explicitly opt into software-emulated subnormal handling via a `<Op>Config` template-parameter struct.

**Why this matters**:
- A3 (V220) hardware natively handled subnormals; existing A3 kernels rely on this behavior implicitly. A5 silently flushes them to zero → precision drift on small-value test cases.
- Default-config calls (`Ln(dst, src, count)` with no config) get FTZ_TRUE behavior. Ops that need FTZ_FALSE MUST pass an explicit config.
- The drift is small in magnitude but consistent — appears as a systematic bias in MERE/MARE metrics on near-zero inputs, easily missed if test cases don't probe the small-value domain.

**Affected APIs** (all in `AscendC::` namespace):

| API | Config struct | Default `algo` |
|---|---|---|
| `Ln` | `LnConfig` | `LnAlgo::INTRINSIC` (= FTZ_TRUE) |
| `Sqrt` | `SqrtConfig` | `SqrtAlgo::INTRINSIC` |
| `Rsqrt` | `RsqrtConfig` | `RsqrtAlgo::INTRINSIC` |
| `Div` | `DivConfig` | `DivAlgo::INTRINSIC` |
| `Reciprocal` | `ReciprocalConfig` | `ReciprocalAlgo::INTRINSIC` |
| `Exp` | `ExpConfig` | `ExpAlgo::INTRINSIC` |

**Config schema** (same across all 6 APIs):

```cpp
// AscendC::LnAlgo enum:
enum class LnAlgo {
    INTRINSIC,                   // hardware path, FTZ_TRUE — subnormals → 0
    PRECISION_1ULP_FTZ_TRUE,     // same as INTRINSIC (alias)
    PRECISION_1ULP_FTZ_FALSE,    // software path, FTZ_FALSE — preserves subnormals via extended-precision
};
struct LnConfig { LnAlgo algo; };
```

**Concrete usage** (input domain includes subnormal-range values):

```cpp
constexpr AscendC::LnConfig CONFIG_PRESERVE_SUBNORMAL = {
    AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE
};

template <typename T>
__aicore__ inline void Compute(GM_ADDR dst, GM_ADDR src, uint32_t count) {
    // ... TPipe / TQue setup ...
    AscendC::Ln<T, CONFIG_PRESERVE_SUBNORMAL>(dstLocal, srcLocal, count);
    // ... etc
}
```

**When to opt into FTZ_FALSE**:

1. **Input domain provably includes subnormal values** — e.g., scaled-output paths where prior compute produces values in 2⁻¹²⁶ range (fp32 subnormal threshold). Common in attention softmax (post-exp range), log-likelihood (ln of small probabilities), norm-with-tiny-eps.
2. **Test cases cover that domain** — ad-hoc fp32 random in [-1, 1] uniform won't trigger the regression; need explicit small-value test points.
3. **Reference (CPU PyTorch or A3 kernel) treats subnormals correctly** — and the precision target requires matching that behavior.

**When FTZ_TRUE is acceptable**:

1. **All inputs guaranteed ≥ 2⁻¹²⁶** in fp32 (e.g., post-clamp, post-eps-add, post-explicit-zero-guard)
2. **Op is dominated by larger magnitudes** — relative-error metrics (MERE/MARE per P-P94) won't be perturbed by exact-zero replacement of values that round to subnormal anyway
3. **Performance-critical** — FTZ_FALSE software emulation has ~2-3× per-element cost on subnormal-domain inputs

**Detection signature**:

```bash
# In an A3→A5 port, grep for math APIs without explicit config — these inherit FTZ_TRUE
grep -nE "AscendC::(Ln|Sqrt|Rsqrt|Div|Reciprocal|Exp)\s*\(" arch35/*.h | \
  grep -v ", CONFIG_"
# Each result is a candidate — needs analysis: is the input domain subnormal-safe?
```

**Anti-pattern**: blanket-applying FTZ_FALSE everywhere "to be safe" — pays 2-3× compute cost without correctness benefit on subnormal-free inputs.

**Evidence**:
- PR 103 基础API迁移指导.md §section7364115741514 codifies as canonical L2 / mathematical-API migration step
- PR 103 SKILL.md §1185 explicitly warns: "L2 中 ReduceSumCustom 不替换为 ReduceSum (950 MicroAPI 版本)" — same family of A3→A5 API substitutions

**Other instances (predicted)**:
- Attention softmax (post-exp values in [0,1], near-zero after normalization) — possible subnormal exposure
- Layer/RMS norm with very small `eps` (e.g., `1e-30`) — Div by near-zero
- Log-likelihood / cross-entropy (Ln of softmax outputs)
- Any pow/exp chain where intermediate underflows

**Cross-reference**:
- OL-148 (RMSNorm/Softmax overflow-mode toggle — uses same `__NPU_ARCH__ == 3510` guard pattern)
- P-P94 (MERE/MARE thresholds — sensitive to FTZ choice on small-value cases)
- ASCENDC_API_CATALOG.md — these 6 APIs should cross-link this entry

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-147（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
