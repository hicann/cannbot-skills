---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "build_ascendc.py default SOC_VERSION causes 507035 on Ascend950PR"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Kernel builds OK but every run produces aclrtSynchronizeStream failed, error code:507035 with \"illegal instruction\" at PC 0x80"
confidence: single_run
original_id: EC-27
timestamp_inferred: true
tags: [507035, ascendc, ec-27]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (22_HybridAttentionMaskPreparation). Do not downgrade.
- **Error pattern**: Kernel builds OK but every run produces `aclrtSynchronizeStream failed, error code:507035` with "illegal instruction" at PC 0x80
- **Root cause**: `build_ascendc.py` defaults to `Ascend910B2` SOC if not specified. Binary is incompatible with Ascend950PR hardware.
- **Fix**: Always use `-v Ascend950PR_9589` flag. Our worker ENV already does this, but if any script omits it → instant 507035.
- **Detection**: If ALL test cases crash (not just some), and error is at very low PC offset (0x80), suspect SOC_VERSION mismatch.
- **Evidence**: 14_AdaptiveInstanceNormalization2DBackward (2026-04-16).
- **Sub-variant & casing clarification (2026-06-24, top_k_top_p_sample port_a3 kw-2)**: the Ascend950PR chip family is NOT a single SOC_VERSION string — CANN 9.0.0 `ascendc_kernel_cmake/legacy_modules/host_config.cmake` lists ~30 sibling variants under `ascend950_list` (`ascend950pr_9599`, `_958a`, `_9589`, `_958b`, `_9579`, `_957b`, `_957c`, `_957d`, plus `ascend950dt_*`), all mapping to arch `ascend950` (`opdesc_parser.py`). This refines the "Fix" above without contradicting it:
  1. **Case is normalized by cmake** — `host_config.cmake` does `string(TOLOWER "${SOC_VERSION}")` before matching the list, so `-v Ascend950PR_9589` (EC-27, build_ascendc.py PascalCase) and `ascend950pr_957b` (worker lowercase, port_a3 CANN cmake path) are BOTH accepted. Do not file PascalCase-vs-lowercase as a contradiction — cmake lowercases.
  2. **The `_95xx` sub-variant suffix is MANDATORY** — a bare family name `Ascend950PR` (or `ascend950pr`) is REJECTED (`FATAL_ERROR ... does not support`), because no `ascend950_list` entry lacks the suffix. The kw-2 note "`Ascend950PR` is rejected" = missing-suffix, NOT a casing bug.
  3. **Pick the variant that matches YOUR chip** — this A5 box (`npu-smi info -t board -i 0`: NPU Name `9579`, Chip `Ascend950PR` V100) built + ran 10/10 PASS with `ascend950pr_957b`; EC-27's anchor used `_9589`. Both are `ascend950`-arch siblings (binary-compatible for general compute). On a `507035`-style mismatch, confirm the variant via `npu-smi` and match the suffix rather than assuming one canonical string.
  - **Source**: CANN 9.0.0 install — `host_config.cmake:14` (`string(TOLOWER)` + `ascend950_list`), `opdesc_parser.py:53` (variant→arch map). Verified 2026-06-24. Resolves the regression-risk flag carried forward in two prior `top_k_top_p_sample` KB merges (they blocked `957b` vs `9589` as an EC-27 contradiction — it is not).

<!-- 迁移自 porter kb/target/ascendc/（EC-27，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
