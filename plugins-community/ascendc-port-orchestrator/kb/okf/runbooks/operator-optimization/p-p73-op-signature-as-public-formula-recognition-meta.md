---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Op-signature-as-public-formula recognition (META workflow for fused activation ops)"
description: "When CANN exposes a fused op via torch_npu.npu_<X> and its parameter list contains scalar attributes with non-trivial default values (i.e., not 0.0 / 1.0 / generic), those defaults are usually the FIN"
severity: high
confidence: single_run
original_id: P-P73
timestamp_inferred: true
tags: [precision, optimization, p-p73, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

When CANN exposes a fused op via `torch_npu.npu_<X>` and its parameter list contains scalar attributes with **non-trivial default values** (i.e., not `0.0` / `1.0` / generic), those defaults are usually the FINGERPRINT of a published public formula. **Recognition checklist**: scan op signature for: `_alpha=<irrational>` (e.g. 1.702 = SiLU/Swish-1 approximation, 1.41 = √2/sqrt(π)), `clamp_limit=<small int>` (e.g. 7.0 = gpt-oss tanh saturation, 6.0 = ReLU6), `_beta=<small float>`, `_eps=<small>` (norm-class). **Action when match suspected**: (1) WebFetch `github.com/openai/gpt-oss` / `github.com/meta-llama/llama` / `github.com/mistralai/mistral-src` / model repos for the default-value triple; (2) read those repos' formula definitions; (3) probe a3 NPU reference vs that public formula on a small test case (4-row × 128-col is enough); (4) on bit-exact match, the kernel formula is now publicly traceable + KB-codifiable. **Why this matters**: avoids 50+ formula hypotheses (op#11 kw-1 worker burned ~2h before pp-1 recognized gpt-oss fingerprint and resolved in ~30 min). Pattern applies to ALL future fused activation ops where CANN op def has non-trivial defaults.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P73，convert_patterns_to_okf.py）。confidence 未升格。 -->
