---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Activation-layout convention disambiguation via CANN op-parameter default fingerprinting"
description: "When a CANN fused op exposes scalar parameters with non-trivial default values (irrational constants, small-int saturation limits, named bias terms), those defaults are usually the FINGERPRINT of a pu"
severity: high
confidence: single_run
original_id: P-P71
timestamp_inferred: true
tags: [precision, optimization, aclnndequantswigluquantv2, datacopypad, mul, p-p71, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

When a CANN fused op exposes scalar parameters with **non-trivial default values** (irrational constants, small-int saturation limits, named bias terms), those defaults are usually the FINGERPRINT of a published public formula — and CANN is faithfully implementing that formula INCLUDING its data-layout convention. **Workflow**: (1) grep the CANN op definition / `torch_npu.npu_<X>` signature for default values; (2) WebFetch known public OSS reference repos (`github.com/openai/gpt-oss`, `github.com/meta-llama/llama`, `github.com/mistralai/mistral-src`) for the default-value triple; (3) if a match is found, the layout (chunked vs interleaved-stride-2 vs others) matches that source's tensor packing. **Canonical instance** (current evidence): SwiGLU has two valid formulations — (A) **chunked halves** (PyTorch native): `gate, linear = chunk(x, 2, dim=-1)`, used by `swiglu_mode=0` of `aclnnDequantSwigluQuantV2` and most PyTorch references; (B) **interleaved stride-2** (gpt-oss / OpenAI): `gate = x[..., ::2]`, `linear = x[..., 1::2]`, used by `swiglu_mode=1` of `aclnnDequantSwigluQuantV2`. CANN convention NOT documented in torch_npu help — must determine empirically via the fingerprint workflow above. Op#11 evidence: probe pp-1 found `α=1.702 + L=7.0 + b=1.0` matched gpt-oss exactly (`gpt_oss/torch/model.py:358-365`) → interleaved stride-2 layout → bit-exact 100% match across 8 case combinations. **AscendC implementation choices for stride-2 extraction**: (a) `DataCopyPad` with `srcStride/dstStride/repeatStride` params, OR (b) full-row load + two `Mul` ops with `BinaryRepeatParams{src1BlkStride, src1RepStride}` patterning (preferred for V220 — avoids extra MTE2), OR (c) scalar-loop GetValue (slowest, only for tiny H). **Generalizes** to any future fused activation op exposing non-trivial scalar defaults: tanh-clamp variants, GeLU-tanh-approximation parameters, RoPE base-frequency, LayerNorm-eps fingerprinting, etc.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P71，convert_patterns_to_okf.py）。confidence 未升格。 -->
