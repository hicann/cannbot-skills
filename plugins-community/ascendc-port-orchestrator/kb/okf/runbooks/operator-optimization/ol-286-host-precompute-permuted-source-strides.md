---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Precompute permuted source strides on host"
description: "Encode a static permutation as permuted source strides so the device directly ravels output coordinates without an inverse-permutation search."
paradigm: ascendc
confidence: single_run
original_id: OL-286
timestamp_inferred: false
tags: [ascendc, permute, transpose, stride, layout, ol-286]
created_at: 2026-07-29T00:00:00Z
updated_at: 2026-07-29T00:00:00Z
---

`applies_to: soc=all; cann=all; op_class=permute/transpose/layout materialization; backend=ascendc`

## Principle

For output coordinate `o`, precompute `permuted_src_stride[k] = src_stride[dims[k]]` on the host. The kernel then evaluates `src_flat = sum(o[k] * permuted_src_stride[k])` directly. This avoids a per-element inverse-permutation lookup and keeps the device path independent of a variable-length `dims` tuple.

Validate that `dims` is a bijection of `[0, rank)`, combine with OL-285 when rank is padded, and use zero strides only for synthetic dimensions that must not contribute. The rule covers forward layout ops and backward transforms of gradient layouts.

**Evidence / provenance**: derived from historical card TR-OL-14. This form passed 149/149 12_Permute cases on 2026-05-17 across fp32/fp16/bf16 and rank-2/3/4 inputs. The stride equivalence is mathematical; the coverage is measured.
