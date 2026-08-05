---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Cast(bf16→f32→bf16) roundtrip is not necessarily bit-exact"
description: "Cast(bf16→f32, CAST_NONE) is lossless (bf16 is a subset of f32), but Cast(f32→bf16, CAST_ROUND) may alter f32 bit patterns through intermediate VEC ops (such as an Adds bridge), producing a different"
phenomenon: build_failure
signal:
  - "bf16 data-movement kernel exhibits precision mismatch"
confidence: single_run
original_id: OL-44
timestamp_inferred: true
tags: [ascendc, platform_bug, ol-44]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
bf16 data-movement kernel exhibits precision mismatch

## 教训 / 根因
Cast(bf16→f32, CAST_NONE) is lossless (bf16 is a subset of f32), but Cast(f32→bf16, CAST_ROUND) may alter f32 bit patterns through intermediate VEC ops (such as an Adds bridge), producing a different bf16 after the roundtrip. In pure-copy kernels, use CAST_NONE for the reverse conversion (truncate to recover original bf16 bits), or SIMT direct assignment to bypass Cast.

## 证据
Pad V4 (2026-04-10): bf16 Cast roundtrip regression

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-44（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
