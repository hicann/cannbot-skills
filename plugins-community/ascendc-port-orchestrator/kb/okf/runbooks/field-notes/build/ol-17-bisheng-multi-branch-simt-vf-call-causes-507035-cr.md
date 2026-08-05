---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "bisheng multi-branch Simt::VF_CALL causes 507035 crash"
description: "Putting multiple `Simt::VF_CALL<template1>(...); Simt::VF_CALL<template2>(...)` branches inside one `extern \"C\" __global__` function can cause ACL ERR 507035 (kernel launch failure). Root cause: bishe"
phenomenon: build_failure
signal:
  - "when writing AscendC __global__ dispatcher with multiple Simt::VF_CALL template instantiations"
confidence: single_run
original_id: OL-17
timestamp_inferred: true
tags: [507035, __global__, ascendc, platform_bug, ol-17]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
when writing AscendC __global__ dispatcher with multiple Simt::VF_CALL template instantiations

## 教训 / 根因
Putting multiple `Simt::VF_CALL<template1>(...); Simt::VF_CALL<template2>(...)` branches inside one `extern "C" __global__` function can cause ACL ERR 507035 (kernel launch failure). Root cause: bisheng compiler/linker binary slot corruption — certain kernel slot positions in a large binary produce broken device code. Fix: split each template instantiation into its own `__global__` entry point. Host-side dispatch selects the right entry.

## 证据
Pooling B variant crashed on cluster 2 (dim=1, edges=1024). After splitting into separate entry points, all 61 clusters pass.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-17（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
