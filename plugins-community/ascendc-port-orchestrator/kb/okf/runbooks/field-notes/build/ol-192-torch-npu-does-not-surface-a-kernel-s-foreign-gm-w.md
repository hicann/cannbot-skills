---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "torch_npu does not surface a kernel's foreign GM writes into a returned raw-byte workspace tensor — probe via TYPED OUTPUT tensors"
description: "applies_to: soc=all; cann=9.0.0; bisheng=n/a; op_class=all (debugging methodology)"
phenomenon: build_failure
signal:
  - "applies_to: soc=all; cann=9.0.0; bisheng=n/a; op_class=all (debugging methodology)"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-192
timestamp_inferred: true
tags: [ascendc, ol-192]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=all; cann=9.0.0; bisheng=n/a; op_class=all (debugging methodology)`
`verified_on: soc=Ascend910_9382; cann=9.0.0`

**Principle**: when debugging a cube↔vec kernel by returning its GM scratch buffer (`at::zeros({totalWs}, kByte)`) from pybind to inspect intermediate workspace tensors, torch_npu reads the buffer back as ALL-ZERO — it does not track the kernel's foreign (device-side, non-torch) writes into a raw-byte tensor's storage, so the readback is not a faithful view of what the kernel wrote. Workspace-tensor readback is therefore an unreliable probe and will send you chasing a phantom "workspace is zero" bug. Instead, probe kernel internals through **typed OUTPUT tensors** (the op's real fp16/bf16/fp32 outputs, which torch_npu DOES track) — e.g. temporarily route an intermediate to a real output slot, or add a debug output tensor of the correct dtype.

## Evidence
- lightning_indexer_grad (A3, 2026-05-27): returning the raw-byte workspace from pybind read back all-zero, falsely suggesting the gather never wrote `ws_gk`; the gather was actually fine. The trustworthy signal was the typed `dq`/`dk`/`dweights` output tensors (e.g. dq's alternating-zero-over-D signature localized a fractal bug).

## Other instances (predicted)
Any AscendC kernel debugged by returning GM scratch to Python; any multi-stage cube↔vec / workspace-queue kernel where you want to inspect an intermediate.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-192（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
