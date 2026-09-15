---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Before spawning aog-kernel-worker, preflight benchmark reference run — schema-invalid inputs poison downstream evidence"
description: "paradigm: ascendc"
phenomenon: precision_issue
signal:
  - "about to spawn aog-kernel-worker on a new op, OR facing a Phase D crash across all cases with identical deterministic NPU PC"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-87
timestamp_inferred: true
tags: [507035, ascendc, ol-87]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: pipeline / preflight / harness-trust
- **Loaded by**: orchestrator (Phase O2.5 or new O2.6), aog-precision-probe (first hypothesis to rule out on all-case crashes), aog-kernel-worker (as safety check before trusting Phase D failure signatures)
- **Trigger**: about to spawn aog-kernel-worker on a new op, OR facing a Phase D crash across all cases with identical deterministic NPU PC
- **Evidence** (op#24 KvCacheUpdateWithRopeBackward, 2026-04-23 — REAL reproduced numbers, not speculation):
  - Before harness fix: reference `Model.forward()` run alone on NPU across 50 benchmark cases → **1/50 PASS + 49/50 CRASH** (error 507035 / AIV 334 subErrType 0x4). Kernel could not be verified against garbage reference outputs.
  - After harness fix (`.json` adds `"range": [0, max_seq - 1]` to cache_position across all 50 cases + `.py` reads the range field): reference `Model.forward()` → **50/50 PASS + 0 CRASH**, NPU stream healthy, kernel can be verified.
  - The fix mirrors PR #84 from 2026-04-13 (L1 op#24 EmbeddingDenseBackward) which addressed the same issue class in a sibling op.
- **Root cause (specific pattern, not overgeneralized)**: many L2 op benchmarks have `Model.get_input_groups()` that falls back to `torch.randint(0, 10000, shape)` for int64 tensor inputs without a range hint. For tensors semantically bounded by another input's dimension (e.g. `cache_position < max_seq`, `input_ids < vocab_size`, `expert_ids < num_experts`), this emits OOB indices in most shape cases. The `.json` schema already supports a `range` field per tensor (used correctly by L1 op#24's custom `get_input_groups()`) — it's just the generic int branch that ignores it.
- **Symptom on NPU** (when reference code uses fancy indexing `t[:, :, oob_indices]`):
  - Hardware fault error `aclrtSynchronizeStream failed, error code:507035` with `AIV 334 subErrType 0x4` ("BIU data to VEC incorrect") at deterministic PC across all 56 cores.
  - Stream enters sticky-error state — subsequent `torch.npu.synchronize()` returns 507035 regardless of input validity. Process restart required to recover (container-level or Python-level). **Crucially, this stream-poisoning cascade makes per-case statistics unreliable**: after the first case crashes, you cannot distinguish "case 1 is legitimately bad" from "NPU is stuck from case 0".
- **Preflight protocol** (Phase O2.6, before Phase O3 worker spawn):
  1. On A5 inside the container, run a standalone script that imports ONLY `model.py` (reference) and runs `Model.forward()` on each of the N cases independently. Use `try/except` per case. Do NOT break on first crash — continue through all N. Record PASS / CRASH per case.
  2. Thresholds:
     - `ref_pass_count == N` → proceed to Phase O3 (aog-kernel-worker spawn).
     - `N * 0.9 ≤ ref_pass_count < N` → WARN + annotate `ref_fail_cases` in PROGRESS.md, still proceed (edge cases).
     - `ref_pass_count < N * 0.9` → ABORT. Do NOT spawn aog-kernel-worker. Mark op as `BLOCKED_BY_HARNESS` in REPORT.md; file a DEBT entry referencing the benchmark branch/commit and evidence; if it's an int64-OOB signature, propose `.json`+`.py` range patch (see "fix pattern" below).
  3. Preflight script MUST handle NPU stream recovery when a case crashes — use a subprocess-per-case isolation (spawn a fresh Python process per case) OR a container restart between cases. Without isolation, case N+1 inherits the stream-poisoned state and all subsequent "crashes" are false positives.
- **Fix pattern** (when preflight finds the int64-OOB root cause):
  - `.json` each case: add `"range": [lo, hi]` (inclusive on both ends) to the int tensor.
  - `.py` int branch (if it's the generic template): change to `rng = inp.get('range', [0, default_max - 1]); torch.randint(rng[0], rng[1] + 1, shape, dtype=dtype)`. Default `default_max - 1` exactly matches legacy `torch.randint(0, default_max)` value-range, preserving backward compatibility for all cases without `range`.
  - Open upstream PR to the benchmark repo (mirror PR #84 / PR #119 style).
- **What this entry explicitly does NOT claim** (lessons from withdrawn OL-86, archived 2026-04-23):
  - This is NOT a universal "every all-case-crash with int64 input = OOB" diagnostic rule. The trigger condition requires BOTH an identified int64 tensor with index/position semantics AND the specific `BIU data to VEC incorrect` signature AND independent verification that ref-alone-per-case (with process isolation) reproduces the stream poisoning.
  - This does NOT justify clamping int64 tensors in orchestrator / pybind as a "kernel workaround" for a harness bug — the schema fix belongs in the benchmark, not in kernel source.
  - This does NOT replace OL-85 (logic-first precision fix). After harness is clean, remaining kernel precision residuals must still be investigated per OL-83 step-4 + OL-85 anti-overfitting.
- **Related**:
  - OL-83 (1-ULP boundary probe), OL-85 (logic-first), OL-80 (API catalog lookup before inventing workaround).
  - Historical precedent: PR #84 (L1 op#24), PR #107/#108 (MoE L2 op#4/5/6/7).
  - Our-side PR: https://github.com/Just-it/AscendOpGenAgent/pull/119 (L2 op#24).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-87（category=pipeline / preflight / harness-trust，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
