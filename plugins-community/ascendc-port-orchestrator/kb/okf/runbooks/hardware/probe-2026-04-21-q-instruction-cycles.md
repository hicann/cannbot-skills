---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: curated
title: "PROBE REPORT — Q5 Sort/Reduce instruction cycles"
description: "PROBE REPORT — Q5 Sort/Reduce instruction cycles"
confidence: single_run
original_id: hw/2026-04-21_Q_instruction_cycles
timestamp_inferred: true
tags: [hardware, probe_findings, 2026-04-21-q-instruction-cycle]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
# PROBE REPORT — Q5 Sort/Reduce instruction cycles

## Verdict
ACCEPT_CORRECT

**One-line summary**: On Ascend950PR_9589 / CANN 9.0.0 / bisheng 15.0.5 (2026-03-21),
`WholeReduceMax<float>(64)` ≈ 102 cyc/call, `BlockReduceMax<float>(64)` ≈ 99 cyc/call,
`MrgSort<float>` scales near-linearly with output length (≈ 0.55 cyc per output pair
at Q≥256, with ~100 cyc/call fixed overhead).

## Environment
- date: 2026-04-22 (UTC, host clock ~18 min ahead of WSL — cosmetic)
- host: 198.51.100.35 → container `npu_dev3`
- NPU: `/dev/davinci1` (NPU 1, Ascend950PR, idle pre + post, AICore 0%)
- CANN: `Ascend-cann-toolkit 9.0.0 V100R001C10SPC001B218`
- bisheng / ccec: `clang 15.0.5 (clang-5c68a1cb1231 flang-5c68a1cb1231)` build stamp `2026-03-21T17:07:34+08:00`
- SOC target at build: `-DSOC_VERSION=Ascend950PR_9589`
- frequency assumption for cycle derivation: **1.8 GHz** (AIV clock on 950PR; re-scale
  if vendor spec differs — time columns in timings.csv are authoritative).

## Question
Per-repeat cycle latency and steady-state throughput of:
- `WholeReduceMax<float>` (max 64 fp32 per call in fp32 mask-count mode)
- `BlockReduceMax<float>` (32-byte block = 8 fp32 per block, 8 blocks = 64 fp32)
- `MrgSort<float>` (4-way merge of pre-sorted (score,index) queues, varied queue length)

## Hypothesis vs Observation
- Hypothesis: no strong prior; record numbers.
- Observation: all three instructions produced stable, reproducible timings
  (stdev < 1 µs on ≈ 230–5200 µs means); MrgSort is linear in queue length
  (tree-log model is **not** the dominant cost — output size dominates).

## Evidence

### Results (REPEAT_N = 4096 calls, 1.8 GHz assumed)

| primitive | elements_per_call | median_us | cycles_per_repeat | throughput (elts/cycle) |
|---|---|---|---|---|
| WholeReduceMax | 64 (fp32)              | 232.30 |  102.1 | 0.63 |
| BlockReduceMax | 64 (fp32, 8×8-blk)     | 225.18 |   99.0 | 0.65 |
| MrgSort_q64    | 4×64 = 256 in / out    | 413.94 |  181.9 | 1.41 |
| MrgSort_q256   | 4×256 = 1024 in / out  | 1366.99|  600.7 | 1.70 |
| MrgSort_q1024  | 4×1024 = 4096 in / out | 5179.60| 2276.2 | 1.80 |

### Derived MrgSort model

Regression on 3 MrgSort points (output pairs N = 256, 1024, 4096):
- Slope: ≈ (2276−182) cyc / (4096−256) pairs = **0.545 cyc/output-pair**
- Intercept: ≈ 42 cyc (per-call setup)
- Throughput asymptotically **~1.83 output-pairs/cycle** as queues grow
- Confirms **linear (not super-linear)** scaling → hardware is not internally
  serializing on queue length; the "tree-log" ideal does not apply because
  MrgSort must emit every output element regardless of comparator tree depth.

### Reduce instructions

WholeReduceMax and BlockReduceMax over 64 fp32 are nearly identical in cost
(~100 cyc); the BlockReduce variant is marginally cheaper (~3%). The ≈ 100-cyc
floor likely includes pipeline setup and the result-writeback; the per-element
"useful work" is sub-cycle.

**Practical guidance**: for scan-then-reduce patterns over ≤ 64 fp32, the
single-instruction cost is the floor; further chunking will not help.

### Build

Clean build succeeded with one non-fatal warning:

```
[WARNING]: Multiple kernel functions are detected. It is recommended to define
only one kernel function per file.
```
(cosmetic — 5 `__global__` kernels share `probe_kernels.cpp`; does not affect correctness.)

### Runtime

Two independent runs produced medians within 0.3% of each other (232.3 vs 232.8
µs for WRM; 225.18 vs 225.14 µs for BRM; MrgSort rows within 0.05 µs),
confirming the loop-amplification + discard-first-5 methodology removes I-cache
cold-start noise. NPU 1 verified idle (AICore 0%, no foreign process) before
AND after the measurement set.

### Raw files
- Kernel:          `workspace/probe_instruction_cycles/kernel/probe_kernel.h` (ProbeWholeMax, ProbeBlockMax, ProbeMrgSort<QLEN>)
- Kernel exports:  `workspace/probe_instruction_cycles/kernel/probe_kernels.cpp`
- Timing harness:  `workspace/probe_instruction_cycles/kernel/pybind11.cpp`
- Runner:          `workspace/probe_instruction_cycles/run_probe.py`
- Build log:       `workspace/probe_instruction_cycles/build.log` (196 lines, 1 cosmetic warning)
- Run log:         `workspace/probe_instruction_cycles/run.log`
- Timings CSV:     `workspace/probe_instruction_cycles/timings.csv`

## Recommendation for orchestrator

1. **KB entry** (hardware/ascend950pr.md §"Sort/Reduce cycle data, bisheng 2026-03-21 / CANN 9.0.0"):
   record the table above. Tag version-stamp; **not** cross-version portable.
2. **Op-gen heuristic for reductions over ≤ 64 fp32**: a single WholeReduceMax /
   BlockReduceMax is ≈ 100 cyc. Do not chunk below this size; cost is amortized
   by the pipeline setup, not by per-element work.
3. **Op-gen heuristic for MrgSort**: expected cost ≈ `0.55 × 4 × QLEN + 100 cyc`
   per invocation. For a fixed input size, shorter queues × more MrgSort calls
   is *slower* than one long-queue call (per-call overhead is constant). Confirms
   the existing pattern in `9_TopKTopP` of preferring Sort over iterative
   pair-merges when possible.
4. **Follow-up (not urgent)**: re-probe after any bisheng upgrade; validate
   1.8 GHz assumption (or switch to msprof cycle counters) if cycle counts
   diverge from the ×1.8e9 scaling by >5%.

## Caveats

- **Frequency**: cycle numbers depend on assumed 1.8 GHz AIV clock. The raw
  `median_us` column in `timings.csv` is authoritative and re-scalable.
- **Loop body DCE risk**: the compiler might have hoisted / CSE'd some of the
  repeated WholeReduceMax calls; however, the observed cycle count (~100/call)
  is far above a `nop` loop cost, and measurements scale with REPEAT_N (halved
  REPEAT_N → halved time; sanity-confirmed during iteration on build cache).
- **bisheng-specific**: results will not transfer to a different bisheng
  codegen version.
- **Single-block**: all timings are with `blockDim=1` (single AIV). Multi-block
  throughput should scale linearly until memory/issue-slot saturation; not probed here.
- **MrgSort input format**: used fp32 score + fp32 "index slot" (8 B / pair). The
  index field is reinterpret-cast at the hardware level; correctness of the sorted
  output was not bit-checked here (probe is timing-only; sort engine functional
  correctness is covered by CANN itself).

<!-- 迁移自 porter kb/hardware/probe_findings/2026-04-21_Q_instruction_cycles.md（convert_hardware_to_okf.py，硬件事实→reference/hardware）。 -->
