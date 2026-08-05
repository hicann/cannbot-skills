---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Wrapper input materialization required — kernel reads zero on layout-conversion tensors (mechanism provisional)"
description: "applies_to: soc=Ascend910_9382; cann=9.0.0; bisheng=n/a; op_class=fa_class"
phenomenon: build_failure
signal:
  - "- AscendC kernel output = all zeros for non-trivial inputs"
confidence: single_run
original_id: EC-66
timestamp_inferred: true
tags: [ascendc, ec-66]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_9382; cann=9.0.0; bisheng=n/a; op_class=fa_class`
`source: DS 2026-05-27`

**Symptoms**:
- AscendC kernel output = all zeros for non-trivial inputs
- Adding `.abs().max().item()` or `print()` on input tensors fixes it
- Only affects non-identity layouts (BSH→BNSD, SBH→BNSD, BSND→BNSD)
- BNSD native layout works correctly (no async copy needed)
- Small shapes (Sq=2, Skv=2) more likely to trigger; large shapes may pass by timing luck

**Root cause**: NOT a simple cross-stream race. 2026-05-27 main C++ verify (198.51.100.92, CANN 9.0.0) tested both `aclrtSynchronizeStream` and `aclrtSynchronizeDevice` before pybind kernel launch — **neither fixed** the zero-output. Device-wide sync should have flushed any pending async copy regardless of stream. This rules out stream-ordering as the mechanism.

**Leading hypothesis (NOT confirmed)**: Python→pybind boundary crossing triggers tensor materialization that pure sync (Python or C++) and C++-side reads do not. The load-bearing invariant is: wrapper-side tensor op on inputs, before pybind call, at Python level. Same op moved into C++ pybind = INEFFECTIVE (main matrix #3).

**REFUTED hypotheses**: stream-ordering race (same-stream in source), sync (Python + C++ all INEFFECTIVE), generic device→host read (C++ read ineffective), record_stream (UNSTABLE 1-3/9), after-launch sync (INEFFECTIVE).

**2026-05-27 verify data** (DS + main + independent prototype, npu-a3@198.51.100.92):
| Method | Verdict |
|---|---|
| Baseline (no fix) | 2/9 (degenerate only); non-BNSD cand=0 |
| +aclrtSynchronizeStream before launch | INEFFECTIVE |
| +aclrtSynchronizeDevice before launch | INEFFECTIVE |
| Python `torch.npu.synchronize()` only (no read) | INEFFECTIVE (non-BNSD still zero) |
| Python `.abs().max().item()` read on inputs | Bug A resolved (9/9 non-zero) |

**Mechanism** (provisional, narrowed): A **wrapper-side** tensor op on the input tensors before the pybind call is required. The same read moved inside C++ pybind does NOT work (main matrix #3: `q.abs().max().to(CPU).item()` inside pybind = INEFFECTIVE). Pure `torch.npu.synchronize()` in wrapper also INEFFECTIVE. The load-bearing element is specifically "a Python-level operation on the input tensors at the wrapper boundary, before crossing into pybind" — not generic read, not sync. Candidate explanation: Python→pybind boundary crossing triggers tensor data materialization that pure stream sync and C++-side reads do not.

**Why cv-agent's pybind isn't the fix**: cv-agent stock pybind is byte-identical in stream handling — same `getCurrentNPUStream().stream(false)`, same `storage().data()` read, zero explicit sync. It works on its stock 16 cases by shape-timing luck (large shapes give async copy enough time to land). Copying cv-agent's pybind verbatim copies the same latent race.

**Confirmed fix (2026-05-27, main C++ verify matrix on npu-a3@92)**:
In the Python **wrapper** (`model_new_ascendc.py`), after `_to_bnsd()` and BEFORE the pybind kernel call, force tensor materialization:
```python
# After _to_bnsd, before kernel call:
_ = q_bnsd.abs().max().item()
_ = k_bnsd.abs().max().item()
_ = v_bnsd.abs().max().item()
```
This is 100% deterministic across runs. Fix lives in **wrapper emission**, NOT pybind C++.

**REFUTED (all pybind-side C++ approaches, main 2026-05-27 NPU 0 9-case verify)**:
- `aclrtSynchronizeStream` before launch: INEFFECTIVE
- `aclrtSynchronizeDevice` before launch: INEFFECTIVE
- C++ `.abs().max().to(CPU).item()` inside pybind: INEFFECTIVE
- `aclrtSynchronizeStream` AFTER launch: INEFFECTIVE
- `recordStream` for all tensors: UNSTABLE (1-3/9 across runs)
- device-sync before + recordStream after: INEFFECTIVE

**STEP_1.4 translator acceptance check**: verify emitted `model_new_ascendc.py` contains tensor materialization (e.g., `.abs().max().item()` or equivalent) after layout conversion and before kernel call. This is a wrapper-emission rule, not a pybind-emission rule.

**STEP_1.4 translator acceptance check**: grep emitted `model_new_ascendc.py` for input materialization (`.abs().max().item()` or equivalent device→host sync) after layout conversion and before pybind kernel call. Absence → `translator_block: wrapper_no_input_materialize`. This is a **wrapper-emission** rule (model_new_ascendc.py), NOT a pybind-emission rule.

**Detection**:
- **Bug pattern**: `model_new_ascendc.py` calls `_to_bnsd()` to convert layouts, immediately passes output to pybind kernel without any sync/materialize step. Kernel on small shapes reads zero.
- **Absence check**: grep `model_new_ascendc.py` for `.abs().max().item()` or `torch.npu.synchronize()` between `_to_bnsd` and the pybind kernel call. Absence = probable Bug A (wrapper-side fix needed).
- **NOT a pybind bug**: pybind11.cpp source shows kernel runs on `getCurrentNPUStream()` (same stream as `.contiguous()`). Cross-stream race is NOT the mechanism (confirmed by independent source read 2026-05-27).

**Evidence**: (B) autonomous chain emit on npu-a3@198.51.100.92, CANN 9.0.0, B=2/S=2/N=16/D=16 BSH fp16. DS + independent prototype Python-level materialization confirmed Bug A resolved (3-agent convergence). Main C++ verify matrix (2026-05-27, NPU 0, 7 variants tested): pybind-side approaches all REFUTED; wrapper-side Python `.abs().max().item()` after `_to_bnsd` = ONLY working fix (deterministic 9/9 non-zero across runs). 6/9 cases remain Bug-B-wrong (kernel=3.17-3.44 vs ref=2.67-3.10 — systematic-high, independent Bug B).

**Cross-ref**: R1-R4 host-API rules (PR #210/#212/#214), STEP_1.4 translator pre-build gate (PR #198), Bug B compute-diag (softmax/cube sub-16 alignment).

<!-- 迁移自 porter kb/target/ascendc/（EC-66，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
