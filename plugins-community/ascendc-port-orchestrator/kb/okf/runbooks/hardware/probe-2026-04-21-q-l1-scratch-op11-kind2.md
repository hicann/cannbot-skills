---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: curated
title: "Q1 Probe Report: TBuf<TPosition::A1> in pure AIV kernel"
description: "Q1 Probe Report: TBuf<TPosition::A1 in pure AIV kernel"
confidence: single_run
original_id: hw/2026-04-21_Q_l1_scratch_op11_kind2
timestamp_inferred: true
tags: [hardware, probe_findings, 2026-04-21-q-l1-scratch-op11-k]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
# Q1 Probe Report: TBuf<TPosition::A1> in pure AIV kernel

## Verdict

**ACCEPT_MISCOMPILE** — bisheng compiles the kernel with no errors or warnings, but the emitted code triggers an `aivec` illegal-instruction fault at runtime (error 259, runtime 507035) the first time either `DataCopy(UB, L1)` or `DataCopy(L1, UB)` executes on the AIV core.

## Environment

Captured 2026-04-22 from A5 container `npu_dev3`:

- **OS**: openEuler 22.03 (LTS-SP4)
- **Bisheng/ccec compiler**: clang 15.0.5 (clang-5c68a1cb1231 flang-5c68a1cb1231), build stamp `2026-03-21T17:07:34+08:00`
- **CANN**: 9.0.0, innerversion V100R001C10SPC001B218, path `/usr/local/Ascend/cann-9.0.0`
- **NPU-SMI**: 25.7.rc1.b087, driver Version 25.7.rc1.b087
- **SOC**: Ascend950PR (per npu-smi), built with `-DSOC_VERSION=Ascend950PR_9589`
- **NPU used**: device 1 (AICore 0%, HBM 5235 / 131072 MB — least-loaded of 3 visible NPUs, 0 running processes)
- **torch / torch_npu**: not cleanly importable standalone in the container shell (libhccl.so missing on PATH during bare-shell probe), but import works from inside the `run_probe.py` script via the launcher env.

## Probe design

4 KB (1024 fp32) round-trip: GM → UB → L1 (`TBuf<TPosition::A1>`) → UB → GM. One block, pure AIV (`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`), no Cube / Mmad / Load2D.

Source: `workspace/probe_l1_scratch/kernel/probe_kernel.h` — see `l1Buf_` declaration `TBuf<TPosition::A1> l1Buf_;` and the UB↔L1 `DataCopy` calls.

## Iterations run (2 / 2 allowed)

### Iter 1 — PipeBarrier<PIPE_ALL> between UB→L1 and L1→UB

- **Build**: OK. No warnings mentioning L1/A1/TPosition. `libkernels.a` and `_probe_ext.cpython-311-x86_64-linux-gnu.so` both linked successfully.
- **Runtime**: Vector core exception, error 259.

Excerpt:
```
EZ9999: ... there is an aivec error exception, core id is 0, error code = 259,
dump info: pc start: 0x120041000000, current: 0x12004100030c, ...
extend info: errcode:(259) errorStr: Illegal instruction, which is usually
caused by unaligned UUB addresses. subErrType: 0x4.
Kernel task happen error, retCode=0x31, [vector core exception].
```

### Iter 2 — explicit MTE3→MTE1 SetFlag / WaitFlag sync (replacing the barrier)

Hypothesis: the fault was a missing cross-pipe ordering (UB→L1 goes on MTE3, L1→UB goes on MTE1), not the L1 access itself.

- **Build**: OK.
- **Runtime**: identical failure mode — error code 259, subErrType 0x4, same "Illegal instruction on aivec core". Current-PC shifted to `0x12004100031c` (the added SetFlag/WaitFlag extended the prologue by ~16 bytes); the trap still fires at the first `DataCopy` touching L1.

Interpretation: the issue is not a pipe-sync bug. It is an instruction-level incompatibility — bisheng emits some opcode / operand that the AIV scalar/MTE unit rejects when the source or destination is an L1 (`A1`) LocalTensor rather than a VECIN/VECOUT UB tensor. Sync does not change which opcodes are emitted.

## What this tells us

1. **Syntactic acceptance ≠ runtime support.** Both `TBuf<TPosition::A1>` / `TPipe::InitBuffer(l1Buf_, bytes)` and the `DataCopy(L1_tensor, UB_tensor)` / `DataCopy(UB_tensor, L1_tensor)` overloads compile cleanly (no warnings), but the instruction stream they produce is illegal on an AIV-only kernel on this CANN 9.0.0 / bisheng 2026-03-21 build.
2. **The generic `DataCopy` overload is not the UB↔L1 bridge for AIV on 351x.** The 351x public doc describes a hardware UB↔L1 path (MTE3 UB→L1, MTE1 L1→UB), and the AscendC API ref's `TPosition` enum does list A1 as a valid position, but the user-facing intrinsic to drive that hardware path from pure AIV code is either (a) not the generic `DataCopy(LocalTensor, LocalTensor)` template, (b) gated on a Cube/AIC context being present in the same kernel, or (c) not yet exposed in this CANN version. Probe cannot distinguish (a) / (b) / (c), but all three have the same consequence for Op#11.
3. **No miscompile of pointer math / alignment that we caused.** The probe buffer is 4 KB, the `LocalTensor<float>` is 32 B-aligned by construction (`InitBuffer` with 4096 B), and the `DataCopy` count is 1024 elements which is a multiple of the fp32 block size (8). The "unaligned UUB" string in the error is misleading — it is the generic `errcode:259` message, not proof that our pointer was unaligned.

## Recommendation to orchestrator (3 lines)

1. **Do NOT pursue the Kind-2 rewrite of Op#11 DequantSwigluQuant that assumed `TBuf<TPosition::A1>` as UB spill.** On this CANN 9.0.0 / bisheng toolchain, the compiler silently accepts the construct but runtime fails — any kernel built on this assumption will crash at deploy, after wasting a worker's 5-iter budget.
2. **Unblock Op#11's 0.54x via a different axis**: reduce per-tile UB footprint (smaller tile, fused buffer reuse, or fp16 intermediate storage with fp32 compute), or split the kernel into two launches so each pass has smaller UB. DEBT-030 should be closed as "A1 scratch not viable on this CANN" with this probe report as evidence.
3. **Escalate upstream only if the 2x gain is critical**: file a ticket/question with CANN asking whether a dedicated UB↔L1 intrinsic (e.g. `Copy`, `CopyUbufToL1`, or an AIV-scope `LoadData` variant) exists on 351x and is not yet in the public API ref. Do not block Op#11 on that answer.

## Evidence files

- `workspace/probe_l1_scratch/kernel/probe_kernel.h` (with explicit MTE3→MTE1 sync, iter 2)
- `workspace/probe_l1_scratch/kernel/probe_kernels.cpp`
- `workspace/probe_l1_scratch/kernel/pybind11.cpp`
- `workspace/probe_l1_scratch/kernel/probe_tiling.h`
- `workspace/probe_l1_scratch/model_new_ascendc.py`
- `/home/npu_user/workspace/AscendOpGenAgent/current_task/run_probe.py` (runner, copies also visible on A5 at `/root/AscendOpGenAgent/current_task/run_probe.py`)
- Full stderr from both iters captured above.

<!-- 迁移自 porter kb/hardware/probe_findings/2026-04-21_Q_l1_scratch_op11_kind2.md（convert_hardware_to_okf.py，硬件事实→reference/hardware）。 -->
