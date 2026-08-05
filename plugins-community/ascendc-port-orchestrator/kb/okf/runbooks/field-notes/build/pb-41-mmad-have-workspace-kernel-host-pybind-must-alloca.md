---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Mmad/HAVE_WORKSPACE kernel — host pybind MUST allocate sysWs + userWs (kernel gets `GetUserWorkspace(w)=w+sysWsSize`) [V220, workspace, matmul]"
description: "applies_to: soc=Ascend910_9382 (V220 / A3); cann=9.0.0; bisheng=n/a; op_class=all (any Mmad/cube kernel)"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend910_9382 (V220 / A3); cann=9.0.0; bisheng=n/a; op_class=all (any Mmad/cube kernel)"
confidence: single_run
original_id: PB-41
timestamp_inferred: true
tags: [507015, ascendc, pb-41]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_9382 (V220 / A3); cann=9.0.0; bisheng=n/a; op_class=all (any Mmad/cube kernel)`
`verified_on: soc=Ascend910_9382; cann=9.0.0`
`unverified_on: soc=Ascend950PR (V351 / A5 — same auto_gen wrapper expected, untested)`

- **Severity**: HIGH — silent data corruption (no error, no hang), produces garbage / nan / degenerate output that masquerades as a precision/algorithm bug.
- **Mechanism**: when a kernel uses `Mmad` (cube), the auto_gen / `KERNEL_TASK_TYPE` wrapper enables `HAVE_WORKSPACE` and the device entry computes the kernel-visible user workspace as `GetUserWorkspace(rawWorkspace) == rawWorkspace + sysWsSize` — it reserves a **system-workspace prefix** (FFTS / matmul scratch) ahead of the user region. If the HOST (pybind) allocates only `userWs` bytes, the kernel's user-workspace window `[w+sysWsSize, w+sysWsSize+userWs)` runs OFF THE END of the buffer → overwrites unrelated GM / reads uninitialized GM → corrupt cube↔vec workspace exchange (gather/scores/dscores/dgk all garbage).
- **Detection**: cube kernel produces structurally-wrong output (huge-finite for bf16 / nan for fp16) on EVERY case, while the same kernel's non-Mmad paths look fine; the magnitude is "uninitialized memory" not "slightly off". Pre-check: kernel calls `Mmad` AND pybind allocates `at::zeros({userWs})` without a `sysWs` term.
- **Fix**: allocate `sysWs + userWs` on the host. Get `sysWs` from `PlatformAscendCManager::GetInstance(soc)->GetLibApiWorkSpaceSize()` (the precise reserve), or over-allocate a safe constant — **16 MiB is a safe V220 over-alloc** for FFTS+matmul. Over-allocation is harmless; under-allocation corrupts.
  ```cpp
  const uint64_t sysWs = 16ull * 1024 * 1024;   // V220 FFTS+matmul system reserve
  uint64_t totalWs = sysWs + userWs;            // NOT just userWs
  at::Tensor w = at::zeros({(int64_t)totalWs}, at::device(kPrivateUse1).dtype(at::kByte));
  ```
- **Evidence**: lightning_indexer_grad (generated AscendC kernel, A3, 2026-05-27) — cube↔vec LIG-backward kernel; allocating only userWs gave garbage/degenerate dq/dk/dweights; adding the 16 MiB sysWs prefix moved precision 3/38 → 12/38 (and removed all garbage/huge values), isolating the remaining failures to genuine compute bugs.
- **Other instances (predicted)**: any cube/matmul AscendC kernel with a user-managed GM workspace (FlashAttention, GroupedMatmul, fused norm+matmul, MoE finalize, any two-stage cube↔vec handoff). The bug is generic to `HAVE_WORKSPACE` + host-allocated workspace, not LIG-specific.
- **A5/V351 CONFIRMED + nuance (FlashAttention-A5, 2026-06-02 — upgrades the `unverified_on: Ascend950PR` line above for the custom-`<<<>>>`-launch case)**: on a hand-written launch (pybind + `<<<>>>`, no aclnn/GE framework) the framework's pre-op workspace registration is ABSENT, so the kernel-visible `GetUserWorkspace(workspace)` returns garbage and cube GM-staging writes OOB → `507015`. Two A5-specific facts beyond the V220 entry:
  - **`GetUserWorkspace(workspace)` IGNORES its argument** — it returns the GLOBAL `g_sysWorkspaceReserved + RESERVED_WORKSPACE` (`RESERVED_WORKSPACE = 16 MiB` on arch35; on dav-c310/`__NPU_ARCH__==3510` the base resolves via `GetSysWorkSpacePtr() = __get_kfc_workspace_addr()`). The launch MUST set that global, not merely pass a sized buffer.
  - **`SetSysWorkspace` is `[[deprecated]]` and CONDITIONAL** (`if (g_sysWorkspaceReserved == nullptr)`) → silent no-op if the global is already set or the call is optimized away → global stays nullptr → `GetUserWorkspace` returns `nullptr + 16MB = 0x1000000` garbage → OOB `507015`. The custom launch must call **`AscendC::SetSysWorkspaceForce(workspace)`** (unconditional) before `GetUserWorkspace`, with the workspace sized `data + 16MB`.
  - **Manifestation split**: the MULTI-CORE FFTS cross-core deadlock (≥2 MIX groups, no registered sys-scratch base → hang at `synchronize()`) is **DS-confirmed** (FA-A5 multi-core CASE14 9.9s RC=0 after the fix; independent --clean build, device.o recompile VALID). The single-core large-D GM-staging OOB recovery via `SetSysWorkspaceForce` is **provisional** (independent prototype FA-A5 large-D, pending DS corroborate — see candidates.md `CAND-FA-A5-KFC-WORKSPACE`).
  - **Diagnostic**: a D≤128 (UB-resident) path never dereferences the user-workspace pointer → it passes even when the global is unset; only D>192 (GM-staging) or multi-core (FFTS) exposes the missing registration. A subset passing while large-D / multi-core crashes is the smoking gun.

<!-- 迁移自 porter kb/target/ascendc/（PB-41，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
