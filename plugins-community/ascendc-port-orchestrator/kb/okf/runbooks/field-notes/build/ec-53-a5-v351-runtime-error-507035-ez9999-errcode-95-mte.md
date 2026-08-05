---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A5 V351 runtime error 507035 + EZ9999 `errcode:(95) MTE instruction DDR address out of range` — kernel GM extent overshoot (fault tree) [V351, port_a3_to_a5]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5"
phenomenon: build_failure
signal:
  - "(full A5 stack):"
confidence: single_run
original_id: EC-53
timestamp_inferred: true
tags: [507035, ascendc, ec-53]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A3 V220 silently tolerated the same overshoot — that is precisely why the upstream bug survived)`

**Error pattern** (full A5 stack):
```
AclrtSynchronizeDeviceWithTimeout, error code is 507035
EZ9999: ... errcode:(95) errorStr: The DDR address of the MTE instruction is
out of range. subErrType: 0x4
```
Distinct from EC-23 (DataCopyPad UB→GM 507035 — pure direction-not-supported)
and EC-27 (SOC_VERSION-derived 507035 — illegal instruction at PC 0x80). EC-53
is specifically the **MTE-OOR sub-class** of 507035, identified by the
`errcode:(95)` line in the EZ9999 detail.

**Root cause family**: kernel issues a MTE2 (GM→UB) or MTE3 (UB→GM) transfer whose
DDR address range extends past the actual allocation of the targeted tensor. A5
V351's MTE hardware boundary-checks; A3 V220 did not, so the same kernel binary
ran "fine" on A3 (silently reading/writing past the buffer end) and traps on A5.

**Fault tree** (port_a3_to_a5 — check in order):
1. **Input-extent conflation (most common)**: kernel uses `SetGlobalBuffer((__gm__ T*)<X>, extent)`
   where `extent` is derived from input A's shape but `<X>` is input/output B with a
   smaller shape. Detect via: grep kernel.h for `SetGlobalBuffer` and audit each `extent`
   argument against the actual tensor it pairs with. Fix: pybind padding wrapper —
   see OL-162.
2. **InitOutput overshoot**: `InitOutput(<tensor>_gm, X, 0)` writes X elements when the
   actual allocation is only Y < X. Triggered same way as #1 (X computed from a
   different shape). Fix: same as #1 (pybind padding) OR cap X at `min(X, actual_size)`
   when both are host-side known.
3. **UB-budget tiling overshoot**: `per_core_do_block_num = UB_budget / one_block_size`
   sized for A5's larger UB exceeds the op's actual `block_num`; kernel loads past
   GM end on small-shape cases. This is **OL-158**'s territory — fix in host tiling
   with `std::min(ub_budget_blocks, block_num)`. Distinct from #1/#2 because the
   overshoot is derived from A5 vs A3 capacity delta, not from a source-level shape
   conflation.
4. **DataCopyPad stride > actual extent**: tile inner-loop with `blockLen + stride`
   exceeding tensor end. Rarer; usually paired with a custom non-aligned tail handler.
   Fix: tile-size cap or kernel-side bound check.

**Diagnostic checklist** (when EZ9999 95/0x4 appears on A5 but A3 was clean):
- Run the same kernel on shapes where all relevant dims are EQUAL (e.g. M==N for
  2-pointcloud ops). If MTE-OOR disappears, root cause is #1 or #2.
- Inspect `verification.json` failing-case shape vs the kernel's `SetGlobalBuffer`
  extents. If extent is derived from `xyz1` dim but applied to `xyz2` GM pointer,
  confirmed #1.
- Audit `InitOutput` arg #2 against output tensor's `numel()`; mismatch → #2.

**Evidence**:
- chamfer_distance_grad kw-1 port_a3_to_a5 (2026-05-17): cases 3 (B=2 M=1 N=128)
  and 6 (B=1 M=1 N=128) triggered errcode 95 on A5. Root cause: kernel uses xyz1's
  N as both batch-stride and `InitOutput(grad_xyz2_gm, B*N*2, 0)` extent for the
  grad_xyz2 tensor whose actual size is B*M*2. Case 6's `InitOutput` wrote 256
  fp32 into a 2-fp32 allocation. Fix: pybind padding wrapper per OL-162. After
  fix: 8/8 cases PASS_WITHIN_TOLERANCE on A5 (vs A3 capture, vs CPU truth).
  Same kernel binary executes cleanly on A3 V220 for those shapes — confirms A3
  hardware silently tolerates GM-out-of-allocation while A5 V351 traps.

**Other instances (predicted)**:
- Any 2-pointcloud / 2-feature-map op family (`loss/*_grad`, `pointcloud/*`)
  whose upstream kernel was authored for symmetric-shape test drivers and never
  hardened for N≠M.
- Any op with `aclrtMemset`-on-output → `InitOutput`-extent inheritance from a
  larger sibling input.
- Any A3→A5 port where the test driver historically used a single N for all
  related tensors. Make the asymmetric-shape sweep mandatory in edge case_gen.

**Cross-reference**:
- OL-162 — pybind padding wrapper (the fix for fault #1/#2 without modifying the
  L1-verbatim kernel body)
- OL-158 — host-tiling per-core-block cap (the fix for fault #3)
- OL-160 — canonical entry-point names (the pybind wrapper lives in pybind11.cpp
  attached to the canonical `model_new_ascendc.py`)
- EC-23 — different 507035 sub-class (DataCopyPad UB→GM, not GM extent overshoot)
- EC-27 — different 507035 sub-class (SOC_VERSION default, not MTE boundary check)

<!-- 迁移自 porter kb/target/ascendc/（EC-53，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
