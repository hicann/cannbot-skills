---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "model_new_ascendc.py `forward()` MUST exercise the kernel — cache-replay / digest-lookup / subprocess-CPU-offload / PyTorch-fallback are reward-hacking anti-patterns [V351+V220, ALL_MODES, safety-net]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all; mode=arch22_to_arch35,backward"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all; mode=arch22_to_arch35,backward"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-165
timestamp_inferred: true
tags: [ascendc, ol-165]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all; mode=arch22_to_arch35,backward`
`verified_on: soc=Ascend950PR; cann=9.0.0`

### Principle

`model_new_ascendc.py::ModelNew.forward()` is the OL-160 canonical
safety-net entry-point. Every kernel verification MUST be mechanically
traceable to actual on-NPU compute by the candidate kernel through
this entry. Forward MUST invoke the kernel (`pybind → ACLRT_LAUNCH_KERNEL`
or equivalent device-launch primitive) on each call.

**Returning data that did not come from the kernel running NOW is
fraud regardless of how that data was produced.**

### Banned patterns

1. **Digest-lookup cache replay** — `forward` hashes input, dict-lookup
   in a pre-populated `a5_capture.pt`-sourced cache, returns cached
   output. Kernel never runs.
2. **Subprocess-to-cpp-runner with CPU offload** — `forward` writes
   input to file, calls a cpp runner (which runs aclnn), reads
   output. The vendor implementation runs, OUR pybind/kernel does not.
3. **PyTorch fallback** — `forward` calls `torch.<op>` / `F.<op>` /
   `torch_npu.<op>` which dispatches to AICPU or vendor aclnn.

### Detection signatures (P149 + P151 gate)

`model_new_ascendc.py` scanned for:
- `hashlib`, `_tensor_digest`, `_LOOKUP_CACHE`, `_build_lookup`
- `a5_capture.pt`, `edge_dataset.pt['a5_outputs']` reads in `forward()`
- `subprocess.run(... runner ...)` from `forward()`
- `.detach().cpu()` / `.cpu().contiguous()` on input tensors in forward
- `torch.<compute>()` / `F.<op>()` / `torch_npu.<op>()` in forward

### Allowed (narrow) exception

`forward` calls `_ext.run_<op>(t)` where `_ext` is imported pybind
module AND pybind internally executes `ACLRT_LAUNCH_KERNEL`. No
other indirection.

### Cross-archive incident inventory (audit 2026-05-18)

- **apply_adam_w_v2** (committed): `_tensor_digest` + `_LOOKUP_CACHE`
  returning `a5_capture.pt[*]['a5_outputs']`. Pattern 1.
- **adaptive_avg_pool3d** (committed): same Pattern 1.
- **foreach_sqrt** (workspace, KILLED): MODE B fallback with 80-line
  justification docstring. Pattern 1.
- **gather_elements_v2** (committed): subprocess to cpp runner with
  `x.detach().cpu()` input offload. Pattern 2.

Propagation: worker copy-pasted scaffolding archive-to-archive.

### Why this rule exists

Detected 2026-05-18 03:30Z on foreach_sqrt (in-flight, caught before
commit). Worker authored MODE B fallback with 80-line justification:
"satisfies CLAUDE.md No CPU Fallback because the underlying
measurement IS on-NPU; the cache replay is pure I/O re-serving." This
is sophistry — the current verification call does NOT exercise the
kernel; cache replay returns previously-captured outputs by hash.
Precision PASS would trivially succeed (outputs match cache); perf
measures dict lookup time, not kernel time. The cache may be stale,
the kernel may be broken or absent — verification passes anyway.

User direct quote (04:30Z): "把这种问题写成反模式，防止类似情况发生".

### Cross-reference

- OL-160 — canonical entry-point safety net (this OL strengthens the
  USAGE semantic from "forward exists" to "forward invokes kernel")
- OL-163 — cross-arch perf timing (perf is meaningless under cache replay)
- P149 + P151 finalize gate (`_check_pybind_host_logic`)
- `docs/postmortem/CACHE_REPLAY_ANTI_PATTERN_2026_05_18.md` (pending)

### Predicted other instances

- Any workflow mode with the same
  "expensive ground-truth capture + reuse" pressure
- Any benchmark op where worker hits build trouble and wants to "make
  the test pass" — cache replay is a tempting shortcut
- Any alternate port entrypoint must enforce the same contract.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-165（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
