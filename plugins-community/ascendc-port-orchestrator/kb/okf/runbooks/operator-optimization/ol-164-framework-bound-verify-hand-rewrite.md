---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Framework-bound target kernels are advisory: generate two task-owned artifacts"
description: "When target prior art is framework-bound, generate task-owned ship and verify artifacts from the selected arch22 contract and validate them against independent truth."
confidence: single_run
original_id: OL-164
classified_by: llm-assisted
timestamp_inferred: true
tags: [dual-output, optimization, ol-164, port-a3-to-a5, scheduling-framework, verify-artifact]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

Applies to `soc=Ascend950PR, cann=9.0.0, op_class=port_a3_to_a5, phase=B+C`.

**Principle**: In arch22→arch35 migration, the **ship artifact** (PR4778 layout:
`workspace/<op>/op_host/` + `op_kernel/`) and the **verify artifact**
(`workspace/<op>/kernel/`) may need different integration shapes whenever target prior art is bound
to a non-portable scheduling framework. Both must be generated from the selected arch22 contract;
neither may be a verbatim target mirror.

**Why they diverge**: the verify path is the canonical entry-point contract (OL-160) — an
`extern "C" __global__ __aicore__` kernel launched with raw `__gm__` pointer args + tiling struct,
invoked from pybind11 via `ACLRT_LAUNCH_KERNEL`. Upstream's arch35 kernel is bound to an
op-family-wide DAG framework (`atvoss/elewise/elewise_sch.h::ElementwiseSch<schMode, OpDag>`, or
family equivalents under `atvoss/reduce/`, `atvoss/scan/`, ...) that requires the full `ops-nn-port`
host-tiling registration to resolve (`<Op>TilingData` + `<Op>Tiling` + `op_def.cpp` registration
must exist before the kernel can be instantiated). `build_ascendc.py` does NOT run the `ops-nn-port`
pipeline — it only compiles what is under `workspace/<op>/kernel/`. So the framework-bound kernel is
structurally unable to serve the verify path.

**Layered rule**:
- **Ship artifact**: use target prior art only for public-API, layout, lifecycle, and coverage
  hypotheses; emit task-owned code from the selected source contract.
- **Verify artifact**: independently author the same declared math in the canonical verification
  shape; it must exercise the current generated kernel.
- **Equivalence proof**: migration uses selected-arch22 source NPU truth; backward uses gradient
  math, saved-tensor contract, and CPU fp64 autograd. Target output is diagnostic only.

**Activation condition** (when this rule fires):
- the arch22→arch35 migration route is active, AND
- upstream `op_kernel/arch35/<op>.h` (or the shared-common variant per OL-141) `#include`s a
  scheduling-framework header (`atvoss/elewise/elewise_sch.h`, or a family-specific equivalent).
