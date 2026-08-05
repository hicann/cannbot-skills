---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "arch22→arch35 port: choose Mode A (macro-guard strip) vs Mode B (regbase variant) before kernel work"
description: "A5 admits two valid host-side wiring shapes when porting a V220 kernel; grep the master kernel for the __NPU_ARCH__ 3003/3113 guard to pick the shape before starting, avoiding wasted rewrites."
original_id: OL-132
confidence: single_run
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-132, port_a3_to_a5, mode-a-vs-b, host-wiring]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

A5 admits TWO valid host-side wiring shapes when porting an existing A2/A3 (V220) AscendC
kernel. Choosing the shape BEFORE kernel work begins prevents wasted rewrites.

**Applies to** `soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5; op_class=port_a3_to_a5`.
Only relevant in the V220→V351 port direction — NOT for from-scratch A3 or A2 kernel creation
(there is no "Mode A vs Mode B" decision when no V220 kernel exists yet).

### Mode A — surgical strip + pre-staged macro guard

The V220 kernel `.cpp` is kept in place, with the V351 codepath enabled by a pre-existing
`__NPU_ARCH__ == 3003 || 3113` guard inside the kernel. The host-side port is essentially
registration-only: add `AddConfig("ascend950", ...)` to `op_def.cpp`, drop in
`binary.json` + `simplified_key.ini`. No `op_kernel/arch35/` directory; no `<op>_apt.cpp`.
**Trigger**: grep the V220 kernel source for `__NPU_ARCH__ == 3003 || 3113` — if present,
Mode A is sufficient.

### Mode B — regbase parallel variant

A new `op_kernel/arch35/<op>.h` (optionally split into per-variant files) AND a new
`<op>_apt.cpp` entry-point AND `ExtendCfgInfo("opFile.value", "<op>_apt")` in the def.cpp
ASCEND950 config block. Used when the V220 kernel does NOT have the pre-staged macro guard,
OR when arch35 needs a meaningfully different algorithm (regbase MicroAPI, different tiling,
etc.).

### Mode detection anchor (grep the master V220 kernel before any port work)

```cpp
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
    // V351 codepath — already exists; port is host-side registration only
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    // V220 codepath
#endif
```

### Mode B-V220-pure sub-mode (post-2026-05-23 default-OFF arch35 rule)

When `OPGEN_PRESTAGE_ARCH35=0` (the new default), Mode B is realized WITHOUT upstream
`arch35/` artifacts. The worker authors a thin `kernels.cpp` build TU + a `pybind11.cpp`
that wraps the staged V220 algorithm headers directly (`op_kernel/*.h`), with
`#if defined(__CCE_AICORE__)` guards around kernel-side includes + bodies (EC-64) and a
minimum-fields `<Op>TilingData` struct grep'd from `tilingData->` accesses (CAND-PP101).
No `arch35/<op>.h` is created; the host-side AddConfig + `<op>_apt.cpp` is replaced by the
worker-authored build TU. (Source text truncated at this point.)

Source: cann-learner CAND-A3A5-1, promoted 2026-05-12 from the PR4778 cross-op-evidence
batch (6 ops); promotion gates C36/C37/C38/C39 (sibling op `gather_elements_v2`)/C40 (codex
unavailable, audit recorded).
