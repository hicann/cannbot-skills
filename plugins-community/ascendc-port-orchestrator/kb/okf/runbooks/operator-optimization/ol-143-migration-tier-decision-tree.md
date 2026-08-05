---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "L1/L2/L3/L4 migration tier decision tree for port_a3_to_a5"
description: "Before invoking the kernel-worker for a port_a3_to_a5 op, classify it into a mutually-exclusive migration tier (L1 basic adapt / L2 MicroAPI register rewrite / L3 SIMT / L4 escalate) evaluated top-down; the tier fixes the change scope and which references the worker loads."
original_id: OL-143
confidence: single_run
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-143, port_a3_to_a5, migration-tier, phase-o25]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

Before invoking the kernel-worker for a `port_a3_to_a5` op (post-prior-art-verify FAIL or
no-upstream path), CLASSIFY the op into a migration tier. The tier determines what changes are
required and which references the worker must load. Mis-tier means burning iterations to
discover the structural mismatch the classifier would have caught.

**Applies to** `soc=Ascend950PR; cann=9.0.0; op_class=all_port_a3_to_a5; phase=O2.5`.
Verified on Ascend950PR / cann 9.0.0. Source: PR 103 (Ascend/agent-skills)
`ascendc-operator-A5-migration` SKILL.md §8-42.

### Tier definitions (mutually exclusive, evaluated top-down)

| Tier | Trigger (any of these) | Change scope |
|---|---|---|
| **L2: MicroAPI Register-based rewrite** | (a) Perf-critical hot-path op (RMSNorm / RoPE / Softmax / Attention); OR (b) complex quant Cast chain (FP32 → FP8 / HiFloat8 / INT8 in same kernel); OR (c) needs overflow-mode control (RMSNorm / Softmax); OR (d) A5 new dtype (FP8 / HiFloat8 Cast unavoidable, needs MicroAPI CastTrait) | `_def.cpp` + `_apt.cpp` + `arch35/` + `CMakeLists.txt` + `config/` + ALL core compute paths rewritten with `__VEC_SCOPE__` + `RegTensor` + `MaskReg` |
| **L3: SIMT optimization** | NOT L2, AND ALL of: (a) has Scatter/Gather (index-based GM r/w); (b) index logic simple (no nested computation); (c) no UB transit needed; (d) high thread parallelism (data volume ≫ 2048 elements) | L1 baseline + new SIMT kernel file in `arch35/`, gated by `__NPU_ARCH__ == 3510`, using `__simt_vf__` + `LAUNCH_BOUND(2048)` + `Simt::GetThreadIdx/GetThreadNum` |
| **L1: Basic adapt (default)** | None of the above | `_def.cpp` independent config + `_apt.cpp` entry + `arch35/` copy with BF16 guard removal + `config/ascend950/{binary.json, simplified_key.ini}` |
| **L4: Escalate** | Tiling needs `IsRegbaseSocVersion()` decision; OR UB underprovisioned for required 40KB SIMT DCache | Out of kw scope → route to researcher |

### Concrete checklist (kw_brief Phase A should apply)

```python
def classify_tier(op_dir: Path, analysis: dict) -> str:
    """Return one of 'L1', 'L2', 'L3', 'L4' for an A3→A5 port candidate."""
    kernel_src = read_all(op_dir / "op_kernel" / "*.h", "*.cpp")
    def_src = read(op_dir / "op_host" / f"{op}_def.cpp")

    # L2 triggers
    if any([
        analysis["op_class"] in ("rmsnorm", "rope", "softmax", "attention"),
        re.search(r"Cast<\s*(fp8_e4m3fn_t|fp8_e5m2_t|hifloat8_t)", kernel_src),
        re.search(r"ReduceSumCustom|DataCopyPad", kernel_src),
        re.search(r"DT_FLOAT8|DT_HIFLOAT8", def_src),
    ]):
        return "L2"

    # L3 triggers (must satisfy ALL)
    if all([
        re.search(r"Gather|Scatter|IndexCopy|IndexPut", kernel_src),
        analysis.get("index_logic_simple", False),
        analysis.get("data_volume_large", False),
    ]):
        return "L3"

    # L4 triggers
    if re.search(r"IsRegbaseSocVersion|UB...", ...):   # (source text truncated here)
        return "L4"
    ...
```

Note: the source text was truncated inside the L4 branch of `classify_tier`; the tail of the
checklist function is reproduced only as far as the source provided.
