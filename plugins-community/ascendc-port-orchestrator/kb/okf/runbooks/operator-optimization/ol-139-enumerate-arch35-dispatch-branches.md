---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Enumerate arch35 dispatch branches before pricing Mode A vs Mode B port complexity"
description: "The presence of arch35/<op>.cpp only proves SOME dispatch exists, not that the benchmark-required variants are covered; explicitly enumerate the TPL_OPTYPE / TILING_KEY branches and cross-check the manifest before pricing, or a missing branch silently fails 8/8 cases."
original_id: OL-139
confidence: single_run
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-139, port_a3_to_a5, arch35, dispatch-coverage]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Applies to** `soc=Ascend950PR; cann=9.0.0; op_class=port_a3_to_a5`. Verified on Ascend950PR
(`fused_quant_mat_mul/op_kernel/arch35/fused_quant_mat_mul.cpp` source inspection).

The presence of `op_kernel/arch35/<op>.cpp` only proves SOME arch35 dispatch exists; it does
NOT prove the variants required by the benchmark are covered. Before pricing a port at
Mode A / Mode B-simple / Mode B-medium / Mode B-complex per OL-134, EXPLICITLY enumerate the
`if (TPL_OPTYPE == ...)` (or `TILING_KEY_IS(...)` per OL-136) branches inside
`arch35/<op>.cpp` and cross-reference them with the benchmark/test manifest.

If the benchmark needs variant X and arch35 has no branch for X, AND the V220 implementation
of X is wrapped in `#if __CCE_AICORE__ == 220` (i.e. not directly portable), the actual port
complexity is Mode B-complex regardless of how thin the host-side wiring looks.

**The trap this guards against**: a partially-staged arch35 directory FEELS like "70% done —
just need host wiring", but if the missing activation/variant requires authoring a new arch35
epilogue chain (600-1000 LOC of reg-based MicroAPI), the host-side work is 15% of the
remaining effort, not 85%. A partial arch35 with NO dispatch branch for the benchmark-required
variant produces a zero-output kernel that silently fails 8/8 cases.

### Concrete anchor (run in Phase A analysis.md, after the OL-134 estimator command)

```bash
op=<op_name>
op_path=~/workspace/cann/ops-nn/<cat>/${op}

# 1. List arch35 dispatch branches (FusedOpType / TPL_OPTYPE / TILING_KEY enumerations)
grep -nE "(TPL_OPTYPE *== *F_OPTYPE_|FusedOpType::|TILING_KEY_IS\()" \
    ${op_path}/op_kernel/arch35/*.cpp 2>/dev/null

# 2. List V220 dispatch branches for the same op (the universe of variants)
grep -nE "(TPL_OPTYPE *== *F_OPTYPE_|FusedOpType::|TILING_KEY_IS\()" \
    ${op_path}/op_kernel/*.cpp 2>/dev/null

# 3. Read the benchmark/test manifest to learn which variant(s) are actually exercised
cat ${BENCHMARK_ROOT}/${op}/cases.json | jq '.[].activation, .[].tiling_hint' | sort -u

# 4. Check whether missing-from-arch35 variants are guarded by V220-only macros
grep -nE "EpilogueTypeTraits<FusedOpType::(GELU|...)>|__CCE_AICORE__ *== *220" \
    ${op_path}/op_kernel/*.h
```

### Evidence

- `fused_quant_mat_mul` (2026-05-13, kw-1 resume): `arch35/fused_quant_mat_mul.cpp` had 4
  KERNELTYPE template instantiations all hard-coded `FusedOpType::RELU` + an
  `__NPU_ARCH__ == 5102`-guarded SWIGLU... (source text truncated here). The benchmark-required
  variant had no dispatch branch, yielding a zero-output kernel that silently failed all cases.

Source: `fused_quant_mat_mul` port spawn (kw-1 + resume, 2026-05-13). Unverified on any future
arch35 port where the kernel-side dispatch table is shape-driven rather than
activation-driven (same principle applies but the enumeration axis changes).
