---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Reference output dtype comes from Model.forward body, not input dtype — read the source before allocating"
description: "When the Python reference promotes to fp32 internally and returns fp32 regardless of input dtype, the kernel must allocate fp32 outputs; the input-dtype heuristic fails silently as an entire fp16/bf16 band with MERE just over threshold."
phenomenon: precision_issue
signal:
  - "Per-dtype-band failure shows dtype(ref=torch.float32, cand=torch.float16/bf16) with MERE just above threshold across an entire fp16/bf16 band while all fp32 cases PASS"
confidence: single_run
original_id: OL-118
classified_by: llm-assisted
timestamp_inferred: true
tags: [precision, kernel-design, ol-118, output-dtype, pybind]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发
An entire fp16 or bf16 dtype band fails with `dtype(ref=torch.float32, cand=torch.float16/bf16)` mismatch and **MERE just above threshold**, while all fp32 cases PASS. This is the smoking gun: the MERE is small because the math is correct — the mismatch is purely fp32→fp16/bf16 storage rounding on the kernel side. fp32 input cases PASS only because their output dtype happens to match. Loaded by aog-kernel-worker (Phase A output-dtype decision, Phase B pybind allocation) and aog-precision-probe.

## 根因 / 教训
A regen worker may default to "outputs match input dtype". That is **wrong** when the Python reference explicitly promotes to fp32 internally (`x.float()`, `torch.zeros_like(x, dtype=torch.float32)`) and returns fp32 regardless of input dtype. The kernel must match the reference's *output* dtype, which is independent of input dtype. Reading the reference source is the ONLY way to know — the input-dtype heuristic fails silently.

### Concrete anchor
```python
# Read this before deciding kernel output dtype:
def forward(self, ..., max_state, num_state, den_state):
    max_state = max_state.clone().float()                # state promoted to fp32
    output = torch.zeros_like(key, dtype=torch.float32)  # output explicitly fp32
    # ... loop computes in fp32 ...
    return output, max_state, num_state, den_state       # ALL outputs fp32
```
```cpp
// pybind11.cpp — allocate ALL outputs as fp32 unconditionally
auto output   = torch::empty({B, T, H}, torch::dtype(torch::kFloat32).device(...));
auto max_st_o = torch::empty({B, H},    torch::dtype(torch::kFloat32).device(...));
// kernel store path is unconditionally StoreF32 — no per-dtype Cast-back branch
```

### Detection workflow (Phase A)
1. Read the reference `Model.forward(...)` source.
2. Grep for `.float()` / `dtype=torch.float32` / `torch.zeros_like(.., dtype=torch.float32)` on: output-construction lines, state-clone lines, and return statements (any explicit Cast before return).
3. If present anywhere on the output path → outputs are fp32 unconditionally. Allocate kernel outputs as fp32; remove fp16/bf16 store branches.

### Evidence
- 30_TimeDecayExponentialStabilization regen 2026-05-02 (kw-1, single-iter fix):
  - Wave 1 hand-crafted (2026-04-15): 50/50 PASS — author caught this in initial design
  - Wave 2 regen-2 (pre-fix): **26/50** — all 24 fp16/bf16 cases failed with the dtype-mismatch + MERE-just-over-threshold signature; 26 fp32 cases PASS
  - Phase D iter 2 fix: pybind11 allocated all outputs as fp32 [source truncated] → recovered.
