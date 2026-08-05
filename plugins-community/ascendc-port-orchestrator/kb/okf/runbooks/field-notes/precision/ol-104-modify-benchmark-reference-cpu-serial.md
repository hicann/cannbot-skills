---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Parallel-vs-serial conflict — modify the BENCHMARK reference to a CPU serial impl aligning with the kernel's algorithm"
description: "When MERE/MARE failure stems from parallel-NPU-reference vs serial-AscendC-kernel (cumsum/scan) rather than ULP noise, the fix can be to change the benchmark reference (model.py forward) to a CPU serial impl — stays CPU-truth while picking the CPU algorithm the kernel matches."
phenomenon: precision_issue
signal:
  - "kernel implements an inherently serial algorithm (cumsum, scan) while the NPU torch reference uses a parallel scan with a different reduction tree"
  - "both the kernel AND CANN diverge from CPU PyTorch (Q4 of the 4-quadrant sweep); the algorithm is correct but the metric is unrealistic against the parallel impl"
confidence: single_run
original_id: OL-104
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, precision, ol-104, cumsum, scan, benchmark-reference]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发

MERE/MARE failures that stem from **algorithmic divergence** (parallel-vs-serial) rather than ULP-level noise:
- Kernel's natural implementation is serial (e.g. AscendC cumsum doing left-to-right accumulation).
- NPU torch reference uses a parallel scan with a different reduction tree.
- The mismatch is inherent in the algorithm, not in numerical noise — the kernel can't fix it without losing the serial benefit.

## 根因 / 教训

The right fix is sometimes to **modify the BENCHMARK reference** (model.py forward) to use a CPU serial implementation that matches what the kernel produces. This stays "CPU truth" (CPU is the reference) while picking the CPU algorithm that aligns with the kernel.

### Concrete example (cumsum, br_430 PR #149)

```python
# model.py forward — modified reference
def forward(self, x, dim):
    # NPU torch.cumsum may use parallel scan; use CPU serial via numpy
    # to match the AscendC kernel's natural left-to-right algorithm
    if x.dtype == torch.float32:
        out = np.cumsum(x.cpu().numpy(), axis=dim, dtype=np.float32)
        return torch.from_numpy(out).to(x.device)
    if x.dtype == torch.float16:
        out = np.cumsum(x.cpu().numpy(), axis=dim, dtype=np.float16)
        return torch.from_numpy(out).to(x.device)
    return torch.cumsum(x, dim=dim)  # bf16 NPU scan happens to be deterministic
```

### When to apply

- Op is a scan/reduction-class where parallel-vs-serial fundamentally differ.
- Per-step rounding behaviors don't match (one direction loses; the kernel can't recover).
- BOTH the kernel AND CANN diverge from CPU PyTorch (Q4 in the 4-quadrant sweep).
- The algorithm itself is correct; the metric is unrealistic against the parallel impl.

### When NOT to apply

- Ops where CANN bit-matches CPU PyTorch (Q1/Q2/Q3) — fix the kernel instead.
- Ops where the issue is bit-level rounding chain (use OL-101 host precompute or OL-102 per-op chain).
- Ops where modifying the reference would silently mask a real kernel bug.

### Process consideration

- Modifying the benchmark requires an upstream PR to `Just-it/AscendOpGenAgent` (or our fork).
- Document the modification in the commit message AND in the modified model.py docstring.
- Require explicit reviewer approval (this changes the test contract).

Cross-reference: the workbench's `AscendCVerification.md` documents the cumsum dim=0 fp16 NPU non-… (source text truncated in the batch excerpt).

Precision-audit (CPU-truth, 2026-04-29): VALIDATED-CPU — `Just-it/AscendOpGenAgent` br_430 commit `540381a` "对齐CPU标杆，优化cumsum算子精度".
