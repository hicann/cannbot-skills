# Deep Note: `agent/example/kernels/a2/flash_attn_full_pj_hif8.py`

Open this file only after the short catalog entry confirmed the kernel is relevant.

## What this kernel is really for

- the scaled-hif8 probability variant of normalized a2 online softmax
- a contract that intentionally changes the delayed value path while keeping `row_sum` in float
- a kernel that exports final `rowmax` / `rowsum` as part of the visible contract

## Decisions worth copying

- update `row_sum` from the float `p_j` tile before any `half` / hif8 cast
- keep stage-1 score scratch and stage-2 `pv` scratch separate in the readable baseline
- implement the non-negative hif8 simulation without relying on unsupported `uint8 -> float` shortcuts
- copy `[M,64]` score slices into contiguous scratch before `reinterpret(...)` when the quantized helper needs contiguous lanes
- handle non-aligned `S2` in score space with suffix invalidation and a sufficiently negative finite sentinel
- handle non-aligned `S1` separately from `S2`; invalid rows should become zero contribution to delayed `p @ v` while GM still writes only valid rows

## Prefer another kernel when

- you still want the plain `p.half().float()` value path
- you are debugging the normalized float/half baseline before introducing hif8 behavior
