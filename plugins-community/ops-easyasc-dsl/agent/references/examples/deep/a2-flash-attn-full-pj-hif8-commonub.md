# Deep Note: `agent/example/kernels/a2/attention/flash_attn_full_pj_hif8_commonub.py`

Open this file only after the short catalog entry confirmed the kernel is relevant.

## What this kernel is really for

- comparing against `flash_attn_full_pj_hif8.py` after the math contract is already understood
- studying how a depth-3 queue and three-slot expdiff scratch change scheduling without changing the visible formula family

## Decisions worth copying

- use depth-3 `CvMutex` / `VcMutex` handoffs with `stage*_slot = var_mod(..., 3)`
- extend the drain loop to `tiles_n + 2` so stage 2 consumes a tile two loop iterations later
- keep `stage1_cnt` and `stage2_cnt` separate even when both use the same modulo depth
- store row scalars in `[1, HALF_M]`, then broadcast with explicit `brcb(..., repeat=HALF_M // 8)`
- treat the gain as a queueing / scheduling variation, not as a new cross-side ownership model

## Prefer another kernel when

- you are still deriving the math contract and want the simpler readable baseline
- you are debugging row-max / row-sum correctness and do not want the depth-3 queueing variation in the picture yet
