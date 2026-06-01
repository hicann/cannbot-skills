# Deep Note: `agent/example/kernels/a2/flash_attn_full_pj_hif8_commonub.py`

Open this file only after the short catalog entry confirmed the kernel is relevant.

## What this kernel is really for

- comparing against `flash_attn_full_pj_hif8.py` after the math contract is already understood
- studying how a shared vec-side slot buffer changes queueing structure without changing the visible formula

## Decisions worth copying

- move vec scratch from two plain `Tensor` views onto one shared `DBuff` family: `ub_score_pv + score_pv_cnt`
- keep `stage1_cnt` and `stage2_cnt` separate even though the shared scratch family exists
- treat the gain as a same-side vec `ubin` queueing improvement, not as a new cross-side ownership model
- do not expect UB-footprint reduction here; the point is cleaner overlap between the next preload and current vec compute

## Prefer another kernel when

- you are still deriving the math contract and want the simpler readable baseline
- you are debugging row-max / row-sum correctness and do not want shared vec scratch lineage in the picture yet
