# Deep Note: `agent/example/kernels/a2/attn_backward_dense_total_tail.py`

Open this file only after the short catalog entry confirmed the kernel is relevant.

## What this kernel is really for

- the full tail-safe a2 dense attention-backward fusion, not the smaller stage-1 or stage-1+2 teaching variants
- a path that has to survive both `S1` and `S2` tails while keeping the cube/vec/cube bridges stable

## Decisions worth copying

- keep both GM workspace bridges on full-tile shapes and push `valid_m` / `valid_n` handling to GM boundaries plus vec masks
- keep the stage-1 vec hot path chunk-local instead of reusing the old half-tile story; separate chunk-sized scratch is easier to validate
- if vec scratch growth becomes risky, prefer smaller chunks over borrowing live stage buffers
- reuse delayed `k_j` on chip for the final `gq += dqk_j @ k_j` stage instead of reloading from GM
- promote only the reused `k` operand family to `TBuff`; leave unrelated families on simpler buffering
- keep tile-level `atomic_add()` narrow and expect caller-side zero initialization

## Prefer another kernel when

- you want the smallest aligned-only backward reference
- you only need the stage-1 or stage-1+2 intermediate contract
