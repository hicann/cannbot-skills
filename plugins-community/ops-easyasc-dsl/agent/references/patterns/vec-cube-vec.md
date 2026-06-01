# Vec-to-Cube-to-Vec Pattern

> Generic baseline only. For a2 (b3) kernels, prefer the a2-specific patterns under `agent/references/patterns/` (e.g., `a2-cube-vec.md`, `a2-cube-vec-cube.md`) and read `agent/references/constraints/a2-device.md` for device-side rules.

Read this file when the kernel needs vec preprocessing, cube compute, and vec postprocessing in one fused flow.

## Use this pattern when

- input data needs a vec-side transform first
- cube does the main matmul or cube-heavy stage
- vec performs the final output transform

## Minimal flow

`GM -> UB -> @vf -> UB -> L1 -> L0 -> L0C -> UB -> @vf -> GM`

## Ownership rule

This pattern has two cross-side handoffs:
- vec -> cube needs `VcMutex`
- cube -> vec needs `CvMutex`

Keep those two ownership edges conceptually separate.
Do not let one counter or one fuzzy stage boundary blur them together.

## What usually matters most

- separate stage counters
- clear stage ownership
- avoiding accidental reuse of one buffer lifetime across both handoff directions
- keeping validation incremental instead of building the whole fusion at once

## Typical files to study

- `agent/example/kernels/a5/vec_cube_vec_scale2_abs_add1_matmul.py`
