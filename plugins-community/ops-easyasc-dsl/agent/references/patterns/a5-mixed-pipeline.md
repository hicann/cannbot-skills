# A5 Mixed Cube/Vec Pipelines

## Applies when

Use this file when an A5 kernel crosses cube and vec through direct on-chip
handoffs.

A5 has on-chip cube/vec handoffs: `l0c_to_ub` for cube→vec and
`ub_to_l1` / `ub_to_l1_nd2nz` / `ub_to_l1_nz` for vec→cube.
Unlike a2, no GM workspace bridge is needed between cube and vec stages.
For a2 variants, see `agent/references/patterns/a2-mixed-pipeline.md`.

Related: `agent/references/constraints/a5.md`, `agent/references/constraints/sync.md`.

## Logical dataflow families

| Section | Dataflow |
| --- | --- |
| 1 | cube -> vec postprocess |
| 2 | vec -> cube preprocess |
| 3 | vec -> cube -> vec fusion |
| 4 | cube -> vec -> cube -> vec lookahead streaming |

Read one matching section. The generic delayed-stage loop and slot-depth model
live in `agent/references/patterns/lookahead-drain.md`.

## Shared physical invariants

- choose ND/NZ publish layout from the cube consumer, not from host convenience;
- keep each cross-side handoff on its own mutex/counter family;
- bind outer ownership so active cube and vec participants execute matching
  handshakes;
- tighten a mutex lifetime only after proving the last physical consumer;
- size local slots from delayed live beats and roles, not from the number of
  source variables.

---

## 1. Cube → Vec

Use when cube computes the main tile and vec applies elementwise or row-wise postprocessing before final writeback.

### Minimal flow

```
GM -> L1 -> L0 -> L0C -> UB -> @vf -> GM
```

### Ownership rule

`CvMutex(n, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)`

Verified in:
- `agent/example/kernels/a5/matmul/matmul_abs_add1_vf.py:19`
- `agent/example/kernels/a5/matmul/matmul_rowwise_norm.py:31`
- `agent/example/kernels/a5/matmul/matmul_half_splitn_bias10p2_vf.py:23`

### What usually matters

- where the cube tile becomes a vec-visible UB tile
- whether the postprocess runs in `float` or after a cast
- standard half-row vec writeback for tails
- separate counters for longer postprocess lifetimes

### Reference kernels

- `agent/example/kernels/a5/matmul/matmul_abs_add1_vf.py` — baseline
- `agent/example/kernels/a5/matmul/matmul_rowwise_norm.py`
- `agent/example/kernels/a5/matmul/matmul_half_splitn_bias10p2_vf.py`
- `agent/example/kernels/a5/matmul/matmul_mknk_2dgrid_splitk_add1.py`

---

## 2. Vec → Cube

Use when vec preprocessing transforms data before cube consumes it in a later matmul stage.

### Minimal flow

```
GM -> UB -> @vf -> UB -> L1 -> L0 -> L0C -> GM
```

### Ownership rule

Default to the conservative full cube-stage lifetime:

```python
vcmutex = VcMutex(n, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
```

Only tighten the consumer side to MTE1 after proving the mutex protects an L1
slot whose final consumer is `l1_to_l0`. This applies to L1 slots published by
`ub_to_l1` / `ub_to_l1_nd2nz` / `ub_to_l1_nz` and not read by later `M` / `FIX` work:

```python
vcmutex = VcMutex(
    n,
    src_start_pipe=Pipe.MTE3,
    src_end_pipe=Pipe.MTE3,
    dst_start_pipe=Pipe.MTE1,
    dst_end_pipe=Pipe.MTE1,
)
```

Do not use the tightened form as the default Vec→Cube template. Treat it as a
trace-driven synchronization optimization; see
`agent/references/constraints/sync.md` for the full lifetime rule and b-series
start-pipe limitation.

Verified in:
- `agent/example/kernels/a5/pipeline_patterns/vec_cube_abs_sqrt_matmul.py:21`
- `agent/example/kernels/a5/gdn_legacy/recompute_wu.py:49`

### What usually matters

- whether the publish path is ND or NZ
- whether the host-side layout stays reshape-only
- how subblock rows are split between vec sides
- whether the preprocessed value must remain in half or float before cube consume
- mixed-stage tile ownership must cover both sides of the handshake. If a stage
  contains cube work or a `VcMutex`, drive the outer tile loop with
  `GetCubeIdx()` / `GetCubeNum()` or otherwise prove that cube and vec lanes
  execute matched `lock` / `ready` / `wait` / `free` calls on the same active
  cores. A vec-only owner around a later cube consumer can leave inactive cube
  lanes waiting on a flag that no vec lane will publish.
- when publishing a UB result to L1 with `ub_to_l1_nd2nz`, make the source
  stride explicit in your mental model. `N_src` defaults to the source UB
  tensor's physical second dimension, so a subview cut out of a wider matrix
  keeps that wider row stride instead of becoming a compact operand. If the
  cube consumer expects a compact logical operand, first produce or copy the
  block into a dedicated compact UB tile. The stub records `N_src` and the
  simulator reads the source as a strided 2-D view
  (`easyasc/stub_functions/vec/datamove.py:207-260`,
  `easyasc/simulator/pipe_vec.py:299-345`).

### Reference kernels

- `agent/example/kernels/a5/pipeline_patterns/vec_cube_abs_sqrt_matmul.py`
- `agent/example/kernels/a5/pipeline_patterns/vec_cube_abs_sqrt_matmul_nz.py`
- `agent/example/kernels/a5/gdn_legacy/recompute_wu.py`

---

## 3. Vec → Cube → Vec

Use when the kernel needs vec preprocessing, cube compute, and vec postprocessing in one fused flow.

### Minimal flow

```
GM -> UB -> @vf -> UB -> L1 -> L0 -> L0C -> UB -> @vf -> GM
```

### Ownership rules

Two cross-side handoffs — keep them conceptually separate:

```
vcmutex = VcMutex(0, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
cvmutex = CvMutex(1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
```

Verified in:
- `agent/example/kernels/a5/pipeline_patterns/vec_cube_vec_scale2_abs_add1_matmul.py:30-32`
- `agent/example/kernels/a5/conv/conv2d_relu_n1c2h8w8_fixed.py:102-103`

### What usually matters

- separate stage counters for each handoff
- clear stage ownership — do not let one counter blur across both handoff directions
- avoiding accidental buffer lifetime reuse across both handoff directions
- validating each stage before building the whole fusion

### Reference kernels

- `agent/example/kernels/a5/pipeline_patterns/vec_cube_vec_scale2_abs_add1_matmul.py`
- `agent/example/kernels/a5/conv/conv2d_relu_n1c2h8w8_fixed.py`

---

## 4. Cube → Vec → Cube → Vec (lookahead streaming)

Use when two cube stages are interleaved with vec logic across iterations, requiring a one-tile lookahead schedule. This is the pattern for multi-head attention IFA and flash attention.

### Minimal flow

```
cube stage 1 -> vec stage 1 -> cube stage 2 -> vec stage 2
(one-tile lookahead: producer runs at s, consumer runs at s-1)
```

### Ownership rules

Three mutexes for the two cube→vec and one vec→cube handoffs:

```
qk_mutex = CvMutex(0, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
p_mutex  = VcMutex(1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
pv_mutex = CvMutex(2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
```

Verified in:
- `agent/example/kernels/a5/attention/mha_ifa.py:105-107`
- `agent/example/kernels/a5/attention/mha_ifa_nz.py:113-115`
- `agent/example/kernels/a5/attention/test_mla_entire.py:107-109`
- `agent/example/kernels/a5/attention/flash_attn_full_fp8_causal.py:204-206` (uses default `src_end_pipe`)

### Lookahead control skeleton

```python
for s in range(0, S + TILE, TILE):
    if s < S:   # producer side: warmup + steady state
        ...
    if s > 0:   # delayed consumer side: steady state + drain
        ...
```

### What usually matters

- keeping producer and delayed consumer lifetimes separate
- within each S-tile iteration, publish current QK/P **before** consuming the
  previous tile's PV/accumulation stage; this keeps the producer one tile ahead
  of the consumer and preserves the one-extra-iteration drain
- delayed scalar state (`row_max`, `row_sum`, rescale factors) must follow the consumer counter, not the producer counter
- if stage 2 reuses a stage 1 operand one iteration later, keep that operand on chip
- in `mha_ifa_nz.py`, the vec-produced `p` tile is packed with `reg_to_ub(...)` then published to `l1p` as `.nz()` before cube consumes it
- for fp8 decode: mask invalid tail columns to `-1.0e30` (finite large-negative sentinel) before `rowmax`; scale probability tile only after the float `row_sum` update; compensate with `scale_v / P_SCALE`
- if stage 2 lifetime does not fit on chip, materialize an explicit GM workspace rather than forcing a false on-chip story

### Reference kernels

- `agent/example/kernels/a5/attention/mha_ifa.py`
- `agent/example/kernels/a5/attention/mha_ifa_256.py`
- `agent/example/kernels/a5/attention/mha_ifa_fp8_scale_256.py`
- `agent/example/kernels/a5/attention/mha_ifa_nz.py`
- `agent/example/kernels/a5/attention/mha_ifa_nz_256.py`
- `agent/example/kernels/a5/attention/flash_attn_full_fp8_causal.py`
- `agent/example/kernels/a5/attention/test_mla_entire.py`

## Do not use when

- the target is A2 and must cross cube/vec through GM workspace;
- the formula is cube-only or vec-only;
- an apparent fusion would require a second runtime kernel or host semantic
  work that changes the contract.

## Source escape

For an uncovered direct handoff, layout, or event lifetime, follow
`agent/references/evidence-escalation.md`. Inspect the exact public facade,
datamove lowering, simulator path, and generated C++ for one isolated handoff
before generalizing a new Pattern.
