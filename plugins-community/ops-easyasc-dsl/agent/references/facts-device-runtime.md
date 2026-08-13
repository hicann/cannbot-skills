# Device and Runtime Facts

Use this file for device caps, pipe mappings, supported `auto_sync()` pairs, and cross-side mutex signatures.
Detailed reasoning still lives in the constraint and pattern pages.

## Devices

| Resource        | a2 (`b3`)            | a3 (`Ascend910_9362`) | a5 (`950`) | a5pr (`950pr`) |
|-----------------|----------------------|-----------------------|------------|----------------|
| Cube core count | 20                   | 20                    | 32         | 28             |
| Vector core count | 40                 | 40                    | 64         | 56             |
| L0A             | 64 KB                | 64 KB                 | 64 KB      | 64 KB          |
| L0B             | 64 KB                | 64 KB                 | 64 KB      | 64 KB          |
| L0AMX           | not a public A2/A3 path | not a public A2/A3 path | 4 KB    | 4 KB           |
| L0BMX           | not a public A2/A3 path | not a public A2/A3 path | 4 KB    | 4 KB           |
| L0C             | 128 KB               | 128 KB                | 256 KB     | 256 KB         |
| BT (bias table) | 512 B total: 2 x 256 B slots | 512 B total: 2 x 256 B slots | 4 KB total: 2 x 2 KB slots | 4 KB total: 2 x 2 KB slots |
| UB              | 192 KB per sub-block | 192 KB per sub-block  | 256 KB; 216 KB when a kernel contains SIMT | 256 KB; 216 KB when a kernel contains SIMT |
| L1              | 512 KB               | 512 KB                | 512 KB     | 512 KB         |
| Vec sub-blocks per cube core | 2       | 2                     | 2          | 2              |

The A3 facade is intentionally the exact `Ascend910_9362` profile observed on
the 910C validation machine: 20 cube and 40 vector workers. `ascend910_93` is a
compile-unit family, not a universal core-count promise; other 93-family SoCs
such as 938x/939x can expose 24 cube and 48 vector workers.

`950` and `950pr` share the same A5/C310 instruction and codegen family; only
the default core count, vector lane count, and CANNSIM/debug SoC string differ.

Source: `easyasc/a2.py` (a2 defaults), `easyasc/a3.py` (9362 overrides), `easyasc/a5.py` (a5 overrides),
`easyasc/a5pr.py` (a5pr core-count override), `easyasc/device_profile.py` (device profile table),
`easyasc/simulator/config.py:6-19` (core count logic), and
`easyasc/utils/capacity.py` (SIMT UB cap override). When SIMT is present, the
remaining a5 UB space is reserved for SIMT functionality.

BT slots store fp32 or int32 bias values. One slot therefore holds 64 elements
on A2/A3 and 512 elements on A5/A5PR. Shortcut matmul and conv2d bias must fit one
slot: A2/A3 tile N/Cout is at most 64; A5/A5PR is at most 512. `nosplit` and
`splitk` validate the full N, while `splitn` validates its declared static tile
N. These BT limits do not waive independent L0/L1/Cube tiling constraints.

A2/A3 `gm_to_ub_pad` and `ub_to_gm_pad` encode `n_burst` in a 12-bit field and
require the resolved value in `[0, 4095]`. The stubs reject known static values
and the simulator repeats the check after resolving a dynamic `Var`. A5/A5PR do
not inherit this C220 restriction.

## Pipe / op mapping (used by `auto_sync()`)

| Pipe | Ops |
|------|-----|
| MTE2 | `gm_to_l1_nd2nz`, `gm_to_l1_pad`, `set_constant_to_l1`, `gm_to_ub_pad` |
| MTE1 | `l1_to_l0`, `l1_to_l0_mx`, `l1_to_bt` |
| M    | `mmad`, `mmad_mx` |
| FIX  | `l0c_to_gm_nz2nd`, `l0c_to_l1`, `l0c_to_ub` |
| MTE3 | `ub_to_gm_pad`, `ub_to_l1`, `ub_to_l1_nd2nz`, `ub_to_l1_nz` |
| V    | remaining vec ops |

Supported `auto_sync()` pipe pairs:
- vec: `MTE2 -> V` (`ubin`), `V -> MTE3` (`ubout`), `MTE2 -> MTE3`
  (`ubrelay`, when no V ops are present)
- cube: `MTE2 -> MTE1` (`l1`), `MTE1 -> M` (`l0`), `M -> FIX` (`fix`)

Source: `easyasc/simulator/bridge.py:24-50` (`_CUBE_PIPE_BY_OP`, `_VEC_PIPE_BY_OP`),
`easyasc/parser/asc_autosync.py:873` (`_insert_autosync_node`; pair-name logic at `884-919`).

Important reminders:
- there is no `V -> MTE2` pair
- on C220 devices (A2/A3), CrossCore wait APIs only accept `Pipe.S`:
  `allcube_wait`, `allvec_wait`, `wait_cube`, `wait_vec`, and
  `intracore_allvec_wait`; a5 can attach these waits to explicit pipe queues
- on a5/C310 devices, CrossCore set/ready APIs do **not** accept `Pipe.S` /
  `Pipe.ALL`; use a real hardware pipe. Cube-side ready/set calls
  (`cube_ready`, `allcube_ready`, and cube-end mutex pipes) must not use
  `Pipe.MTE3`; vector-side ready/set calls may use `Pipe.MTE3`. CrossCore wait
  APIs may still use `Pipe.S`.
- `dup()` before `gm_to_ub_pad` is not auto-ordered
- `l0c_to_l1` is a real FIX-side republish path, not just a codegen detail
- practical consequence: when one matmul's `L0C` result feeds a later cube-side matmul, and the
  intermediate value does not need vec-side UB math first, prefer direct `L0C -> L1` reuse over
  detouring through `UB`

Detail: `agent/references/constraints/sync.md`.

## Cross-side mutex patterns

- a5 cube -> vec: `CvMutex(src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)`
- a2/a3 cube -> vec: `CvMutex(src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)`; requires GM workspace bridge
- a5 vec -> cube: `VcMutex(src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)`
- a5 vec -> cube for UB->L1 handoff consumed only by `l1_to_l0`:
  `VcMutex(src_start_pipe=Pipe.MTE3, src_end_pipe=Pipe.MTE3, dst_start_pipe=Pipe.MTE1, dst_end_pipe=Pipe.MTE1)`
  is valid and releases the L1 slot as soon as the MTE1 consumer finishes
- a2/a3 vec -> cube: same `VcMutex` signature, but the path still goes through GM workspace

On C220 devices (A2/A3), `CvMutex` / `VcMutex` `src_start_pipe` and `dst_start_pipe`
must both remain `Pipe.S`. The end-pipe arguments above still describe where
the ready/free signals attach. Place `lock()` / `wait()` immediately before the
protected instruction on C220 so the main-loop wait does not block unrelated
work too early. Enforcement lives in `easyasc/stub_functions/crosscore.py` and
`easyasc/utils/mutex.py`.

Source: `agent/example/kernels/a5/matmul/matmul_abs_add1_vf.py:19`, `agent/example/kernels/a5/pipeline_patterns/vec_cube_abs_sqrt_matmul.py:21`,
`agent/example/kernels/a5/gdn_legacy/delta_h_psudo_state_bridge_c8.py:137-141`, `agent/example/kernels/a2/attention/flash_attn_full.py`.

### Mutex `depth` — the handoff credit count, NOT a no-op

`depth` (default `2` on both `CvMutex` and `VcMutex`) is the number of initial
capacity tokens for the intra-core producer→consumer handoff: how many producer
beats may be in flight before the consumer frees a slot (the semaphore initial
credit). It is not a tuning hint and is never ignored.

It *looks* useless because it never appears in the kernel body next to
`lock()` / `ready()` / `wait()` / `free()`. It is consumed only in the
auto-emitted prologue/epilogue (`easyasc/kernelbase/kernelbase.py:258-278`):
the prologue pre-publishes `depth` free-tokens (so the first `depth` `lock()`
calls do not block) and the epilogue drains exactly `depth`. Because both stay
balanced, a wrong `depth` raises **no** "unbalanced events" warning.

Set `depth` = the number of physical slots the producer rotates through:

| Buffer family the handoff rotates through | depth |
|-------------------------------------------|-------|
| plain `Tensor` (one fixed view)           | 1     |
| `DBuff`                                    | 2     |
| `TBuff`                                    | 3     |
| `QBuff`                                    | 4     |

- `depth > slots` → producer runs ahead onto a slot the consumer still holds =
  **silent overwrite race** (counts stay balanced, the simulator may not flag it).
- `depth < slots` → safe, but the producer stalls earlier than necessary.
- The default `2` is correct only for a `DBuff` handoff; drop to `1` for a
  single fixed view. A lag-`N` publish pipeline needs `depth = N+1` slots — e.g.
  `VcMutex(..., depth=3)` for a lag-2 P-publish rotating a 3-slot P family
  (`agent/example/kernels/a5/attention/pfa/qk_softmax_pv_flat.py:225`).

Full escalation checklist (which other local buffers must deepen when you raise
a mutex depth) → `agent/references/constraints/sync.md` §6.

Pattern references:
- a2 mixed pipeline: `agent/references/patterns/a2-mixed-pipeline.md`
- a5 mixed pipeline: `agent/references/patterns/a5-mixed-pipeline.md`

## Same-core vec-lane barrier

Use `intracore_allvec_ready(flag_id, pipe=...)` /
`intracore_allvec_wait(flag_id, pipe=...)` when only the two vec lanes inside
one core must rendezvous. Codegen maps this to CANN cross-core flag mode `0x1`
(`INTRACORE_ALLVEC_*` in `easyasc/resources/tensorutils.h`), so other cores do
not participate. This is narrower than `allvec_ready` / `allvec_wait`, whose
mode `0x0` waits for every vec participant across all active cores.
