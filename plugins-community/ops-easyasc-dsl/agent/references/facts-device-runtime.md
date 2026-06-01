# Device and Runtime Facts

Use this file for device caps, pipe mappings, supported `auto_sync()` pairs, and cross-side mutex signatures.
Detailed reasoning still lives in the constraint and pattern pages.

## Devices

| Resource        | a2 (`b3`)            | a5 (`950`) |
|-----------------|----------------------|------------|
| Cube core count | 20                   | 32         |
| L0A             | 64 KB                | 64 KB      |
| L0B             | 64 KB                | 64 KB      |
| L0C             | 128 KB               | 256 KB     |
| UB              | 192 KB per sub-block | 256 KB     |
| L1              | 512 KB               | 512 KB     |
| Vec sub-blocks per cube core | 2       | 2          |

Source: `easyasc/globvars.py` (defaults), `easyasc/a5.py` (overrides).

## Pipe / op mapping (used by `auto_sync()`)

| Pipe | Ops |
|------|-----|
| MTE2 | `gm_to_l1_nd2nz`, `set_constant_to_l1`, `gm_to_ub_pad` |
| MTE1 | `l1_to_l0` |
| M    | `mmad` |
| FIX  | `l0c_to_gm_nz2nd`, `l0c_to_l1`, `l0c_to_ub` |
| MTE3 | `ub_to_gm_pad`, `ub_to_l1_nd2nz`, `ub_to_l1_nz` |
| V    | remaining vec ops |

Supported `auto_sync()` pipe pairs:
- vec: `MTE2 -> V` (`ubin`), `V -> MTE3` (`ubout`)
- cube: `MTE2 -> MTE1` (`l1`), `MTE1 -> M` (`l0`), `M -> FIX` (`fix`)

Important reminders:
- there is no `V -> MTE2` pair
- `dup()` before `gm_to_ub_pad` is not auto-ordered
- `l0c_to_l1` is a real FIX-side republish path, not just a codegen detail
- practical consequence: when one matmul's `L0C` result feeds a later cube-side matmul, and the
  intermediate value does not need vec-side UB math first, prefer direct `L0C -> L1` reuse over
  detouring through `UB`

Detail: `agent/references/constraints/autosync.md`.

## Cross-side mutex patterns

- a5 cube -> vec: `CvMutex(src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)`
- a2 cube -> vec: `CvMutex(src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)`; requires GM workspace bridge
- a5 vec -> cube: `VcMutex(src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)`
- a2 vec -> cube: same `VcMutex` signature, but the path still goes through GM workspace

Pattern references:
- a2 cube -> vec bridge: `agent/references/patterns/a2-cube-vec.md`
- a2 vec -> cube bridge: `agent/references/patterns/a2-cube-vec-cube.md`
