# Sync Constraints

Read this file when designing or debugging synchronization in a kernel — autosync lowering,
cross-side ownership handoffs, or counter/buffer lifetime layout.

## 1. Core rule

`auto_sync()` is same-side only. It orders within one pipeline side (cube or vec).
It does not transfer ownership between cube and vec.

- same-side queueing: use `auto_sync()`
- cube → vec handoff: requires explicit `CvMutex`
- vec → cube handoff: requires explicit `VcMutex`

There is no autosync-managed `V → MTE2` family; a vec op such as `dup(...)` does not become
ordered ahead of a later `gm_to_ub_pad` just because both are inside the same `auto_sync()` region.

## 2. Supported pipe pairs

Use `agent/references/facts-device-runtime.md` for the exact current pipe map
and supported `auto_sync()` pairs. This file owns the synchronization
constraints that follow from that map.

Important consequences:
- `auto_sync()` synthesizes events only for supported families; instructions
  outside those families are ignored.
- The family names used below are `ubin`, `ubout`, `ubrelay`, `l1`, `l0`, and
  `fix`.
- GM/UB load helpers are MTE2 and UB store/publish helpers are MTE3, so they do
  not stay on `Pipe.V` for autosync matching.

### 2.1 Cross-side L1 initialization needs an explicit prologue edge

`auto_sync()` does not order cube-side `set_constant_to_l1(...)` initialization
against the first vec-side `UB -> L1` MTE3 overwrite of the same slots. Mutex
bootstrap tokens allow the first producer beat to start; they do not prove the
initial clear has retired. Gate the first publish once, after all overlapping
L1 initialization writes:

```python
set_constant_to_l1(l1_a[0], 0.0)
set_constant_to_l1(l1_a[1], 0.0)
cube_ready(7, Pipe.MTE2)
wait_cube(7, Pipe.MTE3)
```

Use a flag id not used by another event or mutex in the kernel. The wait belongs
on MTE3 because that is the first operation that can overwrite the initialized
L1 storage. This is an explicit cross-side lifetime boundary, not a reason to
globally serialize the simulator or widen same-side autosync.

### 2.2 Re-initializing L1 inside an outer loop needs a barrier too

The same-side mirror of 2.1: when an outer loop re-clears an L1 buffer for its
next iteration, that `set_constant_to_l1` is not ordered against the *previous*
iteration's `matmul` reads of the same buffer (the L1 → L0 copy runs on MTE1).
`auto_sync()` inserts no edge, so the clear can overtake the last inner-loop
mmad and zero the operand it was about to read.

```python
for tile in range(0, batch, TILE):
    bar_all()                       # load-bearing: drains the previous tile's
    set_constant_to_l1(l1h[0], 0.0) # L1 -> L0 reads before the fill
    set_constant_to_l1(l1h[1], 0.0)
    for t in range(0, seq):
        ...
```

Two things make this expensive to diagnose. The simulator's
`local_memory_hazard_check` does not cover L1, so it runs clean and only the
numbers are wrong. And the signature is narrow: only the *last* inner iteration
of the *earlier* outer iteration is wrong, while every later tile is correct.
Zeroing the operand that flows through the buffer (here `W_hh`) makes the error
disappear entirely, which is what identifies it as a state-path race rather
than an arithmetic bug.

## 3. The FIX→MTE1 gap

`auto_sync()` does **not** synthesize a `FIX → MTE1` edge.

Symptom: `l0c_to_l1(...)` followed by `l1_to_l0(...)` / `mmad(...)` consuming the same L1
tile is not automatically ordered. The common manifestation is "first cube term is correct,
later correction term is missing."

For a one-shot same-iteration dependency, a single fence is enough:

```python
fix_to_mte1 = SEvent(Pipe.FIX, Pipe.MTE1)

l1mid <<= l0c_mid
fix_to_mte1.set()
fix_to_mte1.wait()

matmul(l0c_out, l1mid, l1z, ...)
```

Verified in `agent/example/kernels/a5/matmul/matmul_l0c_to_l1_demo.py:38-53`.

For a reusable slot or overlapping producer/consumer lifetime, use a full ready/valid pair:

```python
ready = SEvent(Pipe.MTE1, Pipe.FIX, preset=True, name="l1fix_ready")
valid = SEvent(Pipe.FIX, Pipe.MTE1, name="l1fix_valid")
# producer side (FIX writes L1):
ready.wait()           # wait for MTE1 consumer to finish
l0c_to_l1(...)        # FIX-side write
valid.set()            # signal MTE1 consumer
# consumer side (MTE1 reads L1):
valid.wait()           # wait for FIX write
l1_to_l0(...)         # or mmad(...)
ready.set()            # signal FIX producer
```

Verified in `agent/example/testcases/simulator/cube/test_shortcut_matmul.py:105-119`.

## 4. Event model

Each autosync slot uses a two-token `ready`/`valid` handshake
(`LocalSync` / `LocalEventBank` in `easyasc/simulator/sync.py`: `LocalSync`
starts at line 536 and `LocalEventBank` starts at line 595):

1. producer waits on `valid` — acquires slot capacity
2. producer writes the slot
3. producer sets `ready` — signals data available
4. consumer waits on `ready` — waits for data
5. consumer reads the slot
6. consumer sets `valid` — returns slot capacity

`preset=True` on the `valid` event pre-publishes one capacity token so the first producer
round does not block. A `DEvent` pre-sets two tokens (double-buffer width).

On the current A2 and A5 backends, one ordered pipe direction has eight hardware event IDs.
`SEvent<SRC, DST, ...>` consumes one ID for that direction,
`DEvent<SRC, DST, ...>` consumes two, and `TEvent<SRC, DST, ...>` consumes
three, while `QEvent<SRC, DST, ...>` consumes four. Count the declarations in the final
lowered lane, including events synthesized by `auto_sync()`. The simulator
preflights this budget and rejects the ninth ID before launching cores; the two
directions of a pipe pair have independent budgets.

Healthy reuse shape: `wait(valid) ... set(ready) wait(ready) ... set(valid) wait(valid) ...`

If the kernel logic does not fit this contract, `auto_sync()` is the wrong tool for that edge.

### 4.1 Shared buffer ownership across nested scopes

A control-flow boundary does not create a new physical buffer lifetime. When a
parent autosync scope and a nested loop or conditional both produce values into
the same `Tensor` / `DBuff` / `TBuff` / `QBuff`, their supported pipe pair must
use one event family. Dynamic slot indices are collapsed to the backing-buffer
identity for this decision, so a monotonically increasing counter may rotate
through the same physical slots across the scope boundary without receiving a
second set of preset capacity tokens.

Family matching uses the buffers produced by the source pipe in each local
scope. It must not use every buffer read by the destination pipe: a child may
consume parent-preloaded state while producing a different per-iteration
buffer, and merging those two streams would serialize unrelated lifetimes or
pair the wrong ready/valid tokens.

For cube `MTE2 -> MTE1`, a common valid shape is a first matmul K chunk in the
parent scope followed by accumulate chunks in a nested loop, all using the same
L1 DBuff pair. Inspect the lowered event declarations whenever this pattern is
changed: the parent and nested writes to that DBuff must refer to the same
`l1_<index>` event pair. The structural regression is
`agent/example/testcases/parser/sync/test_autosync_cube_l0_l1_events.py`.

## 5. Counters and buffer lifetimes

**Core rule**: different buffer lifetimes must use different counters. Same-lifetime paired
operands may share one counter.

Same lifetime means: loaded together, consumed together, retired together (one logical stage pair).

```
# OK — l1x / l1y are consumed together per k-tile
l1x[l1_cnt], l1y[l1_cnt]

# Wrong — l0c lives across a different loop boundary
l1x[l1_cnt], l0c[l1_cnt]   # l0c_cnt must be separate
```

Named counters by stage ownership, not generic sequence:
`l1_cnt`, `l0c_cnt`, `stage1_cnt`, `stage2_cnt`, `vec_in_cnt`, `vec_out_cnt`.

Dynamic `DBuff` / `TBuff` / `QBuff` indices wrap onto the physical slot family
by slot count. A loop counter can keep increasing across chunks or tiles without
exhausting slots, but every wrap is real slot reuse and must still be protected
by the correct handshake, `auto_sync()` scope, or explicit drain at the lifetime
boundary.

Delayed-stage rule: if a consumer reads data one iteration later, treat it as a distinct
lifetime unless the entire slot family stays one coherent delayed pipeline. Do not hide a
delayed lifetime behind the producer counter.

Buffer depth for the delayed overlap window: "one-tile delay" does not automatically mean
`DBuff` is sufficient. If the producer may advance to tile `j+2` while the delayed consumer
still holds tile `j`, `DBuff` can overwrite live data. Size from the real producer/consumer
overlap, not the nominal delay count.

Cross-beat slot rule: count beats by the earliest point where a physical slot
can be written again, and the latest pipe that still reads the old value. If a
value is loaded in producer beat `j` but its last consumer can execute after
the producer for beat `j+2` has started filling the same DBuff slot, the value
has crossed two producer-beat boundaries and needs at least three physical
slots. Escalate from `DBuff` to `TBuff` or `QBuff` according to that live-beat
window; do not decide from source-code proximity or from the mutex depth alone.

Example: a producer MTE2 load into `feat_ub[j]` may appear just above a final VF
consumer in source, but the final V pipe runs after cube and FIX stages. If
producer MTE2 for `j+2` can refill `feat_ub[j % 2]` before that final V consumer
has read beat `j`, `DBuff` is too shallow. In that shape, use a larger slot
family such as `QBuff` or restructure the schedule so the old slot is retired
before reuse.

Chunked vec stages: keep input and output counters separate even when both increment once per
chunk (`vec_in_cnt` for `MTE2→V` families; `vec_out_cnt` for `V→MTE3` families).

### Lookahead drain structure

This file owns the event, mutex-depth, and physical-slot constraints. The
multi-stage schedule composition moved to
`agent/references/patterns/lookahead-drain.md`; use it to write explicit stage
delays before applying the constraints below.

## 6. CvMutex / VcMutex depth

`depth` is not a vague tuning hint. It controls how many initial ready tokens the kernel gets
for the intra-core handoff — the semaphore credit count that sets how many producer beats may be
in flight before the consumer frees a slot. It *looks* orphaned because it never appears in the
kernel body beside `lock()` / `ready()` / `wait()` / `free()`: it is materialized only in the
auto-emitted prologue/epilogue (`easyasc/kernelbase/kernelbase.py:258-278`, the
`for _ in range(mutex.depth)` loop), which pre-publishes `depth` free-tokens and drains exactly
`depth` at the end. A wrong `depth` stays balanced across that prologue/epilogue, so it raises no
"unbalanced events" warning — the overwrite is silent. The mutex shape lives in
`easyasc/utils/mutex.py`.

| Buffer type   | Max depth |
|---------------|-----------|
| plain `Tensor`| 1         |
| `DBuff`       | 2         |
| `TBuff`       | 3         |
| `QBuff`       | 4         |

Source: `easyasc/utils/Tensor.py` — `DBuff` (line 730), `TBuff` (line 838), `QBuff` (line 934)
are 2-, 3-, and 4-slot families respectively.

These are upper bounds, not defaults. Only use depth > 1 when the producer and consumer really
rotate across those distinct physical slots. If the code keeps reusing one fixed view, `depth`
must stay 1 even if the object type is `DBuff` / `TBuff` / `QBuff`. Setting `depth=2` on a
single-slot `Tensor` tells the runtime there are two free slots and can create silent overwrite
races.

### Depth escalation checklist

Raising a `CvMutex` / `VcMutex` depth is a topology change. Do not change the
mutex alone.

For a handoff depth `D`, every local buffer whose live range can overlap `D`
producer beats must have at least `D` physical slots. This includes intermediate
buffers produced after the explicit handoff if they are still consumed by the
same overlapped stage. For example, if a vec→cube stage publishes several L1
operands through `VcMutex(depth=2)`, a later cube-produced L1 intermediate must
also be double-buffered if a following cube matmul can overlap the next work
item. Although that intermediate is not written by vec, it is part of the same
overlapped cube-stage state.

For shared scratch buffers, size by `(overlapped beats) * (simultaneously live
roles)`, not by the number of conceptual tensors in the original formula. If a
two-beat pipeline uses one L0C role for `base` and one role for `tmp`, the
scratch family needs four physical slots:

```python
l0c_base = work_idx * 2
l0c_tmp = l0c_base + 1
l0c = QBuff(DT.float, [M, N], Position.L0C)  # slots: 0/1 for beat 0, 2/3 for beat 1
```

Using `DBuff` with that `work_idx * 2` / `+ 1` pattern is not enough:
`base` always maps to physical slot 0 and `tmp` always maps to physical slot 1,
so the next beat can overwrite a previous beat's still-live `base`.

When a temporary L0C slot is reused by a later M-side matmul immediately after a
FIX-side consumer such as `l0c_to_gm_nz2nd` or `l0c_to_ub`, add an explicit
`SEvent(Pipe.FIX, Pipe.M)` or otherwise prove that autosync already owns that
same physical slot edge. Same-side autosync normally orders `M -> FIX`; it does
not automatically create the reverse `FIX -> M` reuse fence for a scratch slot.

### Slot lift for overlapping local roles

This file owns event, mutex-depth, and last-consumer constraints. The complete
role-overlap diagnosis and `Tensor` -> `DBuff`/`TBuff`/`QBuff` composition moved
to `agent/references/patterns/buffer-slot-lifetime.md`.

Cross-side ownership patterns:
- a5 direct cube → vec (`l0c_to_ub` / later vec compute):
  `CvMutex(..., src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)`
- a2 cube → vec through a GM workspace bridge (`l0c_to_gm_nz2nd` / `gm_to_ub_pad`):
  `CvMutex(..., src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)`
- vec → cube, conservative/full cube-stage lifetime:
  `VcMutex(..., src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)`
- a5/C310 vec → cube, UB→L1 handoff where the cube side only needs the L1
  slot until `l1_to_l0` finishes:
  ```python
  VcMutex(
      ...,
      src_start_pipe=Pipe.MTE3,
      src_end_pipe=Pipe.MTE3,
      dst_start_pipe=Pipe.MTE1,
      dst_end_pipe=Pipe.MTE1,
  )
  ```

The `src_start_pipe` / `src_end_pipe` pair protects the producer pipe; the
`dst_start_pipe` / `dst_end_pipe` pair protects the consumer pipe. For
UB→L1 publish helpers (`ub_to_l1`, `ub_to_l1_nd2nz`, `ub_to_l1_nz`), the producer pipe is
MTE3. If the cube side only consumes the L1 slot through `l1_to_l0`, the
consumer pipe is MTE1 and the slot can be released after MTE1 finishes; later
`M`/`FIX` work consumes L0/L0C state, not that L1 slot. Keep `Pipe.FIX` when the
protected resource lifetime really extends through `M`/`FIX`, or when the
lifetime is not clear.

On b-series devices, cross-lane wait start points are main-loop only:
`wait_cube(..., pipe=...)`, `wait_vec(..., pipe=...)`, and the corresponding
`CvMutex` / `VcMutex` `src_start_pipe` / `dst_start_pipe` arguments must use
`Pipe.S`. Do not attach those waits to `MTE*`, `M`, `FIX`, or `V` queues on
b-series. Because `Pipe.S` blocks the main scheduling stream, place `lock()`
immediately before the protected producer instruction and `wait()` immediately
before the protected consumer instruction to avoid blocking unrelated work too
early.

On a5/C310 devices, CrossCore set/ready calls must attach to a real pipeline
queue. `Pipe.S` and `Pipe.ALL` are illegal for `cube_ready`, `vec_ready`,
`allcube_ready`, `allvec_ready`, and `intracore_allvec_ready`, including the
`src_end_pipe` / `dst_end_pipe` arguments hidden inside `CvMutex` / `VcMutex`.
Cube-side ready/set calls must also avoid `Pipe.MTE3`; vector-side ready/set
calls may use `Pipe.MTE3`.
CrossCore waits may still use `Pipe.S`; this is the normal default start pipe
for mutex `lock()` / `wait()` unless an explicit queue wait is intentional.

`allvec_ready` / `allvec_wait` must not appear in a `@kernel(mode="vec")`
kernel. EasyASC vec mode generates `KERNEL_TYPE_AIV_ONLY`, which is not a valid
A2 task declaration for this mode-0 cross-core collective. Use
`@kernel(mode="mix")` instead; on A2 its 1:2 task ratio launches 20 blocks with
both vec lanes resident, so all 40 AIV participants can reach the barrier. The
simulator rejects the invalid vec-mode combination before starting execution.
`intracore_allvec_ready/wait` is a different, same-core primitive and is not
covered by this restriction.

Same-core vec-lane coordination is a separate problem from cube↔vec ownership.
Use `intracore_allvec_ready(flag_id, pipe=...)` /
`intracore_allvec_wait(flag_id, pipe=...)` when the two vec lanes inside one
core must see each other's GM/L1 staging writes and no cube-side participant is
the consumer. Do not use `vec_ready/wait_vec` as a lane-to-lane barrier; that
pair is a vec→cube token.

Common cases:

- lane 0 reads a full row or matrix whose two halves were produced by lane 0
  and lane 1;
- one vec lane must finish reading a temporary GM slot before the other lane
  overwrites its half;
- a reused GM slot first holds an intermediate and later becomes a final output.

For reused GM slots, split the lifetime into distinct barriers. The
"temporary has been read" point and the later "final output write has drained"
point are not the same synchronization edge. Collapsing them into one token can
let a same-core next tile observe delayed writes or overwrite a live temporary.

Mutex lifetime: `lock()` before producer write; `ready()` after producer completes;
`wait()` before consumer read; `free()` immediately after consumer completes.

## 7. Practical debug hooks

For event-pairing failures, expand the instruction stream and filter to the failing family
before changing the kernel body:

1. Call the `@kernel` once with placeholder `GMTensor(...)` args.
2. Run `split_instructions(...)`, then `insert_auto_sync(..., mode="cube" | "vec")`,
   then `eliminate_duplicated_event_creation(...)`.
3. Filter to the failing family: cube (`l1`, `l0`, `fix`) or vec (`ubin`, `ubout`, `ubrelay`).
4. Print only `event_wait` / `event_set` and the paired producer/consumer ops around them.
5. Compare to a stable baseline kernel that uses the same pipe pair.

When the failure involves auto-inserted events, also inspect the generated C++
ordering. This catches cases where the instruction stream is hard to reason
about because the final release is emitted outside a loop or outside the
control-flow guard that consumed the preset event:

```python
from pathlib import Path

from easyasc.a5 import DT, GMTensor, Var
from easyasc.parser.asc import translate_split

# 1. Call the kernel once with placeholder GMTensor / Var arguments so
#    kernel.instructions is populated.
my_kernel(
    GMTensor(DT.float, [M, K], name="x"),
    GMTensor(DT.float, [N, K], name="y"),
    GMTensor(DT.float, [M, N], name="out"),
    Var(M),
    Var(N),
    Var(K),
)

# 2. Generate the split C++ view without building a custom op.
cube_cpp, vec_cpp = translate_split(my_kernel.instructions, "sync_probe")
Path("tmp/<task>/sync_probe_cube.cpp").write_text(cube_cpp)
Path("tmp/<task>/sync_probe_vec.cpp").write_text(vec_cpp)
```

Then search the generated files for the failing event name and adjacent pipe
ops:

```bash
rg -n "_tmp_sevent_valid_fix_0|\\.set\\(|\\.wait\\(|SetFlag|WaitFlag|MMAD|L0C2|for \\(|if \\(" tmp/<task>/sync_probe_cube.cpp
rg -n "_tmp_sevent_valid_ubout_0|\\.set\\(|\\.wait\\(|SetFlag|WaitFlag|DataCopy|for \\(|if \\(" tmp/<task>/sync_probe_vec.cpp
```

Read the C++ literally. For `preset=True` events, every generated final
`*.set()` must be control-flow reachable only if a matching `*.wait()` consumed
the preset token earlier on that same core/lane. In dynamic work distribution,
empty cores are a common trap: if the work loop is skipped but a final
auto-sync release remains outside the `if work_begin < work_count` guard, the
simulator will report `set on already-set flag`. Move the whole auto-sync work
region under the same guard, or restructure the region so empty cores do not
publish unused capacity tokens.

Simulator notes:
- `bar_all()` is the cross-pipe drain on the simulator path. Single-pipe barriers such as
  `bar_v()` / `bar_m()` / `bar_mte*()` / `bar_fix()` order only their own
  hardware pipe. Each simulated pipe already executes FIFO, so dropping these
  barriers is intentional and must not be replaced with a cross-pipe wait. Only
  `bar_all()` or `barrier(Pipe.ALL)` routes to the cross-lane `bar_all` path in
  `easyasc/simulator/bridge.py`. The simulator drain is cycle-visible: after
  `bar_all()`, every pipe and PipeS advances to the max arrival cycle of that
  barrier round before later tasks can be scheduled.
- `event_wait timeout` usually means the next producer-side `event_set` never happened.
- `event_set on already-set flag` means a duplicate set before a matching wait consumed
  the previous token.

See `agent/playbooks/kernel-debugging.md` for the full event-pairing debug workflow.

## 8. Warning: NOT balanced auto_sync events

Treat this warning as real. Typical causes:

- only one side of a supported pair appears in the region
- parent and child regions both partially own the same handshake
- a slot-family lineage change caused by reusing a live buffer as temporary scratch
- a counter / slot token mismatch that moved later ops onto a different family

Required debug shape: identify which pipe pair failed (`ubin`, `ubout`, `l1`, `l0`, `fix`),
which buffer family caused the mismatch, and whether the fix is region restructuring,
buffer separation, or counter repair.

`ub_to_ub` runs on `Pipe.V`, not MTE3. A `ub_to_ub` step between two vec computations is
already serialized by the V pipeline; no `SEvent(Pipe.MTE3, ...)` fence is needed around it.
On B-device targets, `_insert_b_device_vec_barriers` inserts a `bar_v` when hazard
detection finds a RAW, WAR, or WAW conflict on shared storage keys — including
anonymous UB scratch tensors, their slices, and back-to-back `ub_to_ub` writes to
the same tensor.

### Storage-aware cube autosync modes

`auto_sync(mode="aggressive")` / `auto_sync("aggressive")` is an opt-in mode for
cube-side `MTE1 -> M` scheduling. It groups adjacent MTE1 producer bursts by L0
storage key and also groups `M -> FIX` ownership by the physical L0C backing
allocation. Reach for it only when normal L0 autosync over-serializes independent
L0 producer groups; do not use it as a generic fix for unbalanced events.

`auto_sync(mode="fix_storage")` changes only `M -> FIX` ownership grouping while
retaining conservative `MTE1 -> M` scheduling. Use it when separate runtime
branches access views of one L0C allocation and must share that physical
ownership lineage without perturbing the established L0 operand schedule.

`auto_sync(mode="manual_fix")` retains conservative L1 and L0 autosync but
deliberately skips automatic `M -> FIX` insertion. The kernel must then provide
an explicit `M -> FIX` result-ready edge before every Fix read and a matching
`FIX -> M` retirement edge before each physical L0C slot is overwritten. This
mode is for schedules whose real slot ownership cannot be represented by the
scope-counted `DEvent`; it is unsafe as a synchronization shortcut.

Nested loop placement matters. Loop-carried valid events from child regions must
remain before that child loop's `end_loop`, and ready events for an outer L0
producer must be emitted before a nested child loop consumes the producer's L0
state. If a parent/child region mixes L0 ownership, inspect the generated
event ordering rather than assuming the source indentation tells the full story.
Parser regressions for these shapes live in
`agent/example/testcases/parser/sync/test_autosync_cube_l0_l1_events.py`.

## 9. VF local-memory barriers (`vf_barrier`)

`vf_barrier` is the in-`@vf` local-memory ordering primitive (lowers to
`MicroAPI::LocalMemBar`). It is distinct from `auto_sync`: auto_sync orders pipes
across the kernel; `vf_barrier` orders STORE / LOAD / compute streams inside one
VF body. Hardware can overlap these streams, so a missing barrier races even when
the source order looks sequential.

When a VF writes a UB view and a later iteration of the surrounding loop reads or
overwrites the **same base view** at a **dynamic `Var` offset**, add an explicit
`vf_barrier(VfPipe.STORE, VfPipe.LOAD)` after the writeback, plus
`vf_barrier(VfPipe.STORE, VfPipe.STORE)` if a later iteration also writes that
view. The static micro checker cannot prove disjointness through dynamic Var
offsets, so "writes go to different rows" is not a sufficient argument — the
ordering must be explicit. Treat any micro static-warning of this shape as a real
hazard, not noise.

Repaired sites following this pattern:

- `agent/example/kernels/a5/matmul/matmul_mknk_2dgrid_splitk_add1.py`
- `agent/example/kernels/a5/matmul/matmul_rowwise_l2_norm.py`
- `agent/example/kernels/a5/matmul/matmul_rowwise_norm_large_nk.py`
- `agent/example/kernels/a5/utility/chunk_row_cumsum.py`
- `agent/example/kernels/a5/gdn_legacy/recurrent_state_attn_vec.py`
- `agent/example/kernels/a5/attention/flash_attn_full_fp8_causal.py`

The basic `vf_barrier` rule (any in-place STORE→STORE / STORE→LOAD on the same
UB region) is also recorded in `agent/references/facts-authoring.md` under the
a5 `@vf` hard rules.

`vf_barrier` is an ordering primitive, not a scalar-dependency primitive. Do
not rely on an in-place UB tensor to carry scan/cumsum state inside a VF loop.
This shape is unsafe:

```python
val <<= work_ub[0:1, t:t + 1].single()
acc <<= acc + val
work_ub[0:1, t:t + 1] <<= acc.single_value()
```

Even with `vf_barrier(VfPipe.STORE, VfPipe.LOAD)`, real 950 hardware can treat
the same-base UB read/write recurrence differently from the source order. A
minimal hardware probe showed an aliased reverse scan on all-ones input
returning a constant negative offset across leading columns. The same probe
also showed a separate `source UB -> output UB` round trip failing on random
inputs even though it matched an all-ones case, so "separate UB" is not a free
replacement for direct register accumulation.

For VF scan/reduction-style recurrences:

- keep the loop-carried accumulator in `Reg` / `RegList`;
- prefer computing the per-element base value and recurrence directly in
  registers when the formula allows it;
- if base values must be materialized first, write them to a source scratch UB
  and write scan results to a different UB tensor, but require a hardware probe
  for that exact generated code and input range;
- treat in-place UB scan updates as a correctness bug unless a hardware probe
  for that exact generated code proves otherwise.

Apply the same conservative rule to a loop-carried, same-base dynamic
`DataCopyGather` / `DataCopyScatter` read-modify-write recurrence. On Ascend
950PR with CANN 9.2, `vf_barrier`, `bar_v`, `bar_all`, a V-to-MTE3-to-V event
round trip, and `DataSyncBarrier<MemDsbT::UB>` all failed to publish the next
Sinkhorn iteration reliably. These mechanisms order instruction classes; they
do not create queue-buffer ownership for a plain UB tensor. The value simulator
executes scatter synchronously and therefore emits `VfPublicationWarning` for
this structural pattern instead of pretending that a correct simulated value
is hardware evidence.

Prefer an immutable UB source with recurrence state in `Reg` / `RegList` and a
physically separate output, or use a real queue ownership handoff. A ping-pong
rewrite is still a new schedule and needs a hardware probe for its exact
generated code; distinct base addresses alone do not prove publication.

## 10. Files to study

- `easyasc/parser/asc_autosync.py` — pipe pairs, op→pipe map, event synthesis
- `easyasc/simulator/sync.py` — `LocalEventBank`, `LocalSync`, `set_event`, `wait_event`
- `easyasc/simulator/bridge.py` lines 20–52 — op→pipe map for simulator dispatch
- `easyasc/utils/mutex.py` — `CvMutex` / `VcMutex` depth
- `easyasc/utils/Tensor.py` — `DBuff` (line 730), `TBuff` (line 838), `QBuff` (line 934)
- `agent/example/testcases/parser/sync/test_autosync_vec_event_metadata.py`
- `agent/example/testcases/parser/sync/test_autosync_cube_l0_l1_events.py`
- `agent/example/kernels/a5/pipeline_patterns/vec_cube_abs_sqrt_matmul.py`
- `agent/example/kernels/a5/matmul/matmul_mknk_2dgrid_splitn.py`

## 11. Fused-kernel staging

When multiple pipeline stages are fused into one kernel, every GM staging slot
that formerly crossed a launch boundary now needs explicit synchronization.
`auto_sync()` does not replace that boundary — see
`agent/references/facts-authoring.md` (Fused-kernel GM staging rules) for
cross-side tokens (`cube_ready/wait_cube`, `CvMutex` for cube→vec;
`vec_ready/wait_vec`, `VcMutex` for vec→cube), intracore vec-lane barriers
(`intracore_allvec_ready/wait`), cross-core barriers (`allvec_ready/wait`),
and staging-slot lifetime rules.
