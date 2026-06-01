# Autosync Constraints

Read this file when debugging or designing `auto_sync()` behavior.
Do not use it as a general synchronization guide for every kernel.

## Goal

Use `auto_sync()` correctly as a same-side queueing tool.
Do not confuse it with cross-side ownership transfer.

## 1. Core rule

`auto_sync()` is only a same-side pipeline-ordering mechanism.
It does not transfer ownership between cube and vec by itself.

Practical consequence:
- same-side queueing may use `auto_sync()`
- cube -> vec handoff still needs explicit `CvMutex`
- vec -> cube handoff still needs explicit `VcMutex`
- there is no autosync-managed `V -> MTE2` family, so a vec op such as `dup(...)`
  does not become ordered ahead of a later `gm_to_ub_pad` just because both are
  inside the same `auto_sync()` region

## 2. Mental model

Treat `auto_sync()` as:
- marker emission during authoring
- event synthesis later during split/lowering

Current high-level flow:
1. authoring inserts `start_auto_sync` / `end_auto_sync`
2. split translation rewrites cube and vec instruction lists
3. autosync inserts event declarations plus `event_wait` / `event_set`
4. duplicate event declarations are deduplicated later

`start_auto_sync` and `end_auto_sync` themselves are not the synchronization.
The later synthesized events are.

## 3. Current hard-coded pipe pairs

Vec-side pairs:
- `MTE2 -> V` as `ubin`
- `V -> MTE3` as `ubout`

Cube-side pairs:
- `MTE2 -> MTE1` as `l1`
- `MTE1 -> M` as `l0`
- `M -> FIX` as `fix`

If an instruction is not classified into one of the supported pipe families, current `auto_sync()` will not reason about it.

Exact op-to-pipe mapping used by `auto_sync()`:
- `Pipe.MTE2`: `gm_to_l1_nd2nz`, `set_constant_to_l1`, `gm_to_ub_pad`
- `Pipe.MTE1`: `l1_to_l0`
- `Pipe.M`: `mmad`
- `Pipe.FIX`: `l0c_to_gm_nz2nd`, `l0c_to_l1`, `l0c_to_ub`
- `Pipe.MTE3`: `ub_to_gm_pad`, `ub_to_l1_nd2nz`, `ub_to_l1_nz`
- `Pipe.V`: remaining vec allowlist after subtracting the memory ops above

Because `gm_to_ub_pad`, `ub_to_l1_nd2nz`, `ub_to_l1_nz`, and `ub_to_gm_pad` are reassigned to `MTE2` / `MTE3`, they do not stay on `Pipe.V` for autosync matching.

Practical tail example:
- `dup(ub, 0.0)` runs on `V`
- `ub[...] <<= gm[...]` lowers to `gm_to_ub_pad` on `MTE2`
- `auto_sync()` does not create a `V -> MTE2` ordering edge for that pair

If correctness depends on the zero-fill finishing before the reload starts, do
not assume `auto_sync()` will serialize it.
On the simulator path, the local barrier that actually drains cross-pipe work is
`bar_all()`; single-pipe barriers such as `bar_v()` are no-ops there.

## 4. Event model

Each autosync-managed slot behaves like a two-token handshake:
- `ready`: producer -> consumer
- `valid`: consumer -> producer

Stable mental contract:
1. producer waits on `valid`
2. producer writes the slot
3. producer sets `ready`
4. consumer waits on `ready`
5. consumer reads the slot
6. consumer sets `valid`

If your kernel logic does not fit this contract, `auto_sync()` is probably the wrong tool for that edge.

## 5. Event metadata and naming

Synthesized event metadata rule:
- `ready`: name prefix `_tmp_{s|d}event_ready_`, `src_pipe = producer pipe`, `dst_pipe = consumer pipe`, `preset = False`, `idx = 9998`
- `valid`: name prefix `_tmp_{s|d}event_valid_`, `src_pipe = consumer pipe`, `dst_pipe = producer pipe`, `preset = True`, `idx = 9999`

Generated family names:
- vec `MTE2 -> V`: `_tmp_{s|d}event_{ready|valid}_ubin_<idx>`
- vec `V -> MTE3`: `_tmp_{s|d}event_{ready|valid}_ubout_<idx>`
- cube `MTE2 -> MTE1`: `_tmp_{s|d}event_{ready|valid}_l1_<idx>`
- cube `MTE1 -> M`: `_tmp_{s|d}event_{ready|valid}_l0_<idx>`
- cube `M -> FIX`: `_tmp_{s|d}event_{ready|valid}_fix_<idx>`

Read generated code literally. For example:
- `DEvent<PIPE_V, PIPE_MTE2, true> _tmp_devent_valid_ubin_0;`
  means the vec-side `ubin` slot-0 `valid` token; `V` is the consumer pipe, `MTE2` is the producer pipe.

## 6. Sync grouping is not just by buffer name

## 5a. Practical state machine for same-side reuse

When debugging generated `event_wait` / `event_set`, a compact state machine is often
faster than staring at the whole region.

Useful mental states:
- `idle`: no producer token is currently held
- `producing`: producer already reacquired `valid` for the current round
- `consumed`: consumer already finished the current round and the next producer round
  must first close and reacquire `valid`

Useful transitions:
1. first `src_pipe` of a round:
   `idle -> producing` via `valid.wait`
2. matching `dst_pipe` of that round:
   `producing -> consumed` via `ready.set` then `ready.wait`
3. next `src_pipe` after a completed round:
   `consumed -> producing` via `valid.set` then `valid.wait`

Practical consequence:
- for a `src -> dst -> src -> dst` pattern on the same autosync-managed family,
  the second producer round must reacquire `valid`
- if codegen shows only `valid.set` before the second producer round, the autosync
  insertion is unbalanced for slot reuse even if no warning is printed

Concrete cube-side example to look for:
- expected shape:
  `valid.wait ... ready.set/ready.wait ... valid.set valid.wait ...`
- suspicious shape:
  `valid.wait ... ready.set/ready.wait ... valid.set ...`

This check is especially useful for:
- repeated `l1_to_l0 -> mmad -> writeback` inside one inner loop
- repeated `mmad -> FIX` stages that reuse the same `DBuff` / `TBuff` family

Repository note:
- `testcases/parser/sync/test_autosync_event_metadata.py` contains a regression that checks
  the `src -> dst -> src -> dst` case reacquires `valid` on the second round
- the current implementation path is `easyasc/parser/asc_autosync.py`

Current grouping is based on normalized sync-key groups built from the participating tensors.
That means the event family index depends on:
- logical buffer lineage
- slot expression
- whether the view comes from `DBuff` / `TBuff` or a plain `Tensor`

Pair-specific sync-key collection:
- vec `MTE2 -> V` (`ubin`): producer key from `gm_to_ub_pad` destination; consumer key from vec-op source
- vec `V -> MTE3` (`ubout`): producer key from vec-op destination; consumer key from `ub_to_*` source
- cube `MTE2 -> MTE1` (`l1`): producer key from `gm_to_l1_nd2nz` destination; consumer key from `l1_to_l0` source
- cube `MTE1 -> M` (`l0`): producer key from `l1_to_l0` destination; consumer key from `mmad` sources `src_a`/`src_b`
- cube `M -> FIX` (`fix`): producer key from `mmad` destination; consumer key from FIX-side source

`call_micro` special case: if treated as source pipe `V`, uses `write_sync_keys`; if destination pipe `V`, uses `read_sync_keys`.

Sync-key derivation (from `easyasc/utils/sync_key.py`):
- `DBuff`/`TBuff` view key shape: `("buffer", buffer_type, position, buffer_name, source_index_token)`
- plain `Tensor` view key shape: `("tensor", position, tensor_name)`
- scalar token rule: concrete `Var.value` uses numeric value; symbolic `Var` uses variable name; `None` becomes empty token
- temporary tensors with `_tmp_` prefix are ignored for grouping
- all keys are deduplicated, sorted, and packed into one normalized `SyncKeyGroup`
- if no usable keys remain, fallback is `("scope", buf_name, str(id(node)))`

Practical consequence:
- same buffer family name does not automatically mean same autosync event family
- changing slot lineage can silently create a different event family

## 5b. Workspace-mediated MTE2-reload reuse rule

Use one shared local `DBuff` family plus one shared slot counter only when the
real same-side reuse edge is a workspace-mediated `MTE2` reload into local
storage.

Conclusion:
- this is not an a2-only rule
- this is not a vec-only rule
- the deciding factor is whether multiple stages are being matched as the same
  `MTE2 -> consumer` autosync family after reloading from workspace / GM
- current repository families with an `MTE2` producer include:
  - vec `MTE2 -> V` (`ubin`)
  - cube `MTE2 -> MTE1` (`l1`)
  - the rarer vec-side `MTE2 -> MTE3` (`ubrelay`) when a region really has only
    that pair

Logic chain:
1. `auto_sync()` groups by pipe pair first, not by "these two stages happen to
   increment the same counter".
2. The producer side must really be an `MTE2` read op such as `gm_to_ub_pad` or
   `gm_to_l1_nd2nz`, meaning the local slot is being refilled from workspace /
   GM.
3. Family reuse only happens when the normalized sync-key group also matches.
   For `DBuff` / `TBuff`, that key includes the slot token
   `source_index_token`; for plain `Tensor`, it does not.
4. Therefore "flag and buffer count move together" is a necessary bookkeeping
   consequence, not the root cause. The counter matters because it stabilizes
   the slot token for the same local buffer family under the same `MTE2`
   producer family.
5. If the producer is not `MTE2`, this rule does not apply just because a local
   buffer is being reused.

Counterexample:
- `l0c_to_ub` is classified as `Pipe.FIX`, not `Pipe.MTE2`
- so a `FIX -> consumer` edge does not become an `MTE2` reload family merely by
  sharing a `DBuff` name or co-incrementing a counter
- in those cases the primary correctness mechanism is still the real `FIX`-side
  ownership / handoff model, for example `CvMutex(..., src_end_pipe=Pipe.FIX,
  dst_end_pipe=Pipe.V)` on cube -> vec paths

Practical authoring rule:
- use shared-slot `DBuff + count` reuse when you are intentionally preserving
  one workspace-mediated `MTE2` reload lineage across stages
- do not cargo-cult that pattern into intra-core transfers just because the
  temporary tensors look similar

Event-family index assignment in nested code:
- if a node is not mixed-scope, it does not allocate a new event family
- if a mixed-scope node has a sync-key group not yet seen, it gets the next new `<idx>`
- if a later mixed-scope node has the same normalized group, it reuses the same `<idx>`
- a parent preload plus child mixed pipeline can create multiple `ubin` families when buffer groups differ
- a child reusing the same `ubout` stream stays on `_ubout_0` without needing `_ubout_1`

## 6. `SEvent` vs `DEvent`

Do not assume `auto_sync()` always produces `DEvent`.
Current rule:
- single-buffer style views may produce `SEvent`
- slot-buffer traffic usually produces `DEvent`

Plain `Tensor` views can collapse a stage into `SEvent` behavior.

Concrete comparison in this repo:
- `agent/example/kernels/a2/flash_attn_full_pj_hif8.py` keeps stage-1 `ub_score` and stage-2 `ub_pv` as separate plain `Tensor` scratch views, so the vec-side `ubin` story is the simpler baseline to read first
- `agent/example/kernels/a2/flash_attn_full_pj_hif8_commonub.py` replaces those two views with one shared `DBuff` family plus `score_pv_cnt`; that gives the vec `MTE2 -> V` `ubin` edge stable slot lineage, which is why generated code can move to `_tmp_devent_*_ubin_*` queueing
- `agent/example/kernels/a2/flash_attn_full_pj_hif8_causal.py` and `agent/example/kernels/a2/flash_attn_full_pj_half_block32_causal.py` now reuse that same shared local slot family, so their causal-mask variants inherit the same vec-side queueing structure instead of staying on the older plain-`Tensor` scratch path
- for these specific kernels, the improvement point is on the vec `ubin` edge, not on cube-side `l1/l0/fix`: those cube families were already `DBuff`-based in both kernels
- `stage1_cnt` and `stage2_cnt` still stay separate; the shared scratch gets its own counter because the queueing gain comes from the local slot family, not from merging delayed-stage lifetimes

## 7. When a region actually gets a handshake

A region only receives a full autosync handshake when both sides of the current pipe pair appear inside that region.
If a region contains only the producer or only the consumer side, it does not own a complete handshake by itself.

Practical consequence:
- one large outer `auto_sync()` block may produce events only around some inner loops or branches
- nested code may own the real handshake instead of the parent block

## 8. Nested-region rule

If a child region already owns both sides of a pipe pair, the parent should not wrap it again with another full handshake.
Otherwise you get dangling or duplicated token flow.

When parent and child both seem to control the same edge, suspect the region structure first.

## 8a. Event-pairing debug workflow

When debugging a concrete local event failure, inspect the autosync-expanded
instruction stream before changing the kernel body.

Recommended workflow:
1. Build `kernel.instructions` first by calling the `@kernel` once with placeholder
   `GMTensor(...)` arguments.
2. Run:
   - `split_instructions(...)`
   - `insert_auto_sync(..., mode="cube" | "vec")`
   - `eliminate_duplicated_event_creation(...)`
3. Filter the expanded instructions down to one failing family:
   - cube: `l1`, `l0`, `fix`
   - vec: `ubin`, `ubout`, `ubrelay`
4. Print only:
   - `event_wait`
   - `event_set`
   - the paired producer/consumer ops around them
5. Compare that family's action sequence to a stable baseline kernel that uses
   the same pipe pair.

Useful interpretation rules:
- `event_wait timeout` usually means the next producer-side `event_set` never
  happened.
- `event_set on already-set flag` usually means the family got a duplicate
  `event_set` before a matching `event_wait` consumed the previous token.
- On the simulator path, `preset=True` starts the family with one published
  token. So `set_count == wait_count` on timeout usually means the preset token
  was consumed and no later `event_set` was published.

Repository pointers:
- parser split / autosync: `easyasc/parser/asc.py`, `easyasc/parser/asc_autosync.py`
- simulator local-event semantics: `easyasc/simulator_v2/sync/local_events.py`
- good regression home: `testcases/parser/sync/test_autosync_event_metadata.py`

Practical output shape to look for:
- healthy same-family reuse:
  `wait(valid) ... set(ready) wait(ready) ... set(valid) wait(valid) ...`
- missing publish:
  `... wait(valid)` with no prior `set(valid)` for the next round
- duplicate publish:
  `... set(valid) ... set(valid) ...` without an intervening `wait(valid)`

This workflow is especially effective for:
- `M -> FIX` reuse around `l0c_to_ub` / `l0c_to_l1`
- `MTE2 -> MTE1` reuse around repeated `gm_to_l1_nd2nz`
- nested `start_loop` / `start_if` regions that own different autosync families

When the bug only appears in nested regions:
- inspect parent and child `buf_idx` / sync-key family identity, not just the
  pipe pair
- parent/child state transfer is only safe when both scopes refer to the same
  autosync family
- if parent and child are different families on the same pipe pair, carrying
  `ready` / `valid` state across the boundary can create a missing-set or
  duplicate-set failure

## 9. Warning rule

Treat this warning as real:
- `WARNING: NOT balanced auto_sync events, please check the code logic!`

It means the region ended with an unfinished producer-side token story.
Typical causes:
- only one side of a supported pair appears in the region
- the last producer action never reaches its matching consumer phase
- parent and child both partially own the same handshake
- slot lineage changes inside the region

Do not wave this away just because codegen still succeeds.

## 10. Stable authoring rules

Use these rules for current repository behavior:
- wrap one full same-side pipeline stage, not random fragments
- keep producer and consumer on stable `DBuff` / `TBuff` views when possible
- prefer one stable counter per slot family
- do not reuse one counter across different buffering lifetimes
- let `auto_sync()` handle same-side queueing only
- use explicit mutexes for cross-side ownership

Stable cross-side mappings in this repository:
- cube -> vec: `CvMutex(..., src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)`
- vec -> cube: `VcMutex(..., src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)`

Explicit mutex lifetime rule:
- `lock()` before producer write
- `ready()` after producer completes
- `wait()` before consumer read
- `free()` immediately after consumer completes

## 10a. `ub_to_ub` belongs to Pipe V, not MTE3

Despite being a datamove, `ub_to_ub` is serviced by the vec (V) pipe, not
MTE3. Concrete consequences:

- Same-pipe ordering. A `ub_to_ub` step between two vec computations is
  already serialized by the V pipeline. You do **not** need an explicit
  `SEvent(Pipe.MTE3, Pipe.V)` / `SEvent(Pipe.V, Pipe.MTE3)` handshake to
  fence it against its vec producer or consumer.
- `auto_sync()` bar_v fence. The autosync pass now unconditionally emits a
  trailing `bar_v` barrier after every `ub_to_ub` as a conservative fence
  for slot reuse (see `_insert_b_device_vec_barriers` in
  `easyasc/parser/asc_autosync.py`).
- MTE3 events stay for real MTE3 ops. Keep `V -> MTE3` handshakes only for
  ops that truly run on MTE3 (e.g. `ub_to_gm_pad`). `flash_attn_full_pj_hif8`
  keeps `accum_store_ready/valid` around the accum UB→GM write for this
  reason, but its earlier `quant_chunk_loaded/committed` events were
  redundant and have been removed.

If you see legacy kernels inserting V↔MTE3 events around a `ub_to_ub`
chunked-quantize step, that is almost always leftover noise — delete the
events and rely on pipe-V serial ordering plus the autosync trailing bar_v.

## 10b. `dup` participates in both read and write tracking

For WAW-hazard detection, autosync treats the destination of `dup` as both a
consumer and a producer. Concrete consequence: back-to-back `dup`s that
target the same tensor get a separating `bar_v`, matching intuitive hardware
expectations. This is handled by `_VEC_DST_AS_READ_OPNAMES` in
`easyasc/parser/asc_autosync.py`.

## 10c. Vec-only helpers can still break autosync through buffer lineage

Do not reason about autosync at the Python-helper boundary.
Reason about it at the touched slot-family boundary.

Practical failure mode seen during a2 dense-attention-backward bring-up:
- a new helper contained only vec ops
- the helper reused a live stage buffer such as `dpbuf` as temporary quantization scratch
- the enclosing region then no longer had one clean `MTE2 -> V -> MTE3` story for that slot family
- autosync reported `WARNING: NOT balanced auto_sync events, please check the code logic!`

Why this happens:
- autosync groups by supported pipe pair plus normalized sync-key group, not by Python function name
- writing scratch data through a live `DBuff` / `TBuff` family changes the same slot family's vec-side history
- there is still no autosync-managed `V -> MTE2` repair path

Stable authoring rule:
- if a local family later participates in `ub_to_*`, keep that family on one coherent producer/consumer lineage
- do not borrow delayed or still-live stage buffers as temporary vec scratch
- give helper scratch its own plain `Tensor` or its own dedicated local buffer family

Concrete a2 lesson:
- reusing `dpbuf` as quant scratch was enough to create an unbalanced vec `ubout` story
- moving quant scratch onto dedicated chunk-sized tensors removed the warning

## 10d. When adding vec-side transforms, chunk the whole stage instead of splicing scratch into a half-tile region

If a stage originally processed one half-tile such as `[HALF_M, TILE_N]`, and
you later add quantization, extra masking, or format conversion, prefer
re-chunking the entire vec hot path so each chunk owns a full:
- `MTE2 -> V -> MTE3`

Do not keep the old large region and inject extra vec writes into the middle of
someone else's slot lifetime.

Stable a2 pattern:
- choose a row chunk such as `32 x 128` or `16 x 128`
- load `qk/dp` for one chunk from workspace or GM
- finish all vec work for that chunk
- write `p/dqk` for that chunk before advancing to the next one

Practical consequence:
- the warning usually disappears because each chunk owns a complete same-side handshake
- the code becomes easier to audit than one large region with partial parent/child ownership

## 10e. Do not pass sliced views into helpers that internally `reinterpret(...)`

If a helper internally reinterprets its tensor argument, pass the full backing
tensor unless the helper itself was written to be slice-safe.

Concrete failure mode:
- a helper rebuilt a packed causal mask through `causal_mask.reinterpret(DT.int)`
- the caller passed a sliced view such as `causal_mask_dynamic[0:HIF8_CHUNK_M, ...]`
- simulator-v2 then reported a storage/view-size failure such as `shared tensor view exceeds source storage`

Stable rule:
- pass the full tensor into the helper
- keep the helper's internal full-tile assumptions explicit
- slice later at the use site, for example when calling `select(...)`

## 10f. Warning triage must identify the broken autosync pattern and report it

When `WARNING: NOT balanced auto_sync events, please check the code logic!`
appears, do not stop at "the warning exists" or "the result still passes".

Required debugging behavior for future agent work:
- actively search for the unmatched autosync path
- identify which supported pipe pair failed to close, for example `MTE2 -> V`, `V -> MTE3`, `MTE2 -> MTE1`, `MTE1 -> M`, or `M -> FIX`
- determine whether the real problem is:
  - only one side of the pair appearing in the region
  - parent and child regions both partially owning the same handshake
  - a slot-family lineage change caused by reusing a live buffer as scratch
  - a counter / slot token mismatch that moved later ops onto a different family
- report that concrete mismatch back to the user, not just the presence of the warning

Preferred feedback shape:
- which side reported the imbalance: cube or vec
- which family looks broken: `ubin`, `ubout`, `l1`, `l0`, or `fix`
- which local buffer family or region caused the mismatch
- whether the likely fix is region restructuring, buffer separation, or counter repair

Why this rule exists:
- the warning is often the first sign that the mental model is wrong
- users need the suspected broken pattern to decide whether to change the kernel structure, the helper scratch plan, or autosync assumptions

## 11. Verification workflow

When replacing manual barriers with `auto_sync()`, validate in this order:
1. inspect inserted event signatures in IR
2. inspect generated C++ event declarations
3. run a minimal reproducer with the same pipe topology
4. rerun the real kernel shape

Check these details explicitly:
- event family names match the intended stage (`ubin`, `ubout`, `l1`, `l0`, `fix`)
- `valid` uses reversed pipe direction with `preset=True`
- `ready` uses forward pipe direction with `preset=False`
- nested scopes do not create duplicate event families unless slot grouping truly changed
- no `NOT balanced auto_sync events` warning appears

## Files to study

- `easyasc/parser/asc_autosync.py`
- `easyasc/parser/asc.py`
- `easyasc/utils/sync_key.py`
- `easyasc/decorators.py`
- `testcases/parser/sync/test_autosync_event_metadata.py`
- `agent/example/kernels/a5/vec_cube_abs_sqrt_matmul.py`
- `agent/example/kernels/a5/vec_cube_vec_scale2_abs_add1_matmul.py`
