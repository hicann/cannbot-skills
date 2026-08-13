# Optimization Levers

Read this reference only after `agent/playbooks/kernel-optimization.md` routes a
specific hot-stage symptom here. Keep the playbook open for baseline,
verification, and rollback rules.

This file intentionally lives outside `agent/references/patterns/`: these are
trace-driven changes to an already-correct kernel, not build-time dataflow
patterns for constructing a new kernel.

L0 is a preflight legality guard. L1-L13 are **local levers** — they re-time,
de-duplicate, or re-lay-out work that already exists. L14-L16 are **structural
levers** — they change what work and traffic exist (on-chip reduction, cross-core
operand sharing, split-reduction), for when the local levers plateau against a
still-reducible traffic or occupancy wall (see the Plateau Protocol in the
playbook). Apply one lever at a time and record the result in the optimization
log.

---

## L0. Keep the device API legal

Before trusting a trace, make sure the kernel uses only the public facade for
its target:

- A5 kernels import from `easyasc.a5` and use A5-exposed APIs.
- A2 kernels import from `easyasc.a2` and use A2-exposed APIs.
- Do not direct-import `easyasc.stub_functions.vec.*` or vector APIs from the
  other target to make a tuning attempt work.

An illegal API path can be useful only as a diagnostic experiment. It is not an
acceptable final optimization unless the public facade is updated separately.

### L0b. Every board comparison ships a control

A number remembered from earlier in the day is not a baseline. Shared boards
drift: other users' jobs, a re-installed CANN, a different device index. Rebuild
the **previous** kernel unchanged and measure it in the same session, on the same
device, interleaved with the candidate. If the control has moved, the candidate's
number means nothing yet.

Two specific traps on a multi-device board:

- Devices degrade independently, and a *reference* op is not a health check. On
  a contended `npu:0` an a5 vector-only kernel ran 15x slower while the CANN
  built-in it was compared against moved under 2% — the built-in was not
  competing for the same vector cores. Sweep every visible `/dev/davinci*`.
- `torch_npu.npu.synchronize()` and `torch_npu.npu.Event` act on the **current**
  device, not on the device the tensors live on. A harness with a `--device` flag
  must call `torch_npu.npu.set_device(index)` before timing, or it silently
  reports launch overhead instead of kernel time.

---

## L1. Remove scalar tensor traffic

If a GM value is a true one-cell scalar that only feeds scalar expressions or
VFs, load it into a `Var` instead of staging it through a one-element UB tensor.

```python
# before
ub_last = Tensor(DT.float, [1, 64], Position.UB)
gm_to_ub_pad(ub_last[0:1, 0:1], src[..., i:i + 1])

# after
last = Var(0.0, DT.float)
last.GetValueFrom(src[..., i:i + 1])
```

Precondition: the value is a scalar GM cell. Do not use this for a short vector
or tile. If you keep a UB staging path, keep the parent row-scalar allocation
aligned (for example `[1, 64]`) instead of allocating a legacy `[1, 1]` UB row.

---

## L2. Fuse layout conversion into the producer

Repeated ND-to-NZ or compact-layout staging before matmul is a common hot path.
When a producer `@vf` already computes the data row-by-row, make it write the
layout the consumer actually needs.

```python
@vf()
def produce_consumer_layout(..., dst_nz, ...):
    # compute the logical value
    # apply the required cast/round order
    # write directly in the compact layout expected by the consumer
```

Preconditions:

- The consumer really expects the compact layout at that slot.
- The producer can write that layout without changing the logical value.
- The produced value has already gone through the same cast/round steps as the
  reference.

Risk: `ub.nz()` and `dst <<= src.nz()` label a layout; they do not pack
row-major data into NZ. The bytes must already be in the claimed layout.

---

## L3. Align precision before deleting staging paths

Performance improvements that change cast order are semantic changes. Before
removing bf16/fp32 duplicate paths or fusing producers, compare the kernel with
the reference formula at the exact operation boundary being optimized.

Safe pattern:

- If the reference uses `float(bfloat16(x))`, the kernel must preserve that
  bf16 rounding before the consumer.
- If the reference uses full fp32 `x`, do not add a bf16 round just because it
  allows a cheaper layout path.
- Delete duplicate bf16/fp32 staging only after deciding which value the
  consumer is contractually allowed to see.

Risk: loop-carried updates amplify tiny cast-order drift. A result that is
within tolerance may still be unacceptable when the intended contract is bitwise
match.

---

## L4. Tighten synchronization to real lifetimes

Synchronization should describe the actual ownership boundary, not historical
fear from a previous version.

Useful edits:

- Remove middle `bar_all()` calls when mutexes/events already protect the
  handoff and the surrounding lifetime is still bounded.
- Keep the drain at the outermost reused lifetime boundary. For example, when
  a loop reuses state across an inner tile dimension but then switches to a new
  outer work item, that switch usually needs an explicit drain.
- Set `VcMutex` / `CvMutex` start and end pipes to the real producer and
  consumer pipes that own the protected data. The `src_*` pipes protect the
  producer side; the `dst_*` pipes protect the consumer side. Start pipes place
  the wait before touching the slot; end pipes publish readiness/free after the
  relevant pipe is finished with it.
- For a vec-to-cube L1 handoff on devices that allow explicit wait pipes, if
  vec publishes data with `ub_to_l1`, `ub_to_l1_nd2nz`, or `ub_to_l1_nz` and cube only reads
  that L1 slot through `l1_to_l0`, the protected producer pipe is MTE3 and the
  protected consumer pipe is MTE1:

  ```python
  vcmutex = VcMutex(
      flag_id,
      src_start_pipe=Pipe.MTE3,
      src_end_pipe=Pipe.MTE3,
      dst_start_pipe=Pipe.MTE1,
      dst_end_pipe=Pipe.MTE1,
  )
  ```

  This is valid because the L1 slot can be reused once the MTE1 load has
  completed; later `M` / `FIX` work consumes L0 / L0C state, not that L1 slot.
  Keep a later endpoint such as `Pipe.FIX` when the protected resource really
  stays live through later cube pipes.
- On b-series devices, `wait_cube(...)`, `wait_vec(...)`, and mutex
  `src_start_pipe` / `dst_start_pipe` must use `Pipe.S`. End pipes can still
  describe the real producer or consumer completion pipe. Because a `Pipe.S`
  wait blocks the main scheduling stream, place `lock()` immediately before the
  protected producer instruction and `wait()` immediately before the protected
  consumer instruction to avoid blocking unrelated work too early. Keep
  `ready()` immediately after the producer completes and `free()` immediately
  after the consumer completes.

Risks:

- Simulator success does not prove every cross-iteration lifetime is safe.
- A mutex pipe is wrong if any later producer or consumer on another pipe still
  touches the same protected slot.
- On b-series, an early `Pipe.S` wait can serialize more work than intended
  even when the data-dependency is correct.

---

## L5. Shrink data scope and initialization

Do not load, initialize, publish, or expose data that no later consumer needs.
Wide temporary tensors are often convenient during bring-up, but they turn into
UB pressure and datamove once the schedule is stable.

Useful edits:

- Load source data directly into the compact tiles that the compute path will
  consume instead of loading a full matrix and slicing blocks out of it later.
- Initialize only rows or blocks that are read before being overwritten. For a
  loop-carried resource, a seed row or scalar may be enough; a full
  identity/zero tile can be unnecessary traffic.
- Remove public outputs and input arguments that only existed to inspect an
  intermediate during bring-up. Kernel-visible inputs and outputs should match
  the final contract, not debug scaffolding.
- Hoist invariant tensors, masks, and lookup rows into per-core local buffers
  when reuse beats reloading or recomputing them.

Risk: shrinking initialization is safe only when every skipped cell is
definitely overwritten before read. Validate with a shape that reuses buffers
on the same active core.

---

## L6. Remove duplicated staging and dead paths

After layout and precision are settled, look for staging paths that exist only
because of how the kernel is currently structured: two buffers carrying the same
logical value in different dtypes or layouts, or a single staging path whose
only purpose is to apply an elementwise transformation before cube. Keep only
the path the consumer actually needs.

Typical candidates:

- bf16 and fp32 versions of the same logical or loop-carried value
- separate row-major staging followed immediately by compact-layout staging
- temporary L1/UB buffers kept only for an older consumer that no longer exists
- a vec-side elementwise scale/cast intermediate whose only consumer is the
  next cube matmul: consider letting cube consume the un-transformed operand
  and applying the scaling row-wise in vec postprocess on the cube output. The
  entire vec-to-L1 publish path for the intermediate disappears, and the cube
  inputs become the original tensors instead of a packed scaled tile.

Risk: do not delete a path until all consumers have been audited. In mixed
cube/vec kernels, a value can be read by a later pipe even when the immediate
trace region looks finished. Restructuring a pre-cube transformation across the
cube boundary is also a precision change: the value crosses cube without the
prior rounding step, so validate against the reference at the new boundary
before keeping the rewrite.

---

## L7. Reuse computed intermediates

Repeated matmuls or repeated first-stage products are often more expensive than
keeping the intermediate on chip. If a later formula uses the same cube product
again, cache the product in L1/L0C-compatible form and feed the later matmul
from that cached tile.

Useful edits:

- Replace duplicate cube products with one producer matmul plus L1 reuse.
- Prefer `L0C -> L1 -> L0` reuse for cube-to-cube dependencies when no vec math
  is needed between the stages.
- Add the explicit `FIX -> MTE1` fence or ready/valid pair required by the
  reused L1 slot; `auto_sync()` does not create that edge.

Risk: caching only helps if it shortens the critical path or removes enough
traffic to offset the extra lifetime management. Reused slots need their own
counter and synchronization story.

---

## L8. Split stages to expose overlap

When one large stage blocks a later hot stage, split it at a real data boundary
so the consumer can start on early-ready tiles while the producer prepares the
rest. This is useful for vec setup feeding cube, cube feeding vec postprocess,
and one-tile lookahead pipelines.

Useful edits:

- Split independent VF setup into stage-specific helpers and publish early
  operands with separate mutexes.
- Inside a VF, use a bounded `unroll()` factor over independent rows/chunks and
  stage operations across those lanes so the compiler can overlap load,
  compute, and store latency.  Check the emitted instruction order: expanding
  a loop while leaving a complete dependent chain per lane does not expose the
  intended ILP.  Compare at least one smaller factor when register pressure or
  delayed publication could dominate, and keep the variant only after board
  timing.
- Use separate counters for producer and delayed consumer lifetimes.
- Use a depth that matches the real slot family when a lookahead queue is
  intentional.
- Vectorize row postprocess work into row-wise VF passes instead of scalar
  per-element stores when the row shape is regular.

Risk: stage splitting increases synchronization surface. Do it only at a
formula boundary where partial results are complete enough for the consumer.

Correctness rule for deeper lookaheads: when a stage queue is `lookahead` groups
deep, every workspace ring **and** its `CvMutex` / `VcMutex` depth must satisfy
`slots > lookahead`, or the producer aliases the slot the consumer is still
reading. This is simulator-observable, but only at a shape that wraps the ring
past the lookahead — tiny smoke shapes pass even when undersized. Re-validate at
a ring-wrapping shape whenever you change the depth (see
`agent/references/facts-simulator-opexec.md`).

---

## L9. Pingpong loop-carried resources

When each tile or iteration consumes the previous value of a resource and
produces the next value, use the matching buffer family (`DBuff`, `TBuff`,
`QBuff`, or explicit slots) to separate read and write lifetimes.

Useful edits:

- Replace copy-back VFs or full-slot moves with alternating read/write slots.
- Keep counters tied to one slot family and one reuse lifetime.
- Validate with a stress shape where the same active core reuses the resource
  more than once.

Risk: pingponging is safe only when the old slot is never read after the write
slot becomes current. If two consumers have different lifetimes, they need
separate ownership and synchronization.

Adding slots is not free on `L0C`: grouping several matmul M-tiles to reuse
operands doubles `TILE_M` and can fill all of `L0C`, collapsing the `L0C`
`DBuff` ping-pong the cube overlap depends on — a large hardware regression a
cycle trace does not show (an A2 attention `M_GROUP` probe ran ~80 ms vs ~1.5 ms).
Prefer a *continuous* pipeline that keeps the tile size and a single `L0C`
`DBuff` over one that widens the tile.

Conversely, on **UB** (192 KB, usually slack) adding `DBuff` slots is the cheap
win the L0C case is not. dflash_anchor_mask put its MTE2 load targets (boundary
column buffers) and its MTE3 store source (`u8_out`) on `DBuff`, ping-ponged by
chunk index, leaving the middle scratch single-slot (it lives entirely in V).
That overlapped the three pipes (MTE2 load ‖ V compute ‖ MTE3 store) and ran
**1.5-1.7x faster on real 910B3 across all shapes** (regression 90.2->56us,
huge[128MB] 814.8->479.9 — bigger shapes win more). Two cautions: (1) verify the
HW speedup on a *quiet* box — a busy box made this same DBuff look *slower* than
baseline (HBM-contention noise; follow the multi-tenant-trap protocol in
`agent/references/cycle-model.md`), and note the sim cycle model
over-sells the overlap by ~13%. (2) auto_sync keys a `DBuff` slot by its
*symbolic index var*, so a slot written `buf[cb]` in one loop and read `buf[cs]`
in another gets two keys and the cross-loop WAR is NOT bridged — keep an explicit
`bar_all()` between such loops. (This is distinct from the tensor-slice keying
fix in `sync_key.py`, which made plain-`Tensor` slices share their parent's key;
the `DBuff`-symbolic-index case is still live.)

---

## L10. Pull independent loads across waits

When a GM-to-local datamove is independent of a cross-pipe handoff, issue it
before the wait so it can overlap with producer-side work. Keep the wait
immediately before the first instruction that touches the protected resource.

Risk: do not move a load across a wait if the address, mask, predicate, or
destination slot depends on the protected handoff.

---

## L11. Reduce GM writes

If the user-visible contract only requires a final value, keep intermediate
values on chip and write GM only at the contract boundary.

Risk: do not remove or delay a GM write that is a user-visible output, feeds a
later launched kernel, or participates in synchronization through GM state.

Halving a GM bridge's dtype (e.g. an fp32 intermediate workspace -> fp16) only
helps when the binding pipe is genuinely GM-bandwidth-bound. If that pipe's high
occupancy comes from compute or pipeline latency instead, the byte reduction is
timing-neutral and only costs precision — confirm bandwidth-bound from the
per-pipe occupancy trace before trading precision for bytes.

---

## L12. Normalize datamove syntax last

Tensor assignment syntax is useful once the dataflow is stable:

```python
l1_slice <<= ub_slice.nz()
ub_tensor <<= l0c_tensor
ub_scalar_row <<= gm_slice
```

This is mainly readability and intent normalization. It should not be mixed
with a performance edit in the same measured step.

Risk: assignment syntax inherits the same layout and dtype requirements as the
explicit datamove. It is not a conversion primitive.

---

## L13. Coalesce GM access in SIMT bodies

When a SIMT body must trade off between contiguous UB-side access and contiguous
GM-side access (transpose-like patterns), prefer keeping the **GMTensor read or
write contiguous**, even at the cost of a strided UB access.

UB random access stays inside local memory and is cheap; GM bandwidth is dominated
by burst-friendly contiguous patterns, so coalesced GM read/write is the access
shape that produces the real performance gain on hardware.

```python
# preferred — decompose into the *output* layout so GM write is contiguous
@simt()
def transpose_simt_contig_write(in_ub: Tensor, out_gm: GMTensor, M: Var, N: Var):
    total = M * N
    for i in range(simt_thread_id(), total, simt_thread_num()):
        col = i / M
        row = i % M
        out_gm[i] = in_ub[row * N + col]   # GM contiguous, UB strided
```

```python
# slower — decomposing into the *input* layout makes UB contiguous but strides GM
@simt()
def transpose_simt_contig_read(in_ub: Tensor, out_gm: GMTensor, M: Var, N: Var):
    total = M * N
    for i in range(simt_thread_id(), total, simt_thread_num()):
        row = i / N
        col = i % N
        out_gm[col * M + row] = in_ub[row * N + col]  # UB contiguous, GM strided
```

Use `@simt()`, not `@vf()`, for bodies that call `simt_thread_id()` or
`simt_thread_num()`.

Reference kernels:

- `agent/example/kernels/a5/simt/simt_matmul_transpose_contig_write.py` (preferred)
- `agent/example/kernels/a5/simt/simt_matmul_transpose_contig_read.py` (kept for A/B comparison)

---

## L13b. Shape the work inside a vector function, not around it

A `@vf` launch is expensive relative to the work in a small one - on an
Ascend 950 an AdaptiveAvgPool3D upsample case spent roughly 180 cycles per
launch, having been written as one launch per output row. Moving the row loop
inside the vector function and hoisting the lane-group loop out of it (one
launch per lane group instead of `rows x groups`) cut the small public cases by
2-3.6x.

Three constraints shape how far that goes.

**Loop form.** The a5 vector pipe likes a constant trip count: a constant bound
beats `unroll(constant)` beats `loop(var)`, which is the slowest. `unroll()` is
plain `builtins.range`, so it needs a compile-time constant - a runtime `Var`
bound always lowers to the slow form. If a loop's trip count is one of a few
known values, specialising the kernel on it is a real lever; if it is genuinely
dynamic, expect the loop overhead.

**No scalar division, no scalar compare.** The micro pipe implements only
`var_add`, `var_sub`, `var_mul`, `Min`, and `Max`, so any per-row index that
needs `//` or `%` has to arrive as a recurrence. `floor(n / d)` and `n % d` step
forward with

```python
r <<= r + step_mod
q <<= q + step_div
carry <<= Max(0, Min(1, r - d + 1))     # r, step_mod < d, so r + step_mod < 2d
q <<= q + carry
r <<= r - carry * d
```

**Where the recurrence sits is worth measuring.** Deriving the second window edge
from the first (`end(o) = ceil((o+1)n/d)` is the advanced quotient plus
`Min(1, r)`) halves the scalar assignments, but forces the recurrence to the
*top* of the row loop, ahead of the work whose trip count depends on it, where it
can no longer overlap the previous row's vector work. Whether that trade wins is
shape-dependent and has to be measured - do not assume the cheaper scalar form is
faster. Redundant scalar work that overlaps can beat minimal scalar work that
serialises.

> An earlier revision of this section claimed the top-of-loop form measured 14x
> slower on board. That number came from a contended device and did not survive a
> same-session control build; treat the direction as unmeasured. See L0 - every
> board comparison needs a control build measured in the same session, on the same
> device.

**Once the launches are gone, a reduction `@vf` costs what its `ub_to_reg_*`
loads cost, and nothing else.** Measured on an Ascend 950, every step against a
same-session control build. The profiler reports the vector pipe saturated
(`aiv_vec_ratio` 0.935-0.983, `aiv_mte2_ratio` 0.16) - but "the vector pipe is
busy" turned out to mean "it is busy servicing UB reads", not "it is busy doing
arithmetic". Solving for the two costs across four builds of the same fp32
accumulation loop gives roughly **25 cycles per `ub_to_reg_normal` of a full
256-byte register, and 0.4 cycles per register ALU op**:

| inner loop, per tap | measured on the 268 MB fp32 case |
|---|---|
| 1 load + 7 ALU (Kahan) | 27.2 cycles/tap |
| 1 load + 5 ALU (residual sum) | 26.6 |
| 1 load + 1 ALU (plain sum) | 25.0 |
| 2 loads + 6 ALU (one-ahead prefetch) | +20% overall |
| 2 loads + 5 ALU covering **2** taps (paired) | unchanged |

Read that table before optimizing any `@vf` reduction:

- **Deleting arithmetic buys almost nothing.** Cutting six of seven ALU ops per
  tap - a 3x reduction in instruction count on the phase that owns 60% of the
  kernel - moved the whole operator 2.5%. Compensated summation is therefore
  nearly free; do not trade accuracy for it.
- **Mask registers are free too.** `compare` + `select`, and `mask=` on
  `cast`/`add`, all measured 1.00x when removed.
- **What costs is a load issue, not the bytes.** Pairing two taps into one
  iteration (same loads per element, half the iterations, two independent
  dependency chains) measured exactly neutral, which rules out both loop
  overhead and load latency as the limiter. Adding a *second* load per tap is
  what costs.
- **A `ub_to_reg_gather` prices the same as a `ub_to_reg_normal`.** Measured by
  routing an identity gather through the same loop: 0.995-1.001x on every case
  above the dispatch floor. That is the lever for narrow rows - one gather can
  collect *one row's slice from each of several rows* for the price of one
  ordinary load, so packing `LANES / row_stride` output rows into a register
  divides the load count by that factor without touching the DMA. Prefer it over
  interleaving planes in UB, which would need one burst per row and trade the
  vector win for a DataCopyPad one.
- Consequently the only lever left on such a loop is **fewer load issues per
  input element**. fp16/bf16 pay double here: `ub_to_reg_unpack` moves 64 values
  = 128 bytes per issue because `cast` consumes the even lanes of the half
  register, so a half-precision tap costs the same issue as an fp32 tap for half
  the data. That is visible in the numbers - the bf16 case runs at 3x the
  cycles-per-byte of the fp32 one and is completely insensitive to every change
  above.

---

## Structural levers (L14-L16): change the work, not just the schedule

L1-L13 preserve the algorithm — they re-time, de-duplicate, or re-lay-out work
that already exists. L14-L16 change *what work and traffic exist*: they keep a
loop-carried reduction on chip, make cores share an operand instead of each
re-reading it, or split the reduction dimension for occupancy. Reach for them
when the local levers have plateaued against a traffic or occupancy wall that is
still reducible — see the Plateau Protocol in the playbook. They are larger edits
that usually move reduction or cast order, so apply one at a time and re-validate
correctness against the reference at the new boundary (L3).

---

## L13c. Stage packed and address with a gather

`gm_to_ub_pad` rounds every burst up to 32 bytes in UB. A tile staged one row
per burst therefore pays two prices: the rows come out padded to a 32-byte
pitch, and the transfer costs one burst per row. That burst count is what a
small-row shape actually spends its time on -- on `adaptive_avg_pool_3d` a
23-wide bf16 work item was issuing 69 staging bursts to move 1587 elements.

A row of a contiguous NCDHW tensor is contiguous with the next one, so the whole
run moves as **one** burst and lands packed at the true row width. What that
costs is addressing: a packed row does not start on the 32-byte grid, and
`ub_to_reg_normal` cannot reach it (see a5 constraint 8b). `ub_to_reg_gather`
can, at the same price as an ordinary load -- one extra scalar `adds` to build
the index, against ~25 cycles for the load it feeds.

The same argument runs on the output side: scatter the result into a packed out
tile and the item leaves as one burst instead of one per row.

Two traps on the store side:

- **A masked *continuous* store rounds its active lane count up to a whole
  32-byte block**, so a masked `reg_to_ub_normal` of `n` lanes really writes
  `align8(n)` of them (b32). That spill is harmless while rows are written in
  increasing order -- it lands on a row the loop has yet to write -- but not
  when a row spans several lane groups and the groups are the outer loop,
  because then the last group of row `r` lands on the already written first
  group of row `r+1`. `reg_to_ub_scatter` does **not** do this: its mask is
  lane-wise on the hardware and it writes exactly the active lanes
  (board-verified, `agent/example/kernels/a5/vec_only/scatter_mask_probe.py`; the simulator
  used to expand it and was fixed to match).
- **A zero source stride means "bursts back to back, each rounded up to 32
  bytes".** So an un-fused `ub_to_gm_pad` out of a padded UB tile only lines up
  when the pitch is exactly `align_up(row, 32 / element_bytes)`.

## L14. Keep the loop-carried reduction on chip

When a reduction carries an accumulator (or per-step scores / partial products)
through a **GM workspace** — written and read back every step — and the per-pipe
trace shows a datamove pipe (`MTE2`/`MTE3`) bound, keep the loop-carried value
resident in UB (or L0C) across the reduction and touch GM only at the contract
boundary. This removes both the write and the read-back each step, which is often
the dominant traffic in a low-arithmetic-intensity reduction (for example a
decode-shaped attention with few query rows, where the accumulator round-trip
outweighs the actual math).

Structure (not literal API):

```python
# before: the accumulator round-trips a GM workspace each step
for kv_tile in kv_tiles:
    partial = compute(q, kv_tile)      # this step's contribution (cube + vec)
    acc_ub <<= accum_ws                # read the running accumulator back from GM
    acc_ub  = combine(acc_ub, partial) # vec add / online-softmax rescale
    accum_ws <<= acc_ub                # write it out again — every step
out <<= accum_ws

# after: the accumulator stays resident; GM is touched once
acc_ub = Tensor(DT.float, [...], Position.UB)   # loop-carried, on chip
# init acc_ub to the reduction identity
for kv_tile in kv_tiles:
    partial = compute(q, kv_tile)
    acc_ub  = combine(acc_ub, partial) # no GM round-trip
out <<= acc_ub                         # write GM once, at the contract boundary
```

Preconditions:

- The accumulator plus every value that must stay live across the step fit the
  on-chip budget at the target tile. Bring-up kernels often leave UB largely free
  once wide temporaries are gone — check the real headroom before assuming it
  does not fit.
- The on-chip reduction order preserves the reference's reduction and cast order,
  or the drift stays inside the allowed tolerance (L3).

Relation: L14 extends L11 (which stops at the final user-visible write) to
loop-carried *intermediate* workspaces, and complements L7 (cube-product reuse).

Risk: on-chip accumulation can change reduction/rounding order — validate against
the reference at the new boundary, not only at the final output. L0C slots are
scarce and adding them can collapse the cube `DBuff` ping-pong (see L9's L0C
caution); prefer UB residency for the accumulator when the cube overlap depends
on L0C.

---

## L15. Read each shared operand once across cores

When work is split across cores by a dimension that leaves every core re-reading
the **same GM region** — common at low batch / low occupancy, e.g. each
head-group core re-reads the full KV — the aggregate GM read is
`operand_bytes × core_count`, a redundant-read wall the trace shows as `MTE2`
bound. Restructure so each shared byte is read from GM once and reused across its
consumers.

Options, cheapest first:

- **Let L2 absorb it.** Many cores reading the *same addresses* often hit L2
  rather than HBM, so the redundant reads may already be far cheaper than the byte
  count implies. Measure L2 vs HBM before building an explicit sharing scheme —
  the wall may be smaller than it looks.
- **Broadcast through L1 / shared load.** Have one core (or one load) bring the
  shared operand tile on chip and reuse it across the consumers, arranged so the
  reduction consumes the tile before it is evicted.
- **Re-tile** so the shared dimension is the inner loop and the shared operand is
  loaded once per outer step instead of once per consumer.

Precondition: the operand is genuinely shared (identical addresses) across the
cores, not merely similar.

Risk: an explicit broadcast / shared-L1 scheme adds cross-core synchronization
and can trade a read wall for a sync wall — measure, do not assume. If L2 already
absorbs the reuse, a hand-built sharing scheme adds complexity for little gain;
confirm the reads actually miss L2 first.

---

## L16. Split the reduction dimension to raise occupancy

When the kernel is **occupancy-bound** — it activates far fewer cores than the
device has because the parallelized dimension is small (e.g. decode with few query
rows), and splitting that dimension further is exhausted — the long reduction
dimension (KV / K) is the remaining source of parallelism. Split it across cores:
each core reduces its slice to a **partial** result plus the bookkeeping needed to
merge (for softmax: partial output, running max, running denominator), then
combine the partials in an N-way merge. This is flash-decoding: it buys occupancy
the row/head split cannot reach, at the cost of a cheap final merge.

Preconditions:

- The reduction admits a mergeable partial form. Online softmax merges as
  rescale-by-max then weighted-sum of the partial outputs; a plain sum / max
  merges directly.
- The merge cost (one pass over the per-core partials) is small next to the
  occupancy gained. Sweep the split count — over-splitting makes the merge
  dominate.

Choose the merge independently from the split:

| Merge mechanism | Use when | Required proof |
|---|---|---|
| Single on-chip owner | Partials already share one owner and fit in its local memory | ownership and capacity |
| GM atomic | The device, public API, dtype, operation, and store path all support the update | real hardware compile and run |
| Disjoint GM workspace + owner | Producers can publish separate partials and one owner can reduce them | workspace visibility, owner launch, and synchronization |
| Separate launch | The public contract permits another launch and its overhead is acceptable | end-to-end correctness and timing |

Do not reject the split merely because one merge mechanism fails. Preserve the
same partial contract, then try another legal ownership path within the candidate
budget. Conversely, a simulator pass does not make an atomic or ownership path
hardware-legal. For A2-specific launch-side rules, see
[`constraints/a2.md` section 5.4](../constraints/a2.md#54-split-reduction-merge-ownership).

Risk: the merge has its own precision story — the rescale or reduction order is
a cast-order change (L3), so validate the merged result against the reference.
Split-KV is more complex than row/head splitting; reach for it only when the
simpler split is occupancy-exhausted and the reduction dimension is the
parallelism that remains.
