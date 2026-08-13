# Kernel Optimization Playbook

Use this playbook when an EasyASC DSL kernel is already correct and the next
goal is to reduce simulator trace makespan or another user-named performance
metric.

This is not an initial-authoring guide. For a new kernel, use
`agent/playbooks/pytorch-to-single-kernel.md`.
For a failing kernel, first use `agent/playbooks/kernel-debugging.md`.

This file is the short workflow entry point. For detailed optimization levers,
open only the matching section in
`agent/references/optimization/levers.md`.

Cross-references:

- `agent/references/facts-simulator-opexec.md` for trace makespan accounting
  and simulator configuration.
- `agent/scripts/analyze_sim_trace.py` for quick trace makespan and pipe-utilization
  summaries before opening the full Chrome/Perfetto trace.
- `agent/references/facts-authoring.md` for datamove hard rules and mutex
  depth rules.
- `agent/references/constraints/sync.md` for barrier, mutex, and lifetime
  ownership constraints.
- `agent/references/constraints/precision.md` for cast and rounding-order
  constraints.
- `agent/references/constraints/tiling.md` for matmul operand layout,
  compact publish, and buffer-capacity constraints.
- `agent/references/workflow-state-and-context.md` for candidate state,
  phase-close, background-task, and bounded scout ownership on long runs.

---

## Optimization Target

Optimize the timed trace critical path by default:

```text
makespan = max(ts + dur) over Chrome trace events where ph == "X"
```

If the task names another metric, use that metric and write it next to every
cycle number. Do not optimize the sum of all activated task durations unless
that is explicitly the metric under study.

Keep these invariants throughout the pass:

- Correctness contract comes first. User-visible outputs keep the same shape,
  dtype, tolerance, and bitwise requirements unless the user explicitly changes
  the numerical contract.
- Every cycle number in one table must use the same simulator configuration and
  timing model.
- Measure compose and sub-kernels separately. The compose number tells you user
  impact; per-stage numbers tell you where to edit.
- Warnings are evidence. Investigate synchronization, layout, and precision
  warnings instead of treating a passing output as sufficient.

---

## Entry Protocol

- Use the current task's reproducible runtime environment. Do not mix
  before/after numbers from different environments unless the task is explicitly
  comparing environments.
- Confirm the bench or trace command executes the kernel source being edited.
  Check stale imports, copied helper paths, trace prefixes, and output
  directories before trusting an unchanged number.
- If an ad-hoc simulator runner is needed, put it under `tmp/` and execute it
  as a real script or module. Simulator multiprocessing reloads the main script
  path, so do not launch temporary simulator runners from stdin.
- Keep correctness runs and trace-producing simulator runs serialized when they
  can share output paths, global state, or trace filenames.

---

## Baseline Record

Start only when all of these are true:

- The kernel passes its reference check at the intended tolerance.
- There is a reproducible simulator bench command for compose and for each
  sub-kernel or stage you may edit.
- The chosen target shape is representative enough to expose datamove,
  synchronization, task-launch, and layout-conversion cost. Tiny smoke shapes
  are useful for correctness but often hide performance deltas.
- You know which trace and simulator settings are authoritative for this pass
  before recording the baseline.

Record this before editing:

| Field | Value |
|---|---|
| command |  |
| target shape |  |
| simulator config |  |
| trace path |  |
| metric definition |  |
| compose makespan |  |
| hot stage makespan |  |

After recording the baseline trace, run a quick summary before choosing a
lever:

```bash
python agent/scripts/analyze_sim_trace.py <trace.json>
python agent/scripts/analyze_sim_trace.py <trace.json> --group-by lane-pipe
```

Use the summary to answer three fast questions:

- which pipe family has the highest active/span occupancy
- whether the bottleneck is cube-side `MTE2` or vec-side `MTE2`
- whether `vec0` and `vec1` are balanced or one lane is lagging

For synchronization, buffer lifetime, or loop-carried resource reuse changes,
add a stress shape that forces the same active core to process two or more
tiles or work items that reuse the same resource. Prove this with the current
kernel's tiling formula, for example:

```text
work_items_that_reuse_resource / active_core_count > 1
```

Use the task-specific equivalent when work is distributed by a different
formula.

---

## Symptom Routing

After identifying the hot trace region, open only the matching lever section.

| Symptom in trace or source | Lever |
|---|---|
| Kernel imports helpers outside `easyasc.a2` / `easyasc.a5` | [L0. Keep the device API legal](../references/optimization/levers.md#l0-keep-the-device-api-legal) |
| One GM scalar is staged through a one-cell UB tensor | [L1. Remove scalar tensor traffic](../references/optimization/levers.md#l1-remove-scalar-tensor-traffic) |
| Hot path repeatedly converts row-major data to compact/NZ layout | [L2. Fuse layout conversion into the producer](../references/optimization/levers.md#l2-fuse-layout-conversion-into-the-producer) |
| Proposed speedup changes bf16/fp32 cast or rounding order | [L3. Align precision before deleting staging paths](../references/optimization/levers.md#l3-align-precision-before-deleting-staging-paths) |
| Trace is dominated by waits, barriers, or suspicious handoff placement | [L4. Tighten synchronization to real lifetimes](../references/optimization/levers.md#l4-tighten-synchronization-to-real-lifetimes) |
| Kernel loads, initializes, or exposes data no later consumer needs | [L5. Shrink data scope and initialization](../references/optimization/levers.md#l5-shrink-data-scope-and-initialization) |
| Two dtype/layout paths carry the same logical value | [L6. Remove duplicated staging and dead paths](../references/optimization/levers.md#l6-remove-duplicated-staging-and-dead-paths) |
| Same cube product or first-stage transform is recomputed | [L7. Reuse computed intermediates](../references/optimization/levers.md#l7-reuse-computed-intermediates) |
| A large stage blocks a consumer that could start on early-ready tiles | [L8. Split stages to expose overlap](../references/optimization/levers.md#l8-split-stages-to-expose-overlap) |
| A VF repeats a dependent load/compute/store chain with little lane-level overlap | [L8. Split stages to expose overlap](../references/optimization/levers.md#l8-split-stages-to-expose-overlap) |
| Loop-carried resource copies back into itself | [L9. Pingpong loop-carried resources](../references/optimization/levers.md#l9-pingpong-loop-carried-resources) |
| Independent GM/local loads sit after a wait | [L10. Pull independent loads across waits](../references/optimization/levers.md#l10-pull-independent-loads-across-waits) |
| Intermediate GM writes are not user-visible and feed no later launch | [L11. Reduce GM writes](../references/optimization/levers.md#l11-reduce-gm-writes) |
| Dataflow is stable and only datamove readability remains | [L12. Normalize datamove syntax last](../references/optimization/levers.md#l12-normalize-datamove-syntax-last) |
| A SIMT body strides GM where an output-layout decomposition could coalesce it | [L13. Coalesce GM access in SIMT bodies](../references/optimization/levers.md#l13-coalesce-gm-access-in-simt-bodies) |
| An accumulator or per-step intermediate round-trips a GM workspace every reduction step and a datamove pipe is bandwidth-bound | [L14. Keep the loop-carried reduction on chip](../references/optimization/levers.md#l14-keep-the-loop-carried-reduction-on-chip) |
| Many cores re-read the same GM operand (aggregate read ≈ operand × core count) | [L15. Read each shared operand once across cores](../references/optimization/levers.md#l15-read-each-shared-operand-once-across-cores) |
| Occupancy-bound: far fewer cores active than the device has and the split dimension is exhausted | [L16. Split the reduction dimension to raise occupancy](../references/optimization/levers.md#l16-split-the-reduction-dimension-to-raise-occupancy) |

If no symptom matches, broaden trace-driven tuning: tile shape, DBuff/TBuff
slot pressure, vec micro instruction count, matmul operand dtype, and
critical-path overlap. Record new evidence instead of forcing it into a lever.

---

## Reduction-Split Merge Gate

Before implementing a lever that splits one logical reduction across cores or
workers, choose the merge path as a separate design decision. Record one row:

| Split axis | Partial layout | Merge owner + mechanism | Kernel mode | Visibility / sync edge | Precision change | HW stress case |
|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |

Use the merge matrix in
[L16](../references/optimization/levers.md#l16-split-the-reduction-dimension-to-raise-occupancy).
Do not start from “split first, repair the merge later”: the partial layout,
writer ownership, launch topology, synchronization, and reduction order must be
legal together.

Simulator correctness is not hardware-legality evidence for cross-core atomic
stores or launch-side ownership. Require a real hardware compile and run for the
chosen merge path. If the same failing boundary survives two evidence-driven
fixes, switch merge mechanism or stop that candidate instead of continuing
line-level variants. On A2, also apply
[the split-reduction ownership rules](../references/constraints/a2.md#54-split-reduction-merge-ownership).

---

## Plateau Protocol — local vs structural levers

The levers split into two classes. **Local levers (L0-L13)** preserve the
algorithm: they re-time, de-duplicate, or re-lay-out work that already exists.
**Structural levers (L14-L16)** change what work and traffic exist: keep a
loop-carried reduction on chip, make cores share an operand instead of each
re-reading it, or split the reduction dimension for occupancy.

A plateau against a **traffic or occupancy wall** is a routing signal to the
structural levers, not a stop. When local tuning stops moving the metric and the
per-pipe trace shows a datamove pipe (`MTE2`/`MTE3`) or `FIX` near 100%
active/span, or the kernel activates far fewer cores than the device has:

- That means "stop tuning the *schedule*," not "stop the pass." Ask whether the
  saturating traffic or the idle cores are **reducible**:
  - an accumulator/intermediate still round-trips GM each step → L14
  - cores re-read the same operand → L15
  - occupancy is below the device and the reduction dimension is unsplit → L16
- Name the reducible quantity (redundant bytes per step, redundant reads ×
  cores, unused cores), attempt the matching structural lever, and re-measure.

Stop only at a **fundamental wall proven with data**: the moved bytes are
irreducible (each is read or written once and feeds the contract) *and* occupancy
is at the device limit for the parallelism the algorithm exposes. Record the
measurement that proves it — "`MTE2` is at 98%" alone does not; "`MTE2` is at 98%
and every byte it moves is read exactly once" does. A plateau you cannot yet tie
to an irreducible quantity is a structural plateau, not a wall: try the next
structural lever before declaring it.

Structural levers are larger edits that usually move reduction or cast order, so
keep the one-lever discipline — apply one, re-validate correctness against the
reference at the new boundary (L3), then re-measure under the baseline's
measurement model.

---

## Workflow

1. **Record the baseline.** Save compose makespan, per-stage makespan, bench
   command, target shape, and simulator configuration.
2. **Summarize the trace.** Use `agent/scripts/analyze_sim_trace.py` to identify the
   busiest pipes and lane imbalance before reading the full trace by hand.
3. **Find the hot stage.** Optimize the stage on the critical path first. Leave
   cold stages alone unless they block a hot-stage simplification.
4. **Classify the critical work.** Use the routing table to pick one lever.
5. **Apply one local edit.** The selected lever's preconditions must be true.
6. **Verify before timing.** Run syntax checks when useful, then correctness.
   Then run the same bench as the baseline.
7. **Accept or roll back.** Keep the change only if correctness holds and the
   cycle result is meaningful under the same measurement model. Neutral changes
   are acceptable only when they improve semantic ownership or readability
   without hiding a performance regression.
8. **Record the step.** Log edit, rationale, cycle delta, correctness result,
   and decision.
9. **Normalize last.** Apply syntax/readability cleanup after the performance
   path is stable.

Use a compact step table while tuning:

| Step | Metric | Hot stage | Delta | Correctness | Decision |
|---|---:|---:|---:|---|---|
| baseline |  |  | n/a |  |  |
| edit name |  |  |  |  | kept / rolled back |

Keep target-shape performance rows separate from tiny smoke-shape correctness
rows. If the measured trace is identical after a source edit, confirm the bench
is using the current source and a fresh trace before changing kernel code again.

---

## C->V->C Lookahead Checklist

Use this checklist when a cube result feeds vector work and the vector result
feeds a later cube matmul.

- Pick a stress shape that exercises the intended overlap. For BHC-sharded
  kernels, a default many-core run may give each core only one BHC item and hide
  the lookahead benefit. Use a one-core or same-core multi-item run when the
  optimization depends on a core processing multiple BHC items.
- Split the loop by real readiness. A good one-beat lookahead often has a
  producer beat that computes the cube products and vector transforms needed to
  publish the next cube operand, while the delayed consumer beat contains only
  the dependent matmul and its writeback.
- Keep mutexes on real cross-pipe ownership edges only. Ordinary GM->L1 input
  loads do not need extra mutexes just because they appear near a pipelined
  handoff. Use `CvMutex` for cube/FIX -> vec and `VcMutex` for vec/MTE3 -> cube.
- Release at the earliest pipe that really retires the resource. If a VcMutex
  protects an L1 operand consumed by matmul, `dst_end_pipe=Pipe.MTE1` is usually
  the right release edge; waiting until `Pipe.FIX` can serialize unrelated
  matmul tail work.
- Choose DBuff/TBuff depth from live roles, not from fear. A one-beat delayed
  handoff is usually DBuff. Use TBuff only when one logical value is still live
  across both the current producer beat and the delayed consumer beat, and give
  that slot family its own counter.
- Do not reuse one UB as both input and output source when that couples pipe
  lifetimes. If a VF reads a loaded UB tile and overwrites the same tile with a
  value later stored by MTE3, the next MTE2 load may need to wait for the MTE3
  store. A separate output UB can be cheaper and cleaner than an event.
- Prefer direct `L0C -> GM` for full-tile outputs when the output layout and
  dtype match the destination contract. Avoid staging through UB just to write
  GM unless the vector side genuinely needs to modify the value.
- Use active/span occupancy as a stop signal **for local scheduling** after
  semantic traffic is gone. If MTE or FIX is near 100% active/span, local
  scheduling is data-movement limited — that is the crossover point to the
  structural levers (L14-L16; see the Plateau Protocol), not the end of the pass.
  Stop only when the moved data is also irreducible: each byte read or written
  once and feeding the contract.

---

## Verification Rules

- Compare every user-visible `GMTensor` output returned by the `@kernel`.
- Check shape, dtype, finiteness, and the intended tolerance or bitwise
  contract.
- Preserve cast and rounding order unless the user explicitly changes the
  numerical contract.
- Treat warnings as evidence. Investigate `auto_sync`, layout, precision, and
  buffer-lifetime warnings before accepting a performance result.
- For synchronization and resource-reuse edits, include the stress shape from
  the baseline section before reporting success.

---

## Rollback Rules

Rollback immediately when any of these is true:

- Correctness changes and the user did not explicitly change the numerical
  contract.
- The optimization depends on a layout marker performing real packing.
- The final kernel relies on APIs outside the target public facade.
- A barrier removal crosses an outer lifetime boundary.
- A mutex pipe no longer matches the real producer/consumer lifetime, or a
  b-series `Pipe.S` wait is moved early enough to serialize unrelated work.
- A speedup appears only because the simulator configuration or target shape
  changed.
- A removed staging path still feeds a valid consumer under the reference
  precision contract.
- A narrowed load/initialization skips a value that is read before overwrite.
- A staged-overlap rewrite makes producer and consumer counters describe
  different slot families.
- A pingpong rewrite lets any consumer read a slot after that slot has become a
  later write target.

---

## Reporting

Final optimization reports should state:

- files changed
- before/after metric and metric definition
- commands and correctness checks run
- warnings investigated or left unresolved
- cases not verified, especially missing stress-shape coverage

---

## Provenance

This playbook consolidates reusable optimization lessons and project passes.
Detailed lever rationale lives in
`agent/references/optimization/levers.md`; detailed per-kernel logs
stay out of this playbook once the generalized method exists.
