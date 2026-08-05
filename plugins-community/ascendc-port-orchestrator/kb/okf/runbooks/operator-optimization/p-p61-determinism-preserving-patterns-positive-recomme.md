---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Determinism-preserving patterns (positive, recommended)"
description: "Use these when DET_POLICY=required: ### P-P61.1: Hardware Sort with canonicalized tie order - AscendC hardware Sort is deterministic given deterministic input ordering - BUT its native tie-break may n"
confidence: single_run
original_id: P-P61
timestamp_inferred: true
tags: [determinism, optimization, sort, cast, compare, select, exp, p-p61, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

Use these when `DET_POLICY=required`:

### P-P61.1: Hardware Sort with canonicalized tie order
- AscendC hardware `Sort` is deterministic **given** deterministic input ordering
- BUT its native tie-break may not match reference (see P-P60 anti-pattern)
- Use post-sort canonicalization (bubble-reorder within ties, or explicit secondary sort) to produce a deterministic + reference-compatible output

### P-P61.2: Single-core per-row dispatch
- Each row processed by exactly one core in a fixed mapping (`row_id mod nblk → core`)
- No inter-core communication during row processing
- Result: for each row, output depends only on its inputs → bit-deterministic

### P-P61.3: Pure VEC pipeline (no atomics)
- `Cast`, `Compare`, `Select`, `Exp`, `ReduceSum` (single-core), `Sort` — all deterministic given fixed input
- Bitonic / radix sort hardware is deterministic given fixed input ordering
- Avoid: `SetAtomicAdd` to GM from multiple cores (non-det ordering)

### P-P61.4: Queue depth=1 for observable outputs
- Queues with depth>1 + pipe reordering can expose partial state if not fully drained
- Safer default: depth=1 for TQue holding observable kernel outputs
- Can use depth>1 for internal compute if drained before next row starts

### P-P61.5: Stable tie convention via explicit secondary sort
- When multiple positions tie on primary key and order matters downstream, do explicit secondary sort by `original_index` (ascending or descending per reference semantics)
- E.g. P-P60's post-walk `cutoff_orig_idx` re-selection using n_drop_tied-th smallest idx

### P-P61.6: Deterministic reduction tree
- Multi-core reduction via tree (not atomic): each core writes to a distinct output slot, host sums
- Or: ReduceSum on single core when row fits UB → trivially deterministic

## P-P61 anti-patterns (NO-GO when DET_POLICY=required)

### A-P61.1: AtomicAdd from concurrent cores to shared GM slot
- `SetAtomicAdd(gm_slot, val)` from multiple cores — hardware-level non-deterministic ordering
- fp32 AtomicAdd is order-sensitive (not associative due to rounding)
- Use when: `DET_POLICY=best_effort` (e.g. scatter_add reference also non-det); avoid when required

### A-P61.2: Unordered multi-core merge with shared top-K buffer
- Multiple cores computing local top-K, merging to shared buffer without barrier / fixed order → race
- Fix: each core's top-K to separate GM slot, host merges in fixed order (or single-core merge)

### A-P61.3: Uninitialized UB / GM scratch between rows
- Stale state from previous row can leak into current row if scratch isn't reset
- Same input may see different stale state across runs (if row scheduling order varies) → non-det
- Fix: `Duplicate(scratch, sentinel, size)` at each row's start; or design algorithm so stale state is never read

### A-P61.4: Reduction tree with data-dependent order
- `if (cond) reduce_A else reduce_B` where cond depends on partial state can produce different order across runs if memory timing varies
- Fix: data-independent control flow for reduction order

### A-P61.5: Over-eager pipe barrier omission
- Missing `PipeBarrier<PIPE_V>()` / `PipeBarrier<PIPE_ALL>()` between dependent ops can leak partial state
- Partial state observation is non-det under hardware scheduling variation
- Fix: conservative barriers, validate determinism by running kernel twice

### A-P61.6: Scalar-pipe self-coherence on `TBuf<VECCALC>` RMW under duplicate-key density (NEW 2026-04-26, op#19 evidence)

**Setup that triggers**:
1. Algorithm requires per-lane sequential accumulation on a UB-resident accumulator (`TBuf<TPosition::VECCALC>`) — typically arises in scatter-add / index_put / fused histogram patterns where a single core sequentially folds many (key, value) pairs into a small set of lanes
2. Inner loop reads-modifies-writes the same TBuf cache line: `tmp = accum.GetValue(lane); accum.SetValue(lane, tmp + v);`
3. Input keys have non-trivial **duplicate density** — there exist k, k+1 (or near-by k pairs) such that `lane(k) == lane(k+1)`. Norm in scatter-with-duplicates workloads
4. No explicit *intra-loop* scalar-self fence between SetValue at iter k and GetValue at iter k+1 (and there is no idiomatic AscendC primitive that provides one — `S_V`/`V_S` are scalar↔vector, not scalar↔scalar)

**Empirical signature**:
- Same kernel binary, same inputs, consecutive invocations produce different outputs — META-NONDETERMINISM (not just non-determinism)
- Worst observed (op#19 HODSA kw-1, 2026-04-26): bit_identical 21-42/46 across 4 same-input observations, max_diff 6414-25775 slots

**Why "by-construction deterministic" claims can fail at this layer**:
The algorithm-level invariant (single-core ownership + sequential-input-order traversal) is correctly expressed in source. The implementation gap is that AscendC's scalar pipe model does not promise sequential consistency across consecutive scalar load+store to the same UB cache line. Architecture intent ("each core walks input sequentially") is correctly expressed; the scalar-pipe primitive used to realize per-element sequential reduction is itself non-deterministic at the cache-line level.

**Why a boundary fence does NOT fix this**:
The race is inside the loop body (between consecutive iterations' GetValue and SetValue), not at the loop boundary. A boundary `S_V` fence between FilterLoop end and the next outer-tile's VEC Cast does not affect intra-loop scalar issue order. Adding it can REGRESS the count (op#19 pp-1 H1 tested this, regressed bit_identical 28→20/46 — relocated contention by occupying scalar-pipe slot resources).

**Fix paths (in order of preference)**:
1. **Move accumulator to scalar register** (P-P21 register-accum precedent) — registers self-snoop by scalar ISA design, unlike UB cache lines. Requires algorithm restructure so that consecutive same-lane accesses are gathered (e.g. via sort), not stochastic
2. **Use VEC primitives for the accumulate** — VEC ops have well-defined dependency tracking (V_V, V_S, S_V). Variant (a) sort+segment-reduce path is the canonical example: VEC bitonic sort + VEC segment-reduce eliminates scalar TBuf RMW entirely
3. **AVOID** `accum.GetValue(lane); accum.SetValue(lane, ...)` patterns on `TBuf<VECCALC>` whenever consecutive same-lane access is possible. If such a pattern is unavoidable, switch the accumulator from `TBuf<VECCALC>` to a per-iteration scalar register

**Cross-reference**:
- Anti-pattern was the failure mode of `P-P67-candidate (HODSA)` — see `patterns/unverified/candidates.md` for the cautionary entry + empirical evidence
- OL-78 (TBuf persistent across tiles is safe) needs an addendum: persistent-state allowance does NOT extend to scalar RMW on the same cache line under duplicate-key density. The TBuf may be persistent; the access pattern is the hazard
- Origin: op#19 IndexPut DEBT-053 chain (2026-04-26) — researcher proposed HODSA → kw-1 implemented faithfully → pp-1 falsified obvious race hypotheses → da-1 isolated the scalar-self-coherence mechanism. Evidence: `output/npukernelbench/src/kernels/19_IndexPut/probe_report_DEBT053.md` + `workspace/indexput/determinism_report.md`

## Orchestrator Phase O1.5: Op-level determinism policy classification

Orchestrator classifies each op's `DET_POLICY ∈ {required, best_effort, n/a}` at the start of the workflow, BEFORE spawning any agent. Policy is propagated to all downstream agent briefs.

### Classification heuristics (apply in order)

1. **CLI override**: `/ascendc-op-gen --det=required|best_effort|n/a <op>` → explicit wins
2. **n/a (skip det checks)**:
   - Reference uses stateful RNG (dropout, random, gumbel sampling) — by definition non-det without fixed state
   - Reference reads from device clock / uninitialized memory
3. **best_effort (monitor, don't fail)**:
   - Reference uses `atomicAdd` / `scatter_add` / `index_add_` / concurrent writes — inherently non-det on same hardware
   - Reference uses reductions with documented `deterministic=False` in its API
4. **required (det expected)** (default when above don't match):
   - Pure functional ops: sort, topk, reduce, gather, scatter (deterministic variant), pointwise
   - Normalization: softmax, layernorm, groupnorm, rmsnorm
   - Activation: gelu, relu, sigmoid, tanh, swiglu
   - Reshape: permute, transpose, repeat, cat
   - Most benchmark-style ops

### Examples

| Op reference | Policy | Reason |
|--------------|:------:|--------|
| `torch_npu.npu_top_k_top_p` | **required** | pure functional sort + mask + cumsum |
| `torch.scatter_add_` | **best_effort** | atomicAdd inherent |
| `torch.dropout` | **n/a** | stateful RNG |
| `F.softmax` | **required** | pure reduction |
| `F.layer_norm` | **required** | pure reduction |
| `embedding_dense_backward` | **best_effort** | scatter_add under hood |
| `torch.sort(stable=True)` | **required** | explicit stable semantics |

### When policy is n/a

- Orchestrator still runs precision check
- Determinism check is **skipped** entirely (don't try to run kernel twice and compare — will always differ)
- Workflow proceeds straight to perf

### When policy is best_effort

- Orchestrator still runs determinism check (monitor mode)
- Non-det is **logged** but doesn't block workflow
- verification.json: `policy_satisfied: true` (any observed outcome is acceptable for best_effort)

### When policy is required

- Orchestrator runs determinism check
- Non-det → `policy_satisfied: false` → aog-determinism-analyzer agent spawned
- Optimizer constraint K tunable (today: K=0 monitor; future: K=∞ hard gate)

## Detection methodology

Run the kernel twice on seed-identical inputs, element-wise bit-exact diff (NaN-aware). Implementation: `src/scripts/determinism_check.py`.

Report structure (in verification.json):
```json
"determinism": {
  "policy": "required | best_effort | n/a",
  "observed_deterministic": true | false | null (if n/a),
  "policy_satisfied": true | false,
  "n_cases_checked": N,
  "n_identical_cases": N,
  "n_diff_cases": N,
  "drift_detail": [
    {"case": <id>, "n_diff_elements": N, "ratio_pct": X, "max_abs_diff": Y, "shape": [..], "dtype": "..."},
    ...
  ]
}
```

## When non-determinism is observed (policy=required, satisfied=false)

Escalation path: orchestrator spawns `aog-determinism-analyzer` agent.

Analyzer's task:
1. Minimum repro (which case, which row, which element)
2. Bisect kernel phases to locate non-det-introducing phase
3. Classify root cause (anti-pattern reference from this file)
4. Report candidate fix + estimated perf impact
5. Does NOT modify code (analyzer-only). Future `determinism-fixer` agent would apply fixes.

## Relation to other patterns

- **P-P42** (hardware Sort pipeline): Sort is det given det input → use P-P61.1
- **P-P59** (tied-threshold buffer): tie-cutoff algorithm must be deterministic, use P-P61.5
- **P-P60** (Sort ASC tie-break reversed): demonstrates P-P61.1 + P-P61.5 combination
- **P-P48** (UB histogram bincount): uses atomicAdd → det policy should be best_effort unless reference requires det
- **OL-83** (torch_npu 1-ULP drift): not a determinism issue, it's a ref vs ref difference

## Evidence

- 9_TopKTopP R3b (2026-04-18): hardware Sort + single-core-per-row + post-walk canonicalization + scalar SetValue emit → **DETERMINISTIC** verified (50/50 bit-exact across two runs, pre-optimization AND post 6-iter perf optimization including nblk 40→56, queue depth 2→1, CHUNK_SIZE tweaks). Evidence that these optimizations in P-P61 envelope preserve determinism.
- Data point supports: hardware Sort + chunked merge architecture preserves determinism across this optimization class for ops classified `required`.
- recurrent_gated_delta_rule kw-1 (2026-06-18, A5 Ascend950PR_957b / arch35, CANN 9.1.T500): **paged-state linear-attention / SSM recurrent decode is deterministic by construction** — each `(batch, head)` chain is processed end-to-end by one core; `final_state` slots are owned 1:1 by chains (`ssm_state_indices` select disjoint state blocks); no `SetAtomicAdd`. 30/30 bit-identical across a double-run. Extends P-P61.2 (single-core per-row dispatch) + P-P61.3 (pure-VEC, no atomics) to the paged-state recurrent-decode shape: per-chain end-to-end ownership + disjoint output slots is the determinism-by-construction invariant for linear-attention/SSM decode — no determinism analysis needed.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/determinism.md（P-P61，convert_patterns_to_okf.py）。confidence 未升格。 -->
