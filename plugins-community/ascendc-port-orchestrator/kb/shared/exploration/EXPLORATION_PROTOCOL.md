# Bounded Exploration Protocol

9-step protocol for finding kernel optimizations when pattern matching is exhausted.

## Prerequisites

- Stage 2 (pattern-based optimization) completed, target not met
- All matching patterns from PATTERN_INDEX.md already applied or rejected
- msprof data available for the worst-performing benchmark case

## The 9 Steps

### Step 1: Profile
```bash
msprof --output=/tmp/msprof_out -- ./benchmark_binary --case=<worst_case>
```
Extract: `aiv_vec_ratio`, `aiv_mte2_ratio`, `aiv_scalar_ratio`, `task_duration`.

### Step 2: Match Grounding Chains
Compare msprof metrics against GC-1 through GC-7 in `GROUNDING_CHAINS.md`.

### Step 3: Identify Candidate Dimensions
For each matching chain, look up candidate dimensions (D1-D5).

### Step 4: Enumerate Alternatives
For each candidate dimension, list the 3-4 concrete alternatives from `STRUCTURAL_DIMENSIONS.md`.

### Step 5: Filter Already-Tried
Remove alternatives equivalent to already-applied patterns (check PATTERN_INDEX.md).

### Step 6: Formulate Hypotheses
For each remaining alternative, write a structured hypothesis:
```
HYPOTHESIS: [id]
  Dimension: D1/D2/D3/D4/D5
  Change: [concrete description]
  Grounding: [msprof metric → grounding chain → dimension]
  Prediction: [benchmark case] should improve by [>X%]
  Falsification: If [metric] does not change by [>Y%], hypothesis wrong
  Cost: compile ~N min, test ~M min
  UB Budget: [calculation]
  Rollback: revert to commit [hash]
```

### Step 7: Rank Hypotheses
Sort by (predicted improvement / estimated cost). Prefer cheaper hypotheses first when improvement is similar.

### Step 8: Execute Top Hypothesis
1. Create exploration class (separate from production code)
2. Compile
3. **Precision test first** — if FAIL, revert immediately
4. Benchmark
5. Compare msprof before/after — did the predicted metric change?

### Step 9: Evaluate and Iterate
Apply early termination rules (see below). If successful, generalize to pattern library.

## Bounding Rules

### Depth Bound
- **One dimension at a time**: Change D1 OR D2 OR D3 — never two simultaneously
- **Max 3 structural changes** per campaign (D5 parameter sweeps don't count)
- If 3 changes fail to achieve >10% improvement → STOP

### Budget Bound
| Step Type | Compile | Test | Benchmark | Total |
|-----------|:---:|:---:|:---:|:---:|
| D5 parameter sweep (1 config) | ~2 min | ~3 min | ~2 min | **7 min** |
| D1-D4 structural change | ~5 min | ~5 min | ~3 min | **13 min** |
| D1-D4 with new class | ~8 min | ~5 min | ~3 min | **16 min** |

**Campaign budget**: 3 × 16 min + 5 × 7 min = **83 min max**.
If campaign exceeds 90 minutes wall-clock → STOP, document, escalate.

### Early Termination Rules
| Condition | Action |
|-----------|--------|
| Precision FAIL | Revert. Do NOT debug precision. Count as 1 of 3. |
| Performance regression >5% | Revert. Hypothesis was wrong. |
| Performance improvement <5% | Record as "marginal." Counts as 1 of 3. |
| Performance improvement >10% | SUCCESS. Commit. Update pattern library. |
| 2 consecutive regressions | STOP campaign. |
| msprof shows same bottleneck | Transformation didn't address actual bottleneck. STOP this direction. |

## Exploration Class Convention

New variants are separate classes in the same header file:
```cpp
SparseGatherForwardSimdF32PingPong                   // production
SparseGatherForwardSimdF32PingPong_ExplBatchToken    // exploration
SparseGatherForwardSimdF32PingPong_ExplExpertMajor   // exploration
```

Benchmark dispatches via `--variant=ExplBatchToken`. Production code stays untouched.

## Pattern Library Feedback Loop

When an exploration succeeds (>10% improvement, precision PASS):
1. Describe the transformation generically
2. Check if it's a special case of an existing pattern
3. If new, add to appropriate domain file in `patterns/domains/`
4. Update `PATTERN_INDEX.md` with trigger conditions
5. Future kernels benefit automatically
