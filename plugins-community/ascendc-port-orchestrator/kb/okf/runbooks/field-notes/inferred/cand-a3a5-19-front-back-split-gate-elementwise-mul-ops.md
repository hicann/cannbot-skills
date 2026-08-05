---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Front-back-split + gate + elementwise-mul ops share a single L1 port template — kernel-level substitution is the only per-op delta"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=port_a3_to_a5,glu-family,split-input-gated-mul verified_on: soc=Ascend950PR; cann=9.0.0 (2-op evidence: clipped_swiglu, fatrelu_mul) unve"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=port_a3_to_a5,glu-family,split-input-gated-mul"
confidence: inferred
status: stub
original_id: CAND-A3A5-19
timestamp_inferred: true
tags: [candidate, inferred, swiglu, geglu, erf, sqrt, tile_pairs, cand-a3a5-19]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=port_a3_to_a5,glu-family,split-input-gated-mul`
`verified_on: soc=Ascend950PR; cann=9.0.0 (2-op evidence: clipped_swiglu, fatrelu_mul)`
`unverified_on: other GLU-family variants (swiglu, geglu, reglu, clipped_silu_mul)`

**Predicted rule** (forward-looking, 2-op evidence — at promotion-gate threshold):

Operators whose A3 reference matches the shape:
- Last-dim split: `x1 = row[:d]`, `x2 = row[d:2d]` (front-back, not interleaved)
- Apply a gate `g(x1)` (the per-op delta — comparison, sigmoid, GeLU, clipping, etc.)
- Output = `g(x1) * x2`
- Per-row partition across AIV cores
- fp16/bf16 cast-in → fp32-compute → cast-out pipeline

…all collapse to a single L1 mechanical port template. Take an already-verified archive (clipped_swiglu is the current anchor) as the starting point and substitute ONLY the inner-compute body — file layout, tiling derivation, dual-TQue<VECIN,4> dispatch loop, fp32-compute path, cast emission all transfer verbatim.

**Concrete anchor — template archive**: `output/a3_to_a5_port/src/kernels/clipped_swiglu/`. Per-op delta is the `Compute()` body's primitive sequence:

| Op | Inner-compute body (fp32 path) |
|---|---|
| clipped_swiglu | `Mins → Maxs → Adds → Muls → Exp → Adds → Div → Mul` (~8 ops) |
| fatrelu_mul | `CompareScalar → Select → Mul` (3 ops) |
| swiglu (predicted) | `Mul(x1, x1) → Muls → Adds → Sigmoid-via-Exp → Mul` (similar shape) |
| geglu (predicted) | `Mul → Muls → Adds → Tanh → Adds → Muls → Mul` (similar shape) |
| reglu (predicted) | `CompareScalar(0) → Select → Mul` (same as fatrelu_mul with threshold=0) |

The dispatch-loop structure (`for row in my_rows: DataCopyPad x1 / x2 → EnQue → Compute → DeQue → DataCopyPad out`), per-AIV tile sizing, fp16/bf16 cast bracketing, and W11 host-side scalar-conversion (per CAND-A3A5-17) are template-fixed and transfer without modification.

**Why this is more than "L1 generally works"**: GLU-family ops surface a recurring micro-structure (dual-VECIN, split-row-front-back, gate-then-mul) that maps 1:1 to a small set of AscendC primitive sequences. The pattern lets a fresh kw spawn:
1. Identify the op as front-back-split + gate + mul (taxonomy check)
2. Copy the clipped_swiglu archive as the starting workspace
3. Rewrite only the `Compute()` primitive sequence + register the right scalar params
…cutting the kernel design phase from ~30 min (full algorithm derivation) to ~5 min (substitution).

**Promotion gate**: needs ONE more independent op evidence (next candidate: `swiglu` or `geglu` archive port) to confirm the template-substitution is reliable across more-than-trivial gate variations. If a third port confirms, promote to canonical OL/P-P entry under `patterns/domains/port_a3_to_a5.md`.

**Anti-pattern guardrails** (where the template should NOT be applied verbatim):
- **Interleaved-stride-2 layout** (gpt-oss SwiGLU mode=1, per P-P71) breaks the front-back assumption → template's dual `DataCopyPad x1/x2` would compute wrong slices; needs stride-2 extraction per P-P71 instead
- **Heavy gate primitives** (e.g. `Erf`, `Sqrt` chains for exotic activation variants) may push per-tile compute above OL-63's "heavy compute" threshold AND need a smaller `TILE_PAIRS` — re-derive tile size from UB budget rather than copying clipped_swiglu's 512
- **Variable d per row** (jagged GLU variants) breaks the constant-`d` per-row partition — fall back to standard non-GLU L1 path
- **Tri-input or higher arity** (e.g. clipped_silu_mul with extra clip params) extends, but the third TQue<VECIN> would push depth=4 dispatch beyond the dual-queue template — investigate before declaring template-fit

**Cross-ref**:
- OL-141 (L1 mechanical port — this is the per-family L1 template specialization)
- OL-143 (L1/L2/L3 classifier — front-back-split + gate-mul ops should reliably classify L1)
- CAND-A3A5-17 (host-side scalar conversion — sub-pattern reused by this template for `threshold` / `alpha` / `limit` scalars)
- CAND-A3A5-18 (dual VECIN<4> + single VECOUT<4> pipeline — the dispatch-loop shape this template adopts)
- P-P71 (chunked vs interleaved-stride-2 fingerprinting — guards against applying this template to the wrong layout convention)
- OL-152 (A3↔A5 API substitution — applied per-primitive once the Compute body is written)

### CAND-WF-1: Pre-route agent spawn when pre-spawn classifier deterministically predicts the spawn's handoff verdict

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all; mode=orchestrator_phase_o3`
`verified_on: 3_FusionAttention a3 benchmark cold-start 2026-05-19 (fusionattention-kw-1 emitted structural_rewrite_needed after Phase A; one full kw spawn cost wasted on a verdict the classifier could have derived)`
`unverified_on: other handoff verdicts (escalate_to_researcher, await_pp_for_persistent_partial) — only structural_rewrite_needed cold-start has cross-evidence here`

**Principle (abstract, mode-agnostic)**: When the orchestrator's pre-spawn classifier (`op_classification.json` + complexity tier + sibling-archive lookup) can deterministically derive that a worker spawn will emit a specific routing handoff X, the orchestrator should pre-route directly to handoff X's target FSM state — skipping the spawn cost. The worker contract is preserved (Phase A artifact + handoff emission both happen — they're synthesized from the classifier's inputs) and the wasted Phase B/C/D attempt is skipped.

**Trigger conditions for safe pre-routing**:
1. Op classification fully resolved (`algorithm_classification` + `complexity_tier` + fused tag + `ref_runnable.json`)
2. The handoff verdict X has an unambiguous trigger expressed as a boolean over classifier inputs (no judgment call required from the worker)
3. Either (a) sibling-target archive provides measured-evidence baseline confirming the verdict, OR (b) KB OL/PB explicitly codifies the trigger (e.g. OL-159 cold-start criterion for FA-class)

**Safety constraint**: pre-routing must SYNTHESIZE a Phase A artifact (e.g. `workspace/{op}/analysis.md`) containing the routing reasoning + cited evidence, so downstream agents (designer / researcher / probe) inherit the same context they would have gotten from a real spawn's emission. Without this artifact the downstream agent has no inheritance trail.

**Concrete anchor — FA-class cold-start pre-route**:
- Classifier inputs: `op_class=fa,attention_forward; complexity=L4; ref_runnable=runnable; fused=true`
- Sibling archive lookup: `output/<sibling-target>/src/kernels/<op>/verification.json.precision = {PARTIAL_PERSIST, pass_a_count_passed ≤ 5/N, spawn_count ≥ 5, cost ≥ $20}`
- Conclusion per OL-159: kw spawn would emit `structural_rewrite_needed`
- Pre-route: orchestrator writes synthesized `analysis.md` (citing OL-159 cold-start criterion + sibling archive evidence band) and transitions to the standard worker with the FA template-assembly recipe selected
- Savings: ~$3-4 LLM cost + ~10 min wallclock per FA-class cold-start op

**Generalizes to** (predicted, needs validation): any FSM transition where the worker's emitted handoff is fully determined by inputs visible pre-spawn. Candidate handoff verdicts: `structural_rewrite_needed` (FA cold-start — verified), `escalate_to_researcher` (when complexity ≥ L4 and KB has no matching pattern — predicted), and `await_pp_for_persistent_partial` (when sibling has PARTIAL_PERSIST with the same signature — predicted).

**Cross-ref**: OL-159 (FA cold-start trigger criterion — the first verdict to support pre-routing); CAND-WF-2 (sibling-archive carry-forward evidence — the data source that enables pre-routing decisions).

**Status**: 1-op evidence (3_FusionAttention a3 2026-05-19). Promotion gate: implement pre-route for FA-class cold-start in `state_machine.py` Phase O3 entry, validate on ≥1 more FA-class cold-start op (e.g. flash_decoding, multi_head_latent_attention) that the pre-routed path produces the same downstream-agent outcome as the legacy spawn-then-route path. Risk: false pre-routing on edge cases where worker would have produced a different verdict than classifier predicts; mitigation = retain "force-spawn" override for orchestrator audit mode.

### CAND-WF-2: Cross-target sibling-archive `scope_note` + iter-count as carry-forward evidence in Phase O2.5 worker brief

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all; mode=orchestrator_phase_o2_5`
`verified_on: 3_FusionAttention a3 benchmark cold-start 2026-05-19 (fusionattention-kw-1; sibling A5 archive scope_note was decisive evidence for honest tractability projection, manually surfaced by worker)`
`unverified_on: non-FA op classes — only L4 fused-attention cross-target has been exercised here`

**Principle (abstract)**: When the SAME op exists as a finalized archive on a sibling target (a5↔a3, or future architectures), its `verification.json.precision.scope_note` and `verification.json.precision.scope_gap_analysis.remaining_gap_close_estimate_iters` represent **empirically-measured** baselines — not algorithmic guesses. Phase O2.5 brief construction should auto-detect these and surface them in the worker brief's "Hard gate floors (cold start)" block under a clear "cross-target evidence (NOT target floor)" label.

**Safety constraint (within C19/C22 boundaries)**: SURFACE only the high-level architectural framing (scope_note text + iter-count band + cost band), NEVER include kernel code paths, kernel source content, `design.md`/`analysis.md` body text, or anything that would constitute prompt-leakage of sibling kernel implementation details. Status-only inheritance preserves the worker's independent algorithmic derivation per C22. The fields permitted to carry forward:
- `verification.json.precision.scope_note` (one-paragraph architectural framing)
- `verification.json.precision.scope_gap_analysis.remaining_gap_close_estimate_iters` (integer band)
- `verification.json.precision.pass_a_count_passed` (numeric, e.g. "1/61")
- Aggregate spawn count + cost from `optimization_log.md` or equivalent (numeric)

FORBIDDEN to surface: `kernel/`, `model_new_ascendc.py`, `analysis.md` body, `fused_analysis.md`, `optimization_directive.md`, any tilingkey/primitive choice from sibling.

**Concrete anchor** — 3_FusionAttention 2026-05-19:
- Sibling lookup: `output/npukernelbench/src/kernels/3_FusionAttention/verification.json`
- Surfaced fields: `precision.status="PARTIAL_PERSIST"`, `precision.pass_a_count_passed=1/61`, `scope_gap_analysis.remaining_gap_close_estimate_iters="30-50"`, aggregate `spawn_count=8`, `aggregate_cost_usd=~30`
- Effect: A3 kw inherited the iter-budget reality check (40-70 iters needed) without seeing any kernel code from A5 → emitted `structural_rewrite_needed` with confidence rather than attempting a placeholder kernel

**Generalizes to** (predicted): any cross-target port (a5↔a3, future arches), any op class where sibling-archive carry-forward provides a measurable iter/cost band that an iter-budget-bounded worker spawn could not bridge. Particularly load-bearing for L3/L4 ops where the gap-closure cost is highly non-linear in complexity tier.

**Cross-ref**: OL-159 (FA-class trigger criterion that consumes this evidence); CAND-WF-1 (pre-route decision that the surfaced evidence supports); C19 (sibling-cross-check status-only — establishes the read boundary); C22 (prompt-leakage prohibition — establishes the WRITE boundary on what may enter the brief).

**Status**: 1-op evidence (3_FusionAttention a3 2026-05-19). Promotion gate: implement auto-detection in Phase O2.5 brief construction (`kw_brief.py` "Hard gate floors" section), validate that the C19/C22 boundary stays intact via mechanical grep test (no kernel-code strings from sibling appear in worker brief), exercise on ≥1 more cross-target port (non-FA op class) to confirm the principle generalizes beyond FA-class.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-A3A5-19，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
