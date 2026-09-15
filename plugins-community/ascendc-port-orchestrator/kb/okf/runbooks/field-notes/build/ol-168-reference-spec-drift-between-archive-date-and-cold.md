---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Reference-spec drift between archive date and cold-restart silently invalidates prior bit-exact verdicts — mtime pre-flight before treating archived kernel as starting material [V351+V220, ALL_MODES, workflow]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-168
timestamp_inferred: true
tags: [ascendc, ol-168]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all`
`applies_to_backend: all`
`verified_on: soc=Ascend950PR; cann=9.0.0 (op#11 DequantSwigluQuant kw-1 cold-restart 2026-05-18)`
`unverified_on: soc=Ascend910_V220 (A3 — same workflow shape, no recorded incident yet)`

### Principle

A bit-exact verdict on an archived kernel is bound to **the specific reference used at archive time**. Vendor `model.py`, `<op>.json` case lists, and `Model.forward` implementations evolve independently of the kernel archive. When a cold-restart spawn reuses an archive as starting material — common pattern: workspace wiped, brief says "copy v3.X archive, re-verify" — the implicit assumption is that the reference hasn't drifted. That assumption is **silent**: nothing in the workflow alerts the worker when it has been violated.

A single mtime comparison catches this before any deploy/build/verify cycle wastes wall-clock. Always pre-flight the reference age against the archive age.

### Detection (pre-flight check, ≤2 seconds)

Before deploying an archived kernel during cold-restart Phase A:

```bash
ARCHIVE_TS=$(stat -c %Y output/<project>/src/kernels/<op>_<tag>/PROGRESS.md)
REF_TS=$(stat -c %Y vendor/AscendOpGenAgent/benchmarks/NPUKernelBench/level*/<op>.py)
[ "$REF_TS" -gt "$ARCHIVE_TS" ] && echo "WARN: reference drifted post-archive — diff Model.forward before reusing kernel"
```

If `REF_TS > ARCHIVE_TS`: open both `Model.forward` implementations side-by-side; diff the algorithm step-by-step. Treat the archived kernel as REFERENCE ONLY until the diff is reconciled. **Do not deploy → build → verify before the reconciliation** — the deploy cycle costs ~5 minutes; the mtime check costs 2 seconds.

### Symptom signature when the check is skipped

The pattern is unambiguous and distinct from numerical-drift FAIL signatures (OL-104 fp16 floor / OL-83 1-ULP):

- Kernel builds cleanly, deploys cleanly, runs cleanly.
- Pass A produces a **correlated FAIL pattern that maps 1:1 to a specific input-attribute value** (an enum: dtype value, mode flag, layout flag, etc.).
- The split is "clean": 100% PASS on one attribute value, ≥50% FAIL on another. NOT a noisy boundary.
- `mismatch_ratio` is large per failing case (>50%/row) and `max_abs_diff` covers a large fraction of the output dtype range (e.g. 200+/255 for int8, full mantissa range for fp32).
- Conclusion: wrong algorithm being executed on the failing attribute branch, not numerical noise.

### Evidence

- **op#11 DequantSwigluQuant kw-1 (2026-05-18, cold-restart)**: archive PROGRESS.md mtime = 2026-04-21, vendor `level2/11_DequantSwigluQuant.py` mtime = 2026-04-27. Six-day drift. Vendor introduced a new mode=1 formula (gpt-oss variant with even/odd split + clamp + α-sigmoid + β-shifted lin) that the archived kernel — designed against the prior "no-op flag" assumption from P-P58.X — does not implement. Result: 25/50 PASS (mode=0 all PASS, mode=1 all FAIL, mismatch_ratio ~63%/row, max_abs_diff 200-254). The mtime check would have caught the drift in 2 seconds before the 5-minute deploy cycle.

### Why the rule exists

Cold-restart is structurally vulnerable to this. Three reinforcing pressures:

1. **Brief framing pressure** — when the brief says "the archive verified at 50/50 PASS; just re-run verification", the worker reads "the kernel is known-good; verification is a formality". This frames the archive as proof rather than hypothesis.
2. **Closure desire (P7)** — cold-restart spawns are often re-runs of prior incidents; the worker is motivated to land the artifact quickly and move on. Pre-flight diff feels like overhead.
3. **Spec drift is invisible** — there's no telemetry that says "the reference moved". The kernel doesn't know what version of `model.py` it was validated against; the `verification.json` doesn't record a reference hash. Drift accumulates silently.

### Other instances (predicted)

- Any benchmark op where the upstream reference (vendor `model.py` / case-list `.json` / golden file) is committed separately from the archive. Vendor updates happen out-of-band relative to op-gen archive runs.
- Long-tail of ops on the NPUKernelBench backlog: workspace wipes, P0xx incidents, finalize-gate rollbacks all produce cold-restart spawns weeks-to-months after the original archive. The longer the gap, the higher the drift probability.
- Cross-target ports where the source-target reference pair drifts independently (port_a3_to_a5: A3 reference advances while A5 archive is dormant; same risk shape).

### Mitigation (decision rule)

1. **Cold-restart Phase A step 0**: run the mtime check above. If `REF_TS > ARCHIVE_TS`, mark the workspace `archive_likely_stale: true` in PROGRESS.md and proceed to step 2.
2. **Algorithm diff**: open archived `analysis.md` (or extract algorithm summary from archived `model_new_ascendc.py`) and current vendor `Model.forward`. Diff step-by-step. List every algorithmic change.
3. **Triage**: if the diff touches the kernel's logic, the archive is REFERENCE ONLY — escalate to `aog-researcher` for a redesign brief, OR `await_user_decision` if the rewrite cost is large enough to need operator sign-off.
4. **Do not** silently `skipped_cases` the failing attribute branch to claim PASS — that's the P0abd coverage-fraud trap.
5. **Do not** label the FAIL as "structural ceiling" or "hardware limit" — it's documented spec drift with a concrete diff (P5 anti-pressure).

### Cross-reference

- P-P58.X (reference-bound mode-flag dispatch — the op#11 incident updated 2026-05-18 to scope-tag the original conclusion).
- OL-68 (Path-A CPU-truth vs Path-B torch_npu-direct reference styles — drift risk differs per case).
- P0abd (coverage-fraud — `skipped_cases` as a band-aid is the wrong mitigation here).
- ANTI_PRESSURE_PROTOCOLS P5 (no structural-blame labelling), P7 (no closure-desire shortcuts).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-168（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
