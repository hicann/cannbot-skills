---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Mirror Phase O5 verifier plugin-skip and tier-normalization logic across the local-container and SSH paths — unmirrored divergence falsely rejects correct kernels at finalize"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3 (local-container verify path)"
phenomenon: precision_issue
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3 (local-container verify path)"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-261
timestamp_inferred: true
tags: [ascendc, ol-261]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3 (local-container verify path)`
`verified_on: soc=Ascend950PR; cann=9.0.0`

**Principle (decision rule)** — applies to ANY Phase O5 verifier that ships two parallel implementations (a remote/SSH path and a local-container path):

> Any plugin-skip, tier-normalization, or runner-output-label logic that one path applies MUST be mirrored in the other. The two paths share one finalize contract; a logic fork landed on only one side is a silent verifier-coverage gap. It surfaces as a false `phase_o5_runner_failed` ("no Pass B verifier found" / `tier2_status` silently dropped) on an otherwise-correct kernel. The rejection reads like a worker or precision defect but is actually a path-divergence bug in the runner — so chasing it inside the kernel is wasted spawn cycles.

Three divergence classes observed (all in `phase_o5_runner.py`'s `_run_verifier_local` vs `ssh_runner`, fixed 2026-06-24):

1. **Pass-B plugin-skip missing.** port_a3 ops where `verification.json.precision.pass_b.status == N/A` (the D.6 kw_brief forbids authoring `run_pass_b.py`) must skip pass_b. The SSH path had the P96 skip; the local path did not → "no Pass B verifier found" on a kernel that was never supposed to run pass_b.
2. **Canonical pass_a plugin-skip missing.** Without the `canonical_pass_a_skip_reason` check, the local path fell through to the generic benchmark-style `precision_eval_two_tier.py` (no `tier2_status`) instead of the worker's two-tier `pass_a_runner.py`.
3. **Runner-output label mismatch.** The local path labeled its normalized pass_a output `"pass_a_local"`; the tier-normalization step (`_normalize_port_a3_two_tier_pass_a`) gates on `label == "pass_a"`, so it never fired → `tier2_status` dropped → `_gate_port_a3_two_tier` tripped. (pass_b's `"pass_b_local"` label is harmless — no equivalent normalization gates on it.)

**Concrete anchor** (the mirroring checklist — run it whenever a new verifier-side skip/normalize rule lands on one path):

```
- [ ] pass_b plugin-skip (status==N/A): mirrored in BOTH _run_verifier_local AND ssh_runner?
- [ ] canonical_pass_a plugin-skip (skip_reason): mirrored in BOTH _run_canonical_pass_a_local AND _run_canonical_pass_a?
- [ ] runner-output label: BOTH paths emit the SAME label the downstream normalizer gates on?
```

**Why this matters**: the local-container path is the default on A5 dev boxes; a divergence here means a correct, fully-verified kernel stalls at finalize — burning worker / precision-probe / optimizer spawn cycles on a phantom defect. Per CLAUDE.md "Fix Harness for Next Customer, Not Patch Single Archive": a path-divergence gap blocks the NEXT port_a3 op identically, so the fix belongs in the runner, not in the single archive.

**Evidence**:
- top_k_top_p_sample A3→A5 port, kw-2 (2026-06-24, Ascend950PR CANN 9.0.0): a correct kernel (10/10 T1_PASS, bit-exact index output) was REJECTED at finalize by `phase_o5_runner_failed` ("no Pass B verifier found locally"). Root cause = the 3 local-path divergences above, NOT a worker/precision error. Fix: 3 mirrors added to `phase_o5_runner.py` (`_run_verifier_local` P96 pass_b skip + `_run_canonical_pass_a_local` `canonical_pass_a_skip_reason` + pass_a label `"pass_a"`); 5 regression tests added (`tests/test_p96_local_pass_b_skip.py`); 38 phase_o5 tests pass, 0 regressions. Evidence cross-checked against the runner diff — all 3 `# DEBT-NEW (2026-06-24, top_k_top_p_sample kw-2)` fixes present in the working tree (step 2.7 cross-check PASS).

**Other instances (predicted)**:
- Any future port_a3 op whose pass_b is legitimately N/A and runs on a local container (the exact class top_k_top_p_sample belongs to).
- Future verifier-side skip rules (e.g. a new optional tier added to the D.6 brief) — must be mirrored across both paths at introduction time, not patched per-op.
- Any migration/backward workflow that introduces a second verifier implementation — the mirroring checklist applies by analogy.

**Cross-ref**: OL-238 (finalize `phase_o5_mismatch` from a stale interpreter/env pointer — a DIFFERENT root cause; this entry is the path-divergence cause), OL-218 (emit concrete pass counts so one transient N/A can't sink a correct archive), OL-260 (the top_k_top_p_sample kernel pattern this finalize gap was blocking).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-261（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
