---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Phase O5 re-measurement disagrees with worker `pass_a_runner.py` `n_pass` count → orchestrator infinite finalize→await_worker loop when worker iter_cap exhausted [V220+V351, port_a3_to_a5, orchestrator-FSM]"
description: "applies_to: soc=all; cann=all; bisheng=n/a; op_class=any-with-non-deterministic-quant-or-stale-binary"
phenomenon: build_failure
signal:
  - ".rollback_history.jsonl accumulates many identical phase_o5_mismatch entries (signature phase_o5_mismatch::await_worker) within minutes. state_transitions.jsonl"
confidence: single_run
original_id: EC-59
timestamp_inferred: true
tags: [n_pass, phase_o5_mismatch, ascendc, ec-59]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=any-with-non-deterministic-quant-or-stale-binary`
`verified_on: flat_quant 2026-05-23 (kw-9 worker: tier1_pass=4/8; Phase O5 re-run: tier1_pass=0/8; 84 phase_o5_mismatch rollbacks in 37min)`

- **Symptom**: `.rollback_history.jsonl` accumulates many identical `phase_o5_mismatch` entries (signature `phase_o5_mismatch::await_worker`) within minutes. `state_transitions.jsonl` shows ping-pong `await_worker → finalize → await_worker → finalize → ...` with iter_counts.worker climbing past cap (e.g. 84/9 → 91/9). Orchestrator never terminates.
- **Root cause**: Two compounding bugs:
  1. **Worker `pass_a_runner.py` `n_pass` is not reproducible across runs**: the runner counts a case bit-exact only when BOTH `ok_out` AND `ok_qscale` are true with strict equality. For ops with marginal precision (qscale floor near bf16 ULP, non-deterministic quantization order, or stale-binary re-deploy), worker's first-run count and O5's re-measurement count can disagree (e.g., 4/8 vs 0/8) without the kernel being wrong — just non-deterministic at the bit-exact threshold.
  2. **Orchestrator finalize→O5 MISMATCH branch unconditionally routed to await_worker** even when worker iter_cap was already exhausted. Combined with the P0y "legitimate exhaustion → finalize" route at orchestrator.py:1349, this created an infinite loop: finalize → O5 MISMATCH → await_worker (cap exhausted) → finalize again. No state change between iters → loop never escapes.
- **Fix**:
  1. **FSM loop-guard (orchestrator.py P0bb-loop-guard, 2026-05-23)**: in both `O5 MISMATCH` and `O5 RUNNER_FAILED` branches, check `state_executor.at_iter_cap(workspace, "await_worker")` before routing back. If exhausted, log a FATAL diagnostic naming the four hypotheses (stale binary / different inputs / non-determinism / fabrication) and `return 2` instead of looping.
  2. **Dual-count schema (P0cc, 2026-05-23 — closes schema gap, not just the loop)**: `pass_a_runner.py` MUST emit BOTH `tier1_pass` (strict bit-exact count) AND `tier1_pass_inclusive` (T1+T2-within-tolerance count). `phase_o5.py` honors `tier1_pass_inclusive` when worker-declared `precision.pass_a.status` is in `("PASS_WITHIN_TOLERANCE", "PARTIAL_PASS_WITHIN_TOLERANCE", "PARTIAL_PASS")`. `kw_brief.py` port_a3 phase block instructs worker to emit both counts. `verification.json` MUST include `tier1_pass_inclusive` field for these statuses (assertion in pre-done checklist). Why dual-count (not just loop-guard): the loop-guard makes the orchestrator FAIL cleanly instead of loop forever, but it doesn't fix the underlying schema gap — next customer running a similar marginal-tolerance op still hits MISMATCH → FATAL (clean terminal but no archive). Dual-count lets the FSM correctly promote when worker's T2 verdicts are within tolerance AND O5's inclusive re-measurement agrees. Per `feedback_no_patch_fix_harness_for_next_customer.md`: the harness is the product, per-archive intervention is a patch.
- **Detection** (without watching the orchestrator):
  - `wc -l workspace/<op>/.rollback_history.jsonl` > 20 → loop suspected
  - `grep -c phase_o5_mismatch workspace/<op>/.rollback_history.jsonl` > 5 → confirmed loop
  - `tail .opgen_state.json` shows `invocation_count` growing without commit-on-disk advancing
- **Evidence**:
  - flat_quant 2026-05-23: worker kw-9 emitted real per-case verdicts (4 T1_BIT_EXACT + 4 T2_PASS_WITHIN_TOLERANCE deterministic), pass_a_results.json showed `n_pass=4`. Phase O5 ran the SAME pass_a_runner.py via SSH and got `n_pass=0`. Loop ran 84 times (3:40→4:17Z UTC) before user intervention.
- **Cross-ref**: P0kk (Phase O5 post-verify, orchestrator.py:994-1037), P0y (legitimate pipeline exhaustion → finalize, orchestrator.py:1349), P0bb-loop-guard (this entry's fix landed 2026-05-23).

<!-- 迁移自 porter kb/target/ascendc/（EC-59，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
