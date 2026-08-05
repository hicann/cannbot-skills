---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Phase-O5 re-measure stages the flat `precision_eval_*.py` grader scripts but NOT their package import closure into the DEBT-159 lane-isolated `current_task` → `pass_a: claimed but not measured` (`measured.pass_a=None`) → perpetual `phase_o5_mismatch` finalize rollback on a kernel that is actually PASS"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all; mode=port_a3_to_a5; scope=lane-isolated"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all; mode=port_a3_to_a5; scope=lane-isolated"
confidence: single_run
original_id: PB-52
timestamp_inferred: true
tags: [current_task, phase_o5_mismatch, benchmark_root, ascendc, pb-52]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all; mode=port_a3_to_a5; scope=lane-isolated`
`verified_on: soc=Ascend950PR; cann=9.0.0`

When `BENCHMARK_ROOT` is the shared `/root/AscendOpGenAgent` default, DEBT-159 isolates lane N to `/home/npu_user/workspace/AscendOpGenAgent_lane{N}/current_task`, and `phase_o5_runner._lane_aware_benchmark_root(env, lane=N)` makes the Phase-O5 re-measure READ from that lane-aware path — NOT `/root/...`. But `phase_o5_runner._resync_workspace_to_container` only stages the two FLAT grader scripts (`precision_eval_two_tier.py` + `precision_eval_port_a3_two_tier.py`); it does **not** stage their package import closure (`cannbench_grader/`, `reference_provider/`), which lives only under `/root/...` on this fleet. In the lane-isolated dir those packages are absent → the canonical grader (`from precision_eval_two_tier import classify_output` → `from cannbench_grader import ...`) AND the fallback (`pass_a_runner.py`) both raise `ModuleNotFoundError` at import. The canonical `docker_cmd` suppresses stderr (`>/dev/null 2>&1`) and returns rc=0 from a trailing `rm -f`, so the crash surfaces only as `_run_canonical_pass_a → "no JSON in stdout"` → `measured.pass_a=None`.

- **Failure signature (recognise this class fast)**: `verification.json` claims a real `pass_a`, the kernel precision is independently reproducible as PASS, yet O5 reports **`pass_a: claimed but not measured`** with `measured` containing only `pass_b`. This is NOT a count discrepancy (it can be misread as a "1 discrepancy" count mismatch) — it is an unmeasurable-because-import-crashed condition. Diagnostic: check whether the grader import closure exists at the lane-aware `AscendOpGenAgent_lane{N}/current_task`, not just `/root/...`.
- **Detection trap**: injecting the grader deps into `/root/AscendOpGenAgent/current_task` (and verifying 87/87 there by hand) does NOT clear it — the O5-read path is the lane-aware one. Two prior spawns applied exactly this ineffective fix and the signature repeated. Manual SSH from a shell can hit `/root/...` while the runner hits the lane-aware path; reproduce through the orchestrator's own runner (see [[CAND-O5-REPRO-VIA-RUNNER-NOT-MANUAL-SSH]] in `patterns/unverified/candidates.md`).
- **Permanent fix (owner/harness)**: add `cannbench_grader/` + `reference_provider/` to the `_resync` extra-payload staged into the lane-aware `current_task`, OR vendor the import closure into the pushed `precision_eval_*.py`.
- **kw-scope stopgap** (durable across `--skip-old-files` resyncs): stage the two package dirs FLAT into the lane-aware `current_task` (`.../AscendOpGenAgent_lane{N}/current_task/{cannbench_grader,reference_provider}/`) so the grader script-dir `sys.path[0]` resolves them.
- **Status**: OPEN (harness resync payload gap; fix is owner/harness-side).
- **Evidence**: gelu port_a3_to_a5 kw spawn 3 (2026-07-05, A5 Ascend950PR_957b, CANN 9.0.0, lane 0): repeating `phase_o5_mismatch` rollback (2nd attempt, same signature) root-caused to `measured.pass_a=None` from `ModuleNotFoundError` at the lane-0 grader import; container build independently 87/87 PASS. Local workspace destroyed mid-session by a concurrent external experiment (see PROGRESS.md Part 3) — deliverables unrecoverable, so the fix is documented here rather than claimed as a landed PASS.
- **Cross-reference**: OL-272 #2 (sibling — the backward-mode Phase-O5 resync `push_files` gap for the worker's OWN verify closure; same "resync rsyncs only `kernel/**` + flat top-level files, not the verify-side import closure" root cause, different verify path), DEBT-159 (lane-0 isolation), P0kk / phase_o5 O5 re-measure gate (claim-vs-measured cross-check), P0abh / task#82 (canonical port_a3 two-tier grader authority — author ≠ measurer). backend=ascendc.

<!-- 迁移自 porter kb/target/ascendc/（PB-52，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
