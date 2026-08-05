# KB Usage Log

Start date: 2026-06-26. Records KB_USAGE_LOG entries per aog-knowledge-maintain SKILL.md (Mode 1 P0ff).

---

## 2026-07-06: BASELINE_LOADED codex_e2e_add source E2E rerun

BASELINE_LOADED:
- `src/skills/ascendc-op-gen/SKILL.md`
- `src/skills/references/shared/ANTI_PRESSURE_PROTOCOLS.md`
- `src/skills/references/KB_INDEX.md`
- `/home/npu_user/.codex/skills/aog-clone-reference-source/SKILL.md`
- `CLAUDE.md`
- `../AGENTS.md`

Context: rerun of `/ascendc-op-gen workspace/e2e_sources/codex_e2e_add.cpp --fast --perf-threshold 0` from Codex. `bash src/scripts/check_submodule_freshness.sh` reported all submodules fresh. This entry records the required startup load only; no KB promotion decision was made.

## P0ff heuristic-mode candidate parked (masked_select_v3 kb_draft_from_user_decision.md, 2026-06-26)

The `kb_draft_from_user_decision.md` for op `masked_select_v3` (extracted by orchestrator at 2026-06-26T01:09:53Z) is in **heuristic mode** (`mode=heuristic` — no structured `kb_distillation:` block in original user_decision.md).

Per Mode 1 P0ff rules: heuristic-mode candidates are NOT auto-promoted. The full kb_draft content is verbatim at `workspace/masked_select_v3/kb_draft_from_user_decision.md`. Key content: user directive about proper TQue discipline (QDEPTH=1, sequential use, persistent data in TBuf not TQue) from the group_norm_silu counter-example. This needs the owner to re-distill as a structured `kb_distillation:` block before auto-promotion.

**Next action**: DONE — owner promoted to OL-262.

## 2026-06-26: Promoted to OL-262

Owner reviewed the parked candidate and promoted it to a structured OL entry:
- **OL-262**: pybind path CAN use multiple MTE2 TQue — agent must not blame pybind structure. Core fact: group_norm_silu already proves pybind build path supports 6 TQue. masked_select_v3 worker wrongly attributed two-VECIN-TQue failure to pybind; real cause may have been NPU0 hardware state.
- Full entry: `OPERATIONAL_KNOWLEDGE.md#OL-262`

## P0ff heuristic-mode candidate parked (forward_spec_grad kb_draft_from_user_decision.md, 2026-06-30)

The `kb_draft_from_user_decision.md` for op `forward_spec_grad` (== selective_scan_full_grad; extracted by orchestrator P0ff at 2026-06-30T05:33:16Z) is in **heuristic mode** (`mode=heuristic` — no structured `kb_distillation:` block in the originating user_decision.md).

Per Mode 1 P0ff rules: heuristic-mode candidates are NOT auto-promoted. Full kb_draft verbatim at `workspace/forward_spec_grad/kb_draft_from_user_decision.md`. Content is an **infra/lane-provisioning directive, not a generalizable KB rule**: the `INFRA_BASELINE_VIOLATED` escalation was the known lane-provisioning gap — orch ran `--lane 0` whose `BENCHMARK_ROOT` (AscendOpGenAgent_lane0) is task-only (no `utils/build_ascendc.py`); only lane6 is build-provisioned (verified on-disk; `A5_DEFAULT_NPU_ID=6`). NOT a kernel defect — kw correctly escalated per P9. Resolution recorded in directive: resume build+verify on the build-provisioned lane 6.

**Assessment**: this is operational lane-provisioning state, not a cross-op transferable lesson. If a generalizable rule is wanted ("op-gen lanes must be build-provisioned, or escalate INFRA_BASELINE_VIOLATED rather than blame the kernel"), the owner should re-distill as a structured `kb_distillation:` block. **Next action**: parked for owner re-distillation; no auto-promote. The substantive precision finding from this op was promoted separately as `CAND-BWD-RATIO-DEGENERATE-ZERO` (from knowledge_update.md, not this kb_draft).

## P0ff heuristic-mode candidate parked (selective_scan_full_grad kb_draft_from_user_decision.md CH512 landing, 2026-07-15)

The `kb_draft_from_user_decision.md` for op `selective_scan_full_grad` (extracted by orchestrator P0ff at 2026-07-15T04:04:44Z) is in **heuristic mode** (`mode=heuristic` — no structured `kb_distillation:` block in the originating user_decision.md).

Per Mode 1 P0ff rules: heuristic-mode candidates are NOT auto-promoted. Full kb_draft verbatim at `workspace/selective_scan_full_grad/kb_draft_from_user_decision.md`. Content is an **execution directive, not a generalizable KB rule**: chair (agent-main) + owner-authorized (B) landing of the CH256→CH512 UB-layout rewrite on the reverse SIMD kernel, with a chair-gated precision hard-gate (fp32 no regression / half-prec 9/48 floor no worse / any per-grad margin regression esp grad_delta_bias 1.67× → rollback / no tolerance loosening / preserve PASS-A→PASS-B WAR fence) and a same-card A5 NPU lane back-to-back npu.Event A/B protocol (NOT msprof — DEBT-149 + that A5 host forbids msprof).

**Assessment**: this is a per-op execution directive + gate spec, not a cross-op transferable lesson. The **generalizable substance** of the CH512 experiment (the outcome the directive authorized) WAS distilled from the worker's `knowledge_update.md` (kw-3) into a real candidate this run: **`CAND-CHUNK-SIZE-RAISE-CHUNKED-FP32-REDUCTION-NOT-FREE`** (candidates.md), plus evidence appended to `CAND-SCAN-FP32-ACCUM` and `OL-231` (KO-6b chunked-scan prediction CONFIRMED). **Next action**: parked; no auto-promote of the directive itself. If a generalizable gate-design rule is wanted (e.g. "chunk-size changes on chunked fp32-reduction kernels are gated on fp64-oracle floor ratios, never byte-diff"), the owner should re-distill as a structured `kb_distillation:` block — though that principle is already captured in the CAND above.
