---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Surgical metadata-only response when finalize→await_worker rollback is structural-not-numerical"
description: "applies_to: soc=all; cann=all; bisheng=n/a; op_class=workflow verified_on: synthetic pytest fixture (test_iter_cap_p0aa_drives_fina0 — 9_topktopp underlying) unverified_on: real-op rollback in product"
phenomenon: build_failure
signal:
  - "state_transitions.jsonl shows a finalize → await_worker rollback whose rationale cites a missing Pass-B verifier artifact OR a missing GATE_CONTRACT §\"MANDATORY"
confidence: inferred
status: stub
original_id: CAND-PP92
timestamp_inferred: true
tags: [candidate, inferred, state_transitions.jsonl, rationale, verification.json, persist_verdict, precision.persist_classification, cand-pp92]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=workflow`
`verified_on: synthetic pytest fixture (test_iter_cap_p0aa_drives_fina0 — 9_topktopp underlying)`
`unverified_on: real-op rollback in production`

**Trigger**: `state_transitions.jsonl` shows a finalize → await_worker rollback whose `rationale` cites a missing Pass-B verifier artifact OR a missing GATE_CONTRACT §"MANDATORY artifacts for finalize gate" field, AND the prior pipeline (probe + researcher + optimizer) has already produced terminal evidence on `verification.json` (`persist_verdict` set, `precision.persist_classification` set, optimizer plateau reached). The rollback is a metadata-shape gap, NOT a numerical re-litigation request.

**Recommendation** — kw response on this kind of rollback is surgical, not regenerative:
1. Read GATE_CONTRACT §"MANDATORY artifacts for finalize gate" in FULL. The rollback `rationale` only names one symptom but the gate audits four fields: `run_pass_b.py` artifact + `verification.json.precision.pass_b` + `performance.independent_re_measure` + `performance.ratio_baseline`. Fixing only the symptom named in `rationale` re-triggers rollback on the next-named missing field.
2. Emit the missing artifacts AT THE WORKSPACE ROOT — do NOT re-run analysis / build / verify phases.
3. Preserve prior `precision.persist_verdict` / `precision.persist_classification` / `precision.persist_evidence` verbatim. The upstream pipeline owns those; kw must not overwrite.
4. Use `pass_b: {status: "N/A", reason: "<upstream evidence chain>"}` as canonical encoding when PARTIAL_PERSIST verdict is already established. Reason string MUST cite the upstream evidence (probe_report.md + cann_strategy_inference.md) so the finalize gate sees a non-frivolous N/A.
5. Re-emit `→ orchestrator: PARTIAL_PERSIST — <evidence>` with the same evidence chain — do NOT switch pass_a to PASS or inflate the verdict.

**Concrete anchor** (verification.json pass_b stanza encoding PARTIAL_PERSIST established upstream):
```json
"pass_b": {
  "status": "N/A",
  "reason": "PARTIAL_PERSIST established by upstream probe + researcher; see probe_report.md + cann_strategy_inference.md. Tier-1 residual on 4 fp16 cases — bit-exact edge-dataset would only re-confirm."
}
```

**Anti-patterns explicitly avoided**:
- Regenerating kernel files (burns iter budget on conclusion already reached)
- Re-running probe / researcher / optimizer (verdict=requirement already terminal per V3.8.8 "never let PARTIAL pass" + iter_cap policy)
- Promoting PARTIAL to PASS by lying about pass_b OR switching pass_a status (precision verdict integrity)
- Writing to `state_transitions.jsonl` (orchestrator-owned artifact)

**Cites** ANTI_PRESSURE_PROTOCOLS P5 (closure-pressure / "expected failure") + P7 (closure-desire after long pipeline) — the urge to over-deliver ("regenerate kernel", "re-measure perf") is exactly the pressure mode this pattern guards against. Also cites GATE_CONTRACT §"Phase D Verify Gate" + §"MANDATORY artifacts for finalize gate" (P0aaf #108, P0aba 2026-05-07).

**Promote when**: a real-op rollback in production exercises this surgical-metadata-only response and confirms (a) no re-verify needed, (b) finalize gate accepts the re-emitted PARTIAL_PERSIST, (c) iter cap preserved.

**Evidence**: synthetic pytest fixture `test_iter_cap_p0aa_drives_fina0/test_op` (2026-05-10, 9_topktopp underlying op-class transcendental + sort, DET_POLICY=best_effort). Pipeline pre-state: worker→probe→researcher→optimizer×5 with KO_PERF_PLATEAU. Rollback rationale: "no Pass B verifier found". kw spawn closed metadata gaps without re-litigating precision/perf verdicts. 1 fixture, 0 real-op evidence.

**Source**: pytest fixture iter_cap_p0aa kw spawn (2026-05-10).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP92，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
