---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "grep the op's own call-graph before attributing a defect to a subsystem; NO_FIRE ≠ clean; don't revive a grep-refuted root by assertion"
description: "applies_to: soc=all; cann=all; bisheng=n/a; op_class=all"
phenomenon: precision_issue
signal:
  - "applies_to: soc=all; cann=all; bisheng=n/a; op_class=all"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-209
timestamp_inferred: true
tags: [ascendc, ol-209]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0 (FA-A5 graybox refutation sequence 2026-06-04/05)`

Three tightly-related refutation-discipline rules:
- **grep-the-call-graph before attributing a root to a subsystem**: a disasm/binding-screen can be internally valid yet pin a fault to a subsystem the op never invokes. The cheap decisive check is grepping the op's OWN source for the subsystem's entry symbols. (FA: `grep KfcServer/KfcCommClient/InitSocState op_kernel/` = 0, FA uses hand-`Mmad` → the KFC/DEBT-20 root was a red-herring; FA never instantiates matmul-lib-KFC.)
- **NO_FIRE = inconclusive ≠ clean** (inverse-false-clean): for a rare/non-deterministic fault, a run (even ×N, even ×32000) that does NOT reproduce the fault proves nothing — and instrumenting "the real kernel" only helps if the fault actually fires (if it can't be made to fire, instrumentation captures nothing either).
- **refutation-ledger is authoritative; do NOT revive a refuted root by assertion**: once a hypothesis is grep/deterministically refuted, it stays refuted unless NEW evidence overturns the refutation. A fatigued agent re-citing a refuted root from a stale ledger entry is a flip-flop — catch it by re-grounding against the latest verdict (FA: the DEBT-20/KFC root was revived by-assertion mid-session and re-retracted after a fresh self-grep).

**Evidence**: FA-A5 graybox 2026-06-04/05 refutation sequence (KFC-bind-gap → `%11`-per-core-wrap → flag-isolation → `CalcTschBlockDim`, all refuted), committed as the `DEBT-FA-CASCADE-UNREPRO` refutation-ledger (ROADMAP §6, commit `3f201374`). Cross-ref OL-207 (the reproducibility gate), OL-208 (deterministic-test-target discipline), `feedback_faithful_port_copy_host_not_just_kernel`.

**Other instances (predicted)**: any multi-hypothesis root-cause hunt; any non-deterministic race investigation where a NO_FIRE run risks being mis-read as a clean bill of health.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-209（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
