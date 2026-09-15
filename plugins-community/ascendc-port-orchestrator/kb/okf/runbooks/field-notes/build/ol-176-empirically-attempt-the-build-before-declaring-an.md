---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Empirically attempt the build before declaring an L4 \"structural ceiling\" [V351/Ascend950PR, port_a3_to_a5, L0-agent-discipline + classifier-override]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-176
timestamp_inferred: true
tags: [ascendc, ol-176]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5`
`verified_on: soc=Ascend950PR; cann=9.0.0`

**Principle**: Surface metrics (LOC, Mmad call-site count, CrossCoreSetFlag count, op-class taxonomy) are **necessary-not-sufficient** evidence for routing a port_a3_to_a5 op to L4 ("structurally untranslatable"). They are over-broad classifiers — high metrics correlate with hard ports but do not entail one. Before any L4 declaration, the worker MUST attempt the minimum-viable build once.

**Minimum-viable build procedure**: when upstream A3 lacks `op_kernel/arch35/`, author a thin worker TU (`kernels.cpp` + `pybind11.cpp`) using verbatim upstream headers + `KERNEL_TASK_TYPE_MIX_AIC_1_2` dispatch (no arch35 wrapping, default-OFF prestage per CLAUDE.md port_a3 rule). Run `deploy_to_npu_lane.sh --build` once.

**Decision rule from build outcome**:
- Build PASS → the port is L1/L2 wiring, not L4. The `__CCE_AICORE__==220` guards in V220 upstream are benign on V351 (early-return doesn't fire; code follows the V351 path).
- Build FAIL with arch-specific errors after 3-5 compile-fix iterations → escalate per OL-159 (FA-class structural rewrite) / OL-156 (V220→V351 paradigm rewrite).
- Build FAIL with environment errors (resolver, container, CANN install) → fix infra, NOT escalate.

**Evidence**: `lightning_indexer_grad` 2026-05-23: build PASS in 1 spawn disproved 4 prior kw spawns + 2 researchers + 1 probe that had declared L4 based on surface metrics (2476 LOC / 9 Mmad call-sites / 8 CrossCoreSetFlag — all "high" by V220 standards). Those metrics correctly identified that the port was non-trivial but incorrectly predicted that it was unbuildable. The probe-before-fix discipline (CLAUDE.md) applies here: 1 build attempt is cheaper than 4 reroute spawns + analysis docs.

**Cross-ref**: OL-156/OL-159/OL-141/OL-175, CLAUDE.md "Probe-Before-Fix" + "port_a3_to_a5 default-OFF prestage".

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-176（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
