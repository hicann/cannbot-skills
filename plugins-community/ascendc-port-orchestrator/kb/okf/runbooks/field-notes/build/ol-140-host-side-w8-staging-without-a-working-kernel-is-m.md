---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Host-side W8 staging without a working kernel is misleading scaffolding [V351, port_a3_to_a5]"
description: "Source: fused_quant_mat_mul port spawn (kw-1 resume, 2026-05-13) — self-critic W5 flagged a temptation to stage def.cpp + binary.json + simplified_key.ini + apt.cpp as \"compromise productive work\" whe"
phenomenon: build_failure
signal:
  - "Source: fused_quant_mat_mul port spawn (kw-1 resume, 2026-05-13) — self-critic W5 flagged a temptation to stage def.cpp + binary.json + simplified_key.ini + apt"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-140
timestamp_inferred: true
tags: [fused_quant_mat_mul, ascendc, ol-140]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

> **Source**: `fused_quant_mat_mul` port spawn (kw-1 resume, 2026-05-13) — self-critic W5 flagged a temptation to stage def.cpp + binary.json + simplified_key.ini + apt.cpp as "compromise productive work" when the kernel-side dispatch was structurally blocked. Declining that path was correct; codifying the rule prevents future regressions.

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5`
`verified_on: soc=Ascend950PR (fused_quant_mat_mul kw-1 resume, declined the path)`
`unverified_on: backward modes — these have no "W8" artifact concept; the anti-pattern is port-mode-specific`

### Principle

When the kernel-side dispatch (arch35 epilogue branch, activation lowering, sub-variant kernel) is structurally blocked, staging the host-side W8 canonical artifacts (def.cpp regbase block, binary.json, simplified_key.ini, apt.cpp) is an anti-pattern. The staged files SUGGEST forward motion to anyone reading the workspace afterwards — orchestrator, finalize pipeline, ROADMAP reviewer — but actually unblock nothing: the kernel either does not compile (no Process() symbol), or compiles to a no-op that emits zero output for every benchmark case.

This is a C25 (premature stop after root cause) + C18 (cheating-by-claim) + P5 (reward-hacking) + P7 (closure desire) combined failure mode. The scaffolding LOOKS productive and is mechanical (1-2h), which makes it psychologically attractive under spawn budget pressure. The user-acknowledged routing decision must come FIRST; W8 staging is a downstream consequence of having a working kernel target, not a path-of-least-resistance fallback.

### Concrete anchor (decision rule for the kw / orchestrator)

```
Before staging any W8 artifact in a port-mode workspace, verify:

  1. Does op_kernel/arch35/<op>.{cpp,h} contain a dispatch branch for
     EVERY variant the benchmark requires? (per OL-139 enumeration)
       NO  → STOP. Do not stage W8. Handoff await_user_decision.
       YES → continue to (2).

  2. Does the dispatched code compile against arch35 / __CCE_AICORE__ != 220
     headers (no V220-only EpilogueTypeTraits, no impl/dav_c220/* includes)?
       NO  → STOP. Do not stage W8. The "compile" half of the port is
              not yet defined; staging registration files for a kernel
              that fails to compile is pure scaffolding.
       YES → W8 staging is the actual remaining work; proceed.

  3. Are the 3-4 peer router patches (per OL-131) identified and
     either resolved (verify uses PyTorch front-end → defer per
     CAND-A3A5-14) or staged in the same spawn?
       NO  → flag in PROGRESS.md; W8 may still proceed but record
              the deferred peer-patch list.
       YES → proceed.
```

### Anti-pattern to avoid

- Staging def.cpp + binary.json + simplified_key.ini + apt.cpp because the kw spawn budget feels "wasted" if no files are written. The correct exit is `await_user_decision` with a clear honest description of what work remains. A workspace with W8 staged + zero-running kernel is WORSE than an empty workspace with a clear handoff note — the former looks 70% done when it is in fact 15% done.
- Describing such staging in PROGRESS.md as "infrastructure complete pending kernel author" — the verb "complete" attached to host-only artifacts misleads the reader about residual effort.

### Evidence

- `fused_quant_mat_mul` (2026-05-13, kw-1 resume): brief permitted `await_user_decision` as soft-judgment exit; W5 self-critic flagged W8 staging as the tempting C25 path. Declined; handoff was `→ orchestrator: await_user_decision` with explicit 600-1000 LOC arch35 epilogue authoring as the bottleneck.

### Other instances (predicted)

- Any A3→A5 port where the arch35 directory exists but lacks dispatch for benchmark-required variants (per OL-139)
- Any port-mode workspace where Phase A analysis surfaces a structural gap larger than single-spawn budget — the same temptation to "do something mechanical to look productive" applies
- Hybrid fused-op ports where ONE sub-op of the fusion is V351-ready and the others are V220-only — staging registration for the partial set creates the same misleading state

### Related

- OL-139 (arch35 dispatch-branch enumeration — the positive procedural rule this anti-pattern complements)
- OL-134 (port-complexity estimator — gives the budget signal that surfaces when W8 staging is premature)
- ANTI_PRESSURE_PROTOCOLS.md P3/P5/P7 (closure-desire and reward-hacking patterns that make this anti-pattern attractive under load)
- Self-critic C18 / C25 (cheating-by-claim + premature stop after root cause — the underlying drift modes)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-140（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
