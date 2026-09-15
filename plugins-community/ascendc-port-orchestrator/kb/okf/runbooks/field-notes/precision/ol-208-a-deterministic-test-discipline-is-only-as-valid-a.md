---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A deterministic-test discipline is only as valid as its target — the proxy itself can be refuted; keep the protocol, hold the target as falsifiable"
description: "applies_to: soc=all; cann=all; bisheng=n/a; op_class=all"
phenomenon: precision_issue
signal:
  - "applies_to: soc=all; cann=all; bisheng=n/a; op_class=all"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-208
timestamp_inferred: true
tags: [ascendc, ol-208]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0 (FA-A5 CalcTschBlockDim A/B 2026-06-05)`

**Principle**: the deterministic-test methodology (fail-before → succeed-after on a verifier-controlled build, anti-tailored) is strong **when the test target is the real defect**. But the *target itself* is a hypothesis that can be refuted — a clean deterministic NEGATIVE that refutes the **target's relevance** is as valuable as one that refutes a fix. Keep the methodology; hold the target as falsifiable.

**The protocol elements that ARE durable (keep, reuse)**:
- **fail-before / succeed-after on the SAME verifier-controlled build** (anti-tailored: the verifier applies + builds + runs; a positive must come from the verifier's own build, since only positives can be tailored — negatives can't).
- **author self-test fast-loop AND independent-verifier final-stamp are BOTH needed, not either-or** — anti-tailoring the final stamp must NOT remove the author's quick self-test loop; conflating them makes the author iterate blind + slow (owner catch 2026-06-05).
- **confound-guard**: vary exactly ONE variable; hold topology/everything-else constant (e.g. for a blockDim test, keep the same core-group count — changing both core-count and encoding muddies attribution).
- **non-degeneracy guard**: a multi-arm test needs a control arm that must stay healthy (e.g. ARM_B same-group must still sync) — else "everything broke" masquerades as "the fix worked".
- **delta-applied check ≠ verdict**: confirm the change took effect (the RIGHT artifact's md5 changed) — for a host-tiling change the kernel `.o` md5 is UNCHANGED and the optiling `.so` md5 changes; reading the wrong artifact's md5 mis-reads "delta didn't apply".

**Concrete anchor**: FA-A5 `CalcTschBlockDim` A/B (DS, 2026-06-05) — clean deterministic NEGATIVE (3/3, delta provably-applied via numBlocks=2 + optiling.so md5 change while kernel.o `f792bb24` unchanged, ARM_B same-group preserved) → refuted both the candidate AND (with the vendor-global-flags fact) the isolation TARGET itself.

**Evidence**: FA-A5 graybox 2026-06-05 (committed in `DEBT-FA-CASCADE-UNREPRO`, ROADMAP §6, commit `3f201374`). Cross-ref OL-207 (reproducibility gate that should precede the A/B), OL-209 (refutation-ledger), `feedback_faithful_port_copy_host_not_just_kernel`.

**Other instances (predicted)**: any A/B kernel/host experiment — always ask "does a clean NEGATIVE here refute the FIX, or does it refute the TARGET's relevance?".

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-208（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
