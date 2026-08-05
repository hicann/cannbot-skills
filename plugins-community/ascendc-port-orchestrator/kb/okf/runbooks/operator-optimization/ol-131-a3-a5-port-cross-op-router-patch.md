---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "arch22→arch35 port: patch the peer op's aclnn router for aclnn-shared v2/v3 op families"
description: "When a ported op reuses a peer's aclnn interface (aclnn_exclude), editing only the ported op's files is insufficient — the peer's op_api router must also be patched, but only when the peer's op_api actually changes in the PR."
original_id: OL-131
confidence: single_run
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-131, port_a3_to_a5, aclnn-router, v2-v3-op-family]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

When an op declares `ACLNNTYPE aclnn_exclude` in its `op_def.cpp` — meaning it does NOT
generate an independent aclnn interface and instead reuses a peer op's aclnn entry (common
for v2/v3 op families) — porting it to A5 may also require patching the **peer op's router**
in `op_api/<peer>.cpp`. Editing only the ported op's own files can be insufficient because
aclnn calls dispatch through the peer's router, and that router may not know to route to the
A5-enabled current op.

### Two-step detection (refined from the ctc_loss_v3 false-positive, 2026-05-12)

1. **Peer deps declared?** Check `op_host/CMakeLists.txt` for a `DEPENDENCIES <peer_op>` line.
   If absent → no peer-router work; stop. (Implemented in
   `phase_o25_a3_ref.derive_op_dependencies`.)
2. **Does the peer's `op_api/` actually change in this PR?** This distinguishes a *router-edit
   peer dep* from a *build-system-only peer dep*:
   ```bash
   cd ~/workspace/cann/ops-nn
   git diff master..FETCH_HEAD -- <cat>/<peer_op>/op_api/
   ```
   - **Diff non-empty** → the peer's aclnn router routes to our new A5 kernel and needs the
     3-edit patch below.
   - **Diff empty (build-system-only peer dep)** → the peer is linked via CMakeLists but its
     `op_api/<peer>.cpp` is untouched; the new kernel dispatches at `op_kernel` level via its
     own `<op>_apt.cpp` entry + `regbaseCfg` `opFile.value=<op>_apt`. **No router patch needed.**

Step-1-only detection over-reported router work: ctc_loss_v3 (PR4778) declares
`DEPENDENCIES ctc_loss_v2` but `git diff master..FETCH_HEAD -- loss/ctc_loss_v2/op_api/` is
empty — the PR touches only ctc_loss_v2's `op_host/CMakeLists.txt` + `op_graph/CMakeLists.txt`.
The dispatcher is reused as-is; the new kernel registers via its own apt.cpp entry. Going
forward, emit two flags: `peer_deps_declared` AND `peer_router_edit_required`.

### Three router-edit primitives (apply all three to the peer's `op_api/<peer>.cpp`)

1. **Branch redirect** — find the dispatcher function (typically `<PeerOp>()`), locate the
   `IsregBaseAiCoreSupport(...)` branch, and change its body from `return <PeerOp>AiCore(...)`
   (routes to the peer kernel) to `return <CurrentOp>AiCore(...)` (routes to OUR new A5 kernel).
   Example anchor from the ctc_loss_v3 plan:
   ```cpp
   // Before:
   if (IsregBaseAiCoreSupport(logProbs) && !targets->IsEmpty()) {
       return CtcLossAiCore(...);   // → CTCLossV2 kernel (wrong for A5)
   }
   // After:
   if (IsregBaseAiCoreSupport(logProbs) && !targets->IsEmpty()) {
       return CtcLossV3AiCore(...); // → CTCLossV3 kernel (correct)
   }
   ```
2. **Support-gate extend** — widen the peer's support gate so the A5 kernel's cases pass.
3. **Alignment unify** — unify alignment handling between peer and current op.

Source: W9 (2026-05-12, ROADMAP §1.5), extracted from
`loss/ctc_loss_v3/docs/ctc_loss_v3_a5_migration_plan.md` (gitcode `cann/ops-nn` PR #4778,
§4.2-4.4). `applies_to: paradigm=ascendc`.

Note: primitives #2 (support-gate extend) and #3 (alignment unify) are summarized to their
names here; the arch22→arch35 migration plan (§4.2-4.4) carries their full code — consult it when applying.
