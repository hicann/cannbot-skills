---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "FA-class raw-launch template — host tiling POD → dtype/feature/D-bucket launcher selection → `<<<>>>` raw dispatch"
description: "applies_to: soc=Ascend950PR/V351; cann=9.0.0; bisheng=n/a; op_class=attention-fwd/CUBE_MIX verified_on: soc=Ascend950PR; cann=9.0.0 (FA-A5 forward no-dropout whitebox — 13 bit-exact + 2 within-T1-tol"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR/V351; cann=9.0.0; bisheng=n/a; op_class=attention-fwd/CUBE_MIX"
confidence: inferred
status: stub
original_id: CAND-FA-LAUNCH-DISPATCH-1
timestamp_inferred: true
tags: [candidate, inferred, d512, wp_fa_do_fp16_bnsd_d128_s64, usealigned64kernel, cand-fa-launch-dispatch-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR/V351; cann=9.0.0; bisheng=n/a; op_class=attention-fwd/CUBE_MIX`
`verified_on: soc=Ascend950PR; cann=9.0.0 (FA-A5 forward no-dropout whitebox — 13 bit-exact + 2 within-T1-tol of 15 kp=1 comparable cases [case43 d64, case37 d256]; wholeport base 64/64)`
`derived-from: target prior-art inspection for structure; launch recipe must be re-derived from selected arch22 contract and current arch35 public APIs; self-contained = no external #include arch35`

**Date**: 2026-06-07 (FA-A5 graybox stage-2 template-ization, Phase 1)
**Status**: CANDIDATE — the dispatch-LOGIC recipe (this entry codifies launcher-selection invariants).
Archived launcher bodies and extern declarations are advisory target prior art only. The generator must
emit a task-owned launcher from the selected arch22 contract; copying an archived body cannot satisfy
generation or final validation. Self-contained also requires no external `#include arch35`.

**local-kb-crossref**: CAND-FA-CV-7 (shape-regime variant dispatch — ORTHOGONAL: CV-7 picks the kernel
STRATEGY by shape class; THIS picks the launcher SYMBOL by dtype × D-bucket × feature-gate WITHIN the
fixed-shape FA family, after the host tiling POD is computed), CAND-FA-CARRIER-1 (the host tiling POD this
launch consumes), P-P103 §"Host-tiling LOGIC template" (the POD producer) + §"Concrete functional-block
inventory" (the launcher symbols are the block instantiations).

**Op-class**: any cube-MIX op generated via a raw-launch pybind (port_a3 whole-port style) where a single
host entry computes a tiling POD and dispatches to one of N pre-instantiated kernel-template launchers.

**Pattern** (the launch template, 3 stages):
1. **DoTiling** — compute the tiling POD from op-config (P-P103 host-tiling logic). Allocate outputs with
   `at::empty` (NOT `at::zeros`) for kernel-WRITE-ONLY outputs (attn_out / sm_max / sm_sum) — the kernel
   keeps running max/sum in UB and writes them out; a host pre-zero fires a redundant ZerosLike device op
   per output (measured 0.43-0.54× vs vendor's 1 fused op). Gate write-completeness with an env-toggled
   NaN-poison fill in test builds (any residual NaN in a COMPARED output = under-write = revert that one
   to at::zeros).
2. **Launcher selection** — a decision tree keyed on (feature-gate priority) → dtype → D-bucket: priority
   order `userMask → hasPse → hasDropOut → isFp32 → isBf16 → fp16(default)`; within each, `dBasicBlock ≤
   64 / ≤128 / ≤256 / else` selects `wp_fa_do_<dt>_bnsd[_feature]_d{64,128,256,512}`. **The launcher `d{N}`
   suffix names the D-bucket TIER, not the literal D**: d64=Aligned64, d128=Aligned128, d256=Aligned256
   (Dn path), **d512=Aligned768** (splitD path — the symbol name is `d512`, the device template tier is
   Aligned768). The s1=64 core-fill class (P-P103 host-tiling stage 3) overrides to the
   `S1TemplateType::Aligned64` launcher; in the whitebox only the **fp16 D=128** Aligned64 override is
   wired (`wp_fa_do_fp16_bnsd_d128_s64`) — bf16/other-D core-fill at s1=64 keeps the s1=128 kernel because
   no Aligned64 device variant exists for those buckets in this build (widening = add the variant +
   extern). Each launcher is an `extern "C"` symbol = one (dtype, S1-tier, D-tier, feature) instantiation.
3. **`<<<coreNum>>>` raw dispatch** — launch the selected `extern "C"` symbol with `(blockDim, stream,
   q,k,v,..., tilingPtr, workspace, outputs)`. blockDim = the host-computed core count.

**Load-bearing constraints (transferable insight)**:
- Launcher symbol MUST match the host tiling's S1-tier (Aligned64 vs Aligned128) — a mismatch silently
  computes wrong tiles. The host `useAligned64Kernel` predicate and the launcher choice are ONE decision.
- `at::empty` for kernel-write-only outputs is the perf-correct default (avoid per-output ZerosLike); prove
  write-completeness before banking (poison-gate).
- D-bucket rounds UP (`AlignUp(D,64)` then ceil to the next device tier) — intermediate D reuses the
  next-bucket template (truth-valid up to D≤768).

**Other-instances-predicted**: any port_a3 whole-port cube-MIX op with a raw-launch pybind + multiple
dtype/feature/shape template instantiations (FA+X fusions, quant-attention, MoE-with-cube-stage). The
at::empty-vs-ZerosLike rule generalizes to ANY op whose outputs are fully kernel-written.

**Known gaps**: only the buckets a given spawn wires are reachable (whitebox: fp32 wired d64-only,
user-mask wired d128-only). case37 (d256/Aligned256) + dropout (kp<1) inherited from the whitebox (P-P103).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-LAUNCH-DISPATCH-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
