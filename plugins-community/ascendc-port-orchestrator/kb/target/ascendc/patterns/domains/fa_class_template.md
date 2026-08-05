# FlashAttention-class (FA) Template Domain — A5 reproducible-framework KB

> ## ⚠ ARCH SCOPE — this template is `arch35` / A5 **ONLY**
> `arch_scope: arch35 · soc=Ascend950PR (Ascend950PR_9579) · product=Atlas A5` — the prominent, greppable restatement of the `applies_to: soc=Ascend950PR` line below. Template selection **MUST** be gated on the target SoC: this is the DEBT-208 principle ("honor each entry's own `applies_to`") applied to domain templates. The structural gap — arch-scope living as PROSE, not a machine-checkable field a composer/audit enforces — is tracked as **DEBT-222**.
> **NOT for an `arch22` / `220x` (Ascend910_9382, Atlas A3) FA target.** The `arch35/…` block bodies this template stitches are A5 device code, and the MIX cross-core recipe here (§4 handshake, PB-35, OL-220 library cube) is verified on Ascend950PR — never on Ascend910_9382. Handing this A5 template to a 220x/A3 worker mis-scopes it, exactly the composer over-block DEBT-208 fixes, one layer up.
> **There is NO 220x / A3 FA-class template yet (KB gap — DEBT-222).** For a 220x/A3 FA target the honest state is: hand-rolled AscendC cube+vec MIX on 220x has **no codified recipe** (`CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP`; DEBT-208 surfaces the *A5* recipe, but the 220x recipe was never authored). Do NOT silently reuse this A5 template as if arch-neutral, and do NOT retreat to a pure-VEC fallback (the OL-188 cheat the a3 FA brief used to teach — removed in #170). Flag the gap instead.
>
> **P-P103** body (domain file). Index rows: `PATTERN_INDEX.md` + `KB_INDEX.md` (orphan-free invariant).
> `applies_to: soc=Ascend950PR/V351; cann=9.0.0; op_class=attention-fwd / CUBE_MIX (flash-attention family)`
>
> This domain file is the **interface + recipe + knowledge layer**: generalized structure +
> public AscendC API surface + recipe + meta-lessons + advisory template stitch points, cross-linking
> the already-codified components by ID. The retained arch35 bodies are provenance-bearing target prior
> art: inspect them for structure, API use, and test hypotheses, but do not copy their bodies into a
> generated deliverable. The worker must emit task-owned code from the selected arch22 contract and
> current arch35 public APIs, then pass source-NPU truth validation. Self-contained also means NO external
> `#include "arch35/..."`. No target archive or installed implementation is a generation-success oracle.
>
> Purpose: the worked FA-class template a CUBE_MIX / attention-fwd worker is pointed at, so
> the next FA-class A5 op (and the harness-reproduction graybox) starts from knowledge, not
> from scratch. Cross-links: **P-P101** (online-softmax), **P-P102** (cube-MIX),
> **CAND-FA-COREDIST-1** (core distribution), **CAND-FA-CARRIER-1** (scalar carrier blit +
> host-tiling structure + shared-header rule), **cv_reference_concrete_params.md** (cube
> executable params), **`fa_class/cross_core_sync.md`** (the cube↔vec MIX_AIC_1_2 cross-core
> handshake — **load-bearing + the current repro frontier**. State as of 2026-06-04: a §1-§3
> hand-roll still DEADLOCKS on A5/V351 (kw-gb2 2026-06-03, PB-35 confirmed_on), but the §4
> 4-specifics handshake (SYNC MODE 4 + per-sub-block disjoint flag-ids `id`/`id+16` +
> direction-pinned literal pipes + ring-depth ≥ skew) is **PUBLIC-API-runnable** — the deadlock
> is no longer the open issue. The frontier has MOVED PAST deadlock to a narrower **non-deterministic
> high-core cross-core sync-execution fault** (507015 / CCU-263 cascade): the kernel RUNS and produces
> results but faults non-deterministically at high core counts. **Device-instrument verdict (2026-06-04,
> measured + control):** the `%11` flag-id pool-collision hypothesis (`regbase_buffer.h
> MAKE_ID=(++idCounterNum)%11` reuse exhausting the 11-id pool) is **empirically REFUTED** — measured
> max concurrent live ids = 8 (never >11, pool always ≥3 headroom), and the raw "collision" count is a
> Set/Wait-mark-asymmetry artifact that is IDENTICAL on no-fault runs (control disproves causality).
> **So there is NO kernel-id-level fix — do not attempt to "fix the `%11` lifecycle", there is no
> collision to fix.** The fault DOMAIN is confirmed as cross-core **sync-execution / MTE** (deterministic
> fault PC, sync-unit + MTE dominated, vec/cube/L1 compute-error = 0, KFC flag-15 pre+post markers both
> present so the rendezvous completes). The fault CAUSE is **OPEN, leading-hypothesis-by-elimination =
> LAUNCH MODEL** (our custom `<<<coreNum>>>` MIX_AIC_1_2 launch vs production KFC-framework launch that
> configures the FFTS+/CCU cross-core-sync subsystem) — **NOT positively proven** (the completed flag-15
> handshake rules out the simplest launch-hang story; an aog-researcher scope is bounding whether the
> fix is a small launch-side flag/API config or a large KFC-framework port). The graybox kernel itself
> is **byte-faithful to production arch35** (md5 f792bb24); what is missing is the production launch
> integration, not kernel code. Read PB-35 + CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP + cross_core_sync.md §4
> BEFORE attempting the sync; treat high-core determinism as not-yet-guaranteed and do NOT assume the
> launch-config cause is proven,
> **OL-190** (WorkspaceQueue ring-buffer sync), **OL-200** (pipeline stagger), **OL-202**
> (optional-input gating). Synthesis source: `docs/status/FA_A5_CANN_DESIGN_UNDERSTANDING.md` §J (2026-06-01).

## Purpose — template, not a one-time copy

The FA-class template KB exists so the NEXT FA-class op (an `FA+X` fusion, a decode
variant, a quantized attention) comes from a **reproducible framework**, not from a
one-time CANN-source copy. The vendor arch35 FA is itself a CRTP template — a kernel
base templated on `<CubeBlockType, VecBlockType>` (the child kernel injected via CRTP):
a ~90%-reuse skeleton plus pluggable cube/vec block-types. This domain file **mirrors
that intrinsic structure** — it does not force-generalize a flat recipe over it.

- **applies_to**: `soc=Ascend950PR/V351; cann=9.0.0; op_class=attention (FlashAttention
  forward + FA+X fusions); arch=arch35-class CRTP`.
- **product framing**: the benchmark op is `npu_fusion_attention` (vendor
  `3_FusionAttention`, 50 cases). The template is a reproducible *guide*
  (architecture + public facts + method), NOT a verbatim kernel.

## The benchmark contract + the firm ceiling

- Op axes (the full contract, not a pilot subset): **4 layouts** (BSH / SBH / BSND /
  BNSD), fp16×25 + bf16×25, sparse_mode 0–5, keep_prob ∈ {1.0, 0.9, 0.8}, head-dim
  D from 8 up to 1280.
- **Ceiling = 48/50, empirically firm.** The 2 unreachable cases are **D=1280 + D=1024**:
  `npu_fusion_attention` itself rejects them — the op_host hard-rejects D>768 (the
  public truth-limit `HEAD_DIM_MAX <= 768`; the aclnn FA path returns an explicit
  "Head dim must <= 768" error). This is the OP's limit, not ours. All other 48 cases
  (including non-template D=320/384/512/640) truth-RUN-valid → no hidden sub-768 limit.
- **Meta-lesson (empirical-ceiling discipline)**: run the truth to find the ceiling;
  do NOT assume a case is "impossible" without running the reference. The 48/50 number
  is grounded in the op erroring on D>768, triple-confirmed across independent measurers.

## The reusable skeleton (~90% reuse)

The skeleton mirrors arch35's CRTP kernel base, templated on `<CubeBlockType,
VecBlockType>`. Six components; a new FA-class op reuses all six and
swaps only the block-type deltas (see FA+X below).

1. **Carriers** — a host-filled per-iteration carrier struct (per-task/per-block runtime
   state: loop indices, ring-slot indices, per-head offsets) + an invariant carrier
   struct (compile-time / per-launch constants: shapes, tiling params, layout). These are
   the shared state structs the cube and vec block-types both read; keep them in shared
   common headers, not duplicated per block-type.

2. **Staggered AIC/AIV Process pipeline (4-deep)** — the perf-load-bearing structure
   (**OL-200**). Outer loop over output-blocks, inner kv-loop over kv-tiles, a software
   pipeline staggered by a small modulus on the task index so the cube (AIC) and the
   vector (AIV) cores are both kept busy: while vec consumes block *i*, cube produces
   *i+1*. The per-stage order within one kv-tile iteration is
   **BMM1 (QK^T) → release-S → softmax → release-P → BMM2 (PV) → release-O → ProcessVec2**.
   A serialized (lockstep) cube↔vec schedule is precision-correct but leaves each core
   ~half idle — **OL-200** is exactly this: correct ≠ perf-normal. Reproduce the
   staggered schedule, not a specific tile size.

3. **Cube block-type (manual Mmad matmul)** — QK^T (BMM1) and P@V (BMM2) via manual
   `AscendC::Mmad` (NOT `matmul::Matmul<>`), per **P-P102**. This is the cube-MIX
   canonical: manual Mmad cube + vector epilogue + cross-core sync. BMM1 dispatch keys
   off head-dim buckets (a non-split inline path for D≤bucket, an L0-split path for
   larger D); BMM2 likewise has an inline path (small D, non-fp32) and an L1-split-N
   path (fp32 inputs or large D). The cube→UB Fixpipe writes S; the consumer is the vec
   block-type. See P-P102 body (`patterns/domains/cube_vector_fusion.md`) for the
   Mmad/cross-core-sync contract. **EXECUTABLE params (the contract→code level) are in
   [`fa_class/cv_reference_concrete_params.md`](../../fa_class/cv_reference_concrete_params.md)
   (repo path `src/skills/references/target/ascendc/fa_class/cv_reference_concrete_params.md`):
   §`matmul_primitive` (the `Mmad` signature + full `MmadParams` fields, `cmatrixInitVal=(ki==0)`
   for the K-tile accumulate, and the Fixpipe `srcStride` PER-CONTEXT units — `/C0` for
   L0C→workspace vs ELEMENTS for L0C→GM); §`block_tile_alignment` (cube dims MUST be 16/C0-aligned
   — `mActAlign=AlignUp(BLOCK_M,C0)`, never raw seq); §`qk_load_form` (QK-load `ifTranspose` form +
   L0A/L0B Nz-fractal 16×C0). P-P102 gives the CONTRACT; that file gives the code-level params —
   you need BOTH to write the cube. Read it before writing Phase 2.**

4. **Vec block-type (online-softmax)** — the softmax epilogue.
   **BASE / correctness form (use this FIRST — it compiles + runs on arch35-AIV):
   the vendor `AscendC::SoftmaxFlashV2` API.** The wholeport's runnable 64/64 base uses
   it (P-P102 "vec epilogue = `SoftmaxFlashV2`"). **Do NOT hand-roll the online-softmax
   for the first runnable kernel** — a raw `WholeReduceMax`/`Exp`/`Sub`/`WholeReduceSum`
   sequence emits an internal `PipeBarrier<PIPE_S>` that **arch35-AIV REJECTS at compile**
   (`kernel_event.h "range of 1st parameter must be [4,6]"`, EC-15 class, no backtrace —
   kw-gb3 2026-06-03 hit exactly this). On arch35 the intra-vec S↔V ordering must use
   `SetFlag/WaitFlag<HardEvent::S_V>` (or rely on `SoftmaxFlashV2`'s own internal sync),
   NEVER a bare `PipeBarrier<PIPE_S>`.
   **PERF-OPTIMIZATION form (P-P101, apply ONLY after the base runs + ONLY if perf needs it):**
   hand-rolled mem-based VEC online-softmax (running max / running sum / expMax rescale),
   replacing the scalar pole of `SoftmaxFlashV2` — and on arch35 it must carry the
   `S_V` sync above to compile (the perf gain is de-scalarizing; it is NOT required for a
   correct/runnable FA). Softmax stat (max/sum/expMax) is
   **fp32, UB-resident, CONTIGUOUS `[m]`** in a small depth ring (no GM round-trip for
   the stat ring). The `[.,8]` broadcast layout is a *separate* temp produced by a
   broadcast step for the row-multiply — never read the stored contiguous stat with a
   `[.,8]` stride. P-P101's precision triad (masked→finite minValue not −inf /
   subtract-rowmax-BEFORE-Exp / vector-Div-with-sum>0) applies. Online correction
   (newSum = oldSum·expMax + curTileSum, O-accumulator ×expMax) runs on non-first kv-tile.

5. **Host-tiling (op_host)** — *part of the op*, not an afterthought. The host computes
   the per-head / per-block tiling (population of both carrier structs, workspace sizing,
   core-split). **Meta-lesson: copy the op_host too — do NOT hand-roll the host tiling.**
   A device cube that is byte-correct can still fault (e.g. multi-head N>1) when the host
   multi-head carrier (per-head offset + workspace + core-split) is only partially
   hand-rolled. (See the large-D TODO below — this is exactly the gap being localized.)
   The core-split distribution math itself is carried generically by **CAND-FA-COREDIST-1**
   (see the Host-Tiling §Core-split cross-link) — that part need not be hand-rolled.

6. **Cross-core buffer-sync policy** — the cube↔vec handshake uses the cross-core
   buffer-sync policy: round-robin slot SELECTORS + per-slot typed-handshake buffers
   (the public attention-common buffer abstraction), NOT hand-rolled `CrossCoreSetFlag`
   in the loop. **Two distinct mechanisms coexist** — do not conflate:
   - **S-channel (QK^T result, cube↔vec)**: `CROSS_CORE_SYNC_BOTH` — a **depth-2
     ping-pong** with an explicit **reverse-release** handshake. The correctness lever is
     the reverse-release wait-before-overwrite (the producer waits for the consumer's
     release of a slot before reusing it), NOT the ring depth. Depth-2 is provably
     sufficient *with* the reverse handshake; a hand-rolled forward-only ring (any depth)
     races on the data-dependent S channel (softmax cost varies with rows/mask → unbounded
     producer↔consumer skew).
   - **Forward-only channels (P→BMM2-A on L1, O=P@V on GM)**: `CROSS_CORE_SYNC_FORWARD` —
     a **depth-N (≈3) margin**, no reverse flag, safe IFF the dataflow structurally bounds
     producer↔consumer skew ≤ depth−1.
   - **Secondary hard ceiling**: all sync'd Buffer instances draw flag IDs from one pool
     of 11 (a `% 11` allocation). A typical FA instantiation uses ~10 of them — adding
     another sync'd ring risks wrap → flag-ID collision → silent cross-core corruption.
     Budget any new channel against the existing in-use IDs.
   - This refines/cross-links **P-P102**'s "one queue per producer↔consumer direction"
     contract: the per-direction handshake is the load-bearing primitive; the SyncType
     (BOTH vs FORWARD) is chosen by where skew is bounded.

## Concrete functional-block inventory (verified whitebox — graybox template substrate)

> Added 2026-06-07 (FA-A5 graybox stage-2 template-ization; retained as historical prior art). The
> verified whitebox FA-A5 forward
> (no-dropout; **13 bit-exact + 2 within-T1-tol** of 15 keep_prob=1 comparable cases — the 2 within-tol
> are case43 d64 + case37 d256 + near-vendor perf; wholeport base 64/64)
> decomposes the §"reusable skeleton" CRTP into NAMED reusable blocks, each with a clear interface.
> The arch35 block bodies remain in the KB as advisory snapshots. The generator may retrieve them to
> identify block interfaces and completeness risks, but must re-derive each selected block from the
> selected arch22 contract rather than inline-copying target code. **Self-contained** = NO external
> `#include "arch35/..."`; it does not waive the no-target-body-copy rule. This table is the
> block-INTERFACE inventory + the stitch points, not a list of include-ready deliverables.

Each row: block (deliverable file) → role → interface (template params / I/O) → which §"reusable
skeleton" component (#1–#6) it instantiates.

| Block (deliverable file) | Role | Interface (params / I/O) | Skeleton component |
|---|---|---|---|
| `wholeport/wp_kernel_base.h` / `wp_kernel_train.h` | orchestration: outer output-block loop + inner kv-loop + the 4-deep staggered AIC/AIV Process pipeline | CRTP base templated on `<CubeBlockType, VecBlockType>`; reads both carrier structs; drives BMM1→release-S→softmax→release-P→BMM2→release-O→ProcessVec2 | #1 carriers + #2 staggered pipeline |
| `wholeport/wp_block_cube.h` | cube block-type: manual `Mmad` BMM1 (QK^T) + BMM2 (P@V) + Fixpipe→UB | template on dtype + D-bucket (`S1TemplateType` Aligned64/128, `DTemplateType` Aligned64/128/256/768); BMM1 dispatch keys off D-bucket (inline vs L0-split); BMM2 inline (small-D non-fp32) vs L1-split-N (fp32/large-D) | #3 cube block-type |
| `wholeport/wp_block_vec_base.h` / `wp_block_vec_train.h` | vec block-type: online-softmax epilogue (`SoftmaxFlashV2` base; UB fp32 contiguous stat ring) | reads S from cube via cross-core S-channel; running max/sum/expMax in UB; writes P forward to BMM2-A | #4 vec block-type |
| `wholeport/wp_attenmask.h` / `wp_attenmask_gs1.h` | feature: atten-mask Select (NO_COMPRESS user-mask read OR self-built compress-causal [2048,2048]) | gated on `hasAtten`; reads mask GM, applies finite-minValue Select before softmax | #4 vec, +1 step (FA+mask delta) |
| `wholeport/wp_pse.h` | feature: PSE bias add to QK^T score before softmax (`PSE_OUTER_ADD_MUL`) | gated on `pseMode!=NONE` (auto-routes Nd path); `[B,N,S1,S2]` or `[B,N,1,S2]`-broadcast bias; optional per-head learnable-sink `[N]` | #4 vec, +1 step (FA+pse delta) |
| `wholeport/wp_dropmask.h` | feature: Philox dropmask gen + apply (keep_prob<1.0) | gated on `hasDropOut`; seeded by truth-returned (seed,offset); s2-align-16 counter→element via public `adv_api/math/philox.h` | recipe phase 6 (dropout) |
| `wholeport/wp_mc_*` (`wp_mc_gm_layout.h`, `wp_mc_offset_calculator_v2.h`, `wp_mc_copy_gm_to_l1.h`, `wp_mc_parser.h`, `wp_mc_fa_gm_tensor.h`, `wp_mc_fa_l1_tensor.h`, `wp_mc_gm_coord.h`) | layout: per-(b,n,s) GM offset compute for BSH/SBH/BSND/BNSD + GM→L1 staging | layout-selector picks which axis-index × which precomputed stride; compute is layout-invariant | dimension lever: layout offset |
| `op_kernel/regbase_copyin.h` / `regbase_matmul.h` / `regbase_fixpipe_out.h` / `regbase_buffer*.h` | copyin/matmul/fixpipe primitives (C310 Fixpipe, 2D-V2 load `dstStride=mStep`) + the sync-flag buffer pool | `regbase_matmul.h` = the `Mmad`+`MmadParams` wrapper; `regbase_fixpipe_out.h` = L0C→UB/GM Fixpipe (`srcStride` per-context units); `regbase_buffer*.h` = the `%11` flag-id pool (§6 secondary ceiling) | #3 cube + #6 cross-core sync |
| `wholeport/wp_common_regbase.h` / `wp_util_regbase.h` / `wp_util.h` / `wp_infer_comm.h` / `wp_fa_entry.h` | shared carriers, util, the kernel entry-point | the per-iteration + invariant carrier structs (ONE shared host+kernel header per CAND-FA-CARRIER-1 PIECE-D copy-line rule) | #1 carriers |

**Stitch points (the load-bearing template-assembly fact)**: a new FA-class op REUSES `wp_kernel_base`
+ `wp_block_cube` + `wp_block_vec_base` + the matching `wp_mc_*` layout block + `regbase_*` UNCHANGED, and
adds/swaps only the feature blocks (`wp_attenmask`/`wp_pse`/`wp_dropmask`) per the FA+X delta table below.
The block INTERFACES (template params: dtype, S1/D bucket, feature gates) ARE the stitch points.

**Generation-chain verified (gap#1 closed, DS-measured 2026-06-07)**: the W4 fresh-stitch proved this
template substrate GENERATES a working op — assembling the copied block bodies + host `DoTiling` for one
config (fp16/BNSD/D128/dense, B1·N32·S128) produced a self-contained build (no `#include "arch35/"`) that
is **bit-exact** vs the vendor oracle (DS 3-bar: subset 320/1346, provenance clean, max_abs=0). I.e.
"templates + knowledge → fresh-build → bit-exact" is an empirically-closed chain, not just copy+compile.
Evidence artifact: branch `archived/pr/fa-a5-w4-freshstitch` @ `004f8c43`. The 15-config spec-map (which
stitch-class per dtype×D-bucket×layout, with the bit-exact-vs-within-tol criterion) is in the design doc
`docs/design/FA_A5_WHITEBOX_DESIGN_AND_PLAN.md` §14.

**Known gaps the template carries (inherited from whitebox — NOT fixed here, separate workstreams own them)**:
- **case37 (d256, the s1=64 core-fill class)** — root (corrected 2026-06-07): the host core-fill
  override (§"Host-tiling LOGIC" step 3 below) drops `s1BasicBlock`→64 for small shapes, which REQUIRES a
  matching `S1TemplateType::Aligned64` device kernel; but the Aligned64 s64 device-variant is wired ONLY
  for fp16-D128 (`wp_fa_do_fp16_bnsd_d128_s64`) — the d256/Aligned256 band has NO Aligned64 s64 variant, so
  host (s1=64) and device (falls back to the s1=128 kernel) mismatch → 507015 aicore-exception. **Faithful
  fix** = parameterize `S1TemplateType` as a common function shared across D-buckets (every bucket gets the
  s64 variant), NOT a per-bucket host-gate. The shipped WS1 host-gate (`dBasicBlock<=256`→`==128`) is a
  tracked-temp workaround, not the faithful fix. Template carries the gap.
- **dropout (keep_prob<1.0)** — WIRED + verified (2026-06-11; was previously scoped-out). Dropout is
  **wiring, not new math**: the train-path block (`wp_fa_entry.h → wp_kernel_train.h → wp_block_vec_train.h`)
  already constructs `DropMaskInfo` from `ip.{seed,offset,keepProb,keepProbUint8,needDropMaskOp,dropMaskOuter}`
  under `if constexpr(hasDrop)` and calls `GenDropMask<hasDrop>` (`wp_dropmask.h` faithful vendor Philox:
  7-round, `CompareScalar LE` vs `keepProbUint8=(uint8)(keep_prob*255)`); the APPLY (Select×P + Muls 1/keep_prob)
  lives in the vf basic blocks. The recipe is: (1) add `hasDrop=true` FA_DO_LIST variants mirroring each
  no-drop suffix; (2) pybind sets `ip.{seed,offset,keepProbUint8,needDropMaskOp=0,dropMaskOuter=0}` (drop_mask
  absent ⇒ internal-gen path, OL-202) and dispatches the `_drop` instance when keep_prob<1.0; (3) harness
  captures the reference's returned `(seed,offset)` and feeds the candidate (capture-and-replay, OL-217).
  **Do NOT rewrite the Philox math.** no-drop instances stay byte-identical (hard floor preserved by
  construction). Evidence: kw-2 instantiated 28 instances (14 `_drop`) → BUILD RC=0 first try (K5 dual-pass
  guard, see §K5), precision 48/48 bit-exact incl. 33 dropout cases via Philox replay, 0 compile-fix +
  0 precision-fix iters. (FORBIDDEN, per §K5: the `#define __NPU_ARCH__ 3510` at kernel-TU top to give the
  host pass MicroAPI — it pulls arch35 DEVICE headers into the host_bisheng pass → host BLOCKED 0/21; the
  host pass must EXCLUDE the device body via `#ifndef __ASC_NPU_HOST__`, not acquire MicroAPI.)

## Dimension handling

The skeleton handles the contract's dimension axes via these levers (each a small,
named knob on the unchanged skeleton):

- **Layout offset** (BSH / SBH / BSND / BNSD) — the only layout-dependent code is the
  GM offset computation per (batch, head, seq) into Q/K/V/O. The compute is
  layout-invariant; the carrier offsets differ. (Empirically: with no atten_mask, all
  layouts produce dense attention — see OL-202 below.)
- **D-tiling** — head-dim buckets select the cube dispatch (small-D inline vs large-D
  L0/L1 split). The vendor provides D-templates for specific buckets (e.g. 160/192/256/
  768); intermediate sizes (320/384/512/640) use the next-bucket template (all truth-
  valid). D>256 enables the split path; D≤768 is the op truth-limit.
- **GQA (gSize)** — grouped-query attention is a `gSize` multiplier in the head-index
  mapping (multiple Q heads share one KV head); it is a carrier/offset knob, not a
  structural change to the pipeline.
- **dtype dispatch** — fp16 / bf16 / fp32 select cube paths (fp32 inputs take the
  L1-split-N BMM2 path); softmax stat stays fp32 regardless of input dtype.
- **head-dim C0-align** — when D is not C0-aligned, pad on-device in L1 (an on-device L1
  pad), NOT a host-side pad of the input tensor. (Host-pad was an interim hack on the
  clean lane; the on-device L1 pad is the correct form.)

## Hardware / op facts (public, KB-able)

These are public AscendC / op facts — no CANN source.

- **C310 Fixpipe** — the A5/V351 cube→UB/GM datapath uses the **C310 Fixpipe**, NOT the
  V220 Fixpipe. A V220-line-port that assumes V220 Fixpipe semantics is wrong for A5.
- **2D-V2 load `dstStride = mStep`** — the 2D-V2 load path uses `dstStride` equal to the
  m-step. (Public load-API fact; matters for correct L1→L0 staging.)
- **`hasAttenMask` → sparse-inert** (codified **OL-202**) — the sparse feature is gated
  on the OPTIONAL `atten_mask` input. When `atten_mask` is absent, the op stays dense
  regardless of `sparse_mode` — `sparse_mode` is a PHANTOM axis for the benchmark
  (it passes 0/50 atten_mask → all 50 dense). The benchmark's input config, not the
  op's documented capability, defines the truth. (OL-202 caveat: `drop_mask` is ALSO an
  optional input but its absence-behavior differs — the FA generates the dropMask
  internally; absence-behavior is op-specific, verify empirically.)
- **D≤768 truth-limit** — `HEAD_DIM_MAX <= 768` in op_host; the op errors on D>768
  ("Head dim must <= 768"). Defines the 48/50 ceiling.
- **Dropout = Philox via public `adv_api/math/philox.h`** — when keep_prob<1.0, dropout is
  bit-reproducible: `npu_fusion_attention` RETURNS the (seed, offset) pair it used; the
  kernel replays the same Philox stream → bit-match. The mask derivation is an
  **s2-align-16 counter→element** scheme (the Philox counter is derived from
  (seed, offset) + (b, n2, g, s1, s2) with s2 size aligned up to a multiple of 16;
  the Philox dropmask generator runs Philox4x32-10 via the PUBLIC `adv_api/math/philox.h`).
  The FA op_kernel
  CONSUMES a bit-packed dropMask (DataCopyPad + Select); the generator is a separate
  step. In-kernel reproduction = public `philox.h` + the counter-derivation + apply.
  The internal scheme is NOT a standalone `npu_dropout_gen_mask` (that variant gives the
  right rate but the wrong mask).

## Recipe — phase order

Generate an FA-class op in this dependency order (each phase verifies before the next):

1. **Foundation** — carriers (the per-iteration + invariant carrier structs), file split (`<op>_cube.h` /
   `<op>_vec.h` / `<op>_kernel.h`+`.cpp` orchestrator, class names carrying the literal
   `Cube`/`Vec` markers per P-P102), task type `KERNEL_TYPE_MIX_AIC_1_2`.
2. **Cube** — manual Mmad BMM1 (QK^T) + BMM2 (PV), Fixpipe to UB, dtype/D dispatch.
   **Before writing: read the EXECUTABLE params in `fa_class/cv_reference_concrete_params.md`
   (repo `src/skills/references/target/ascendc/fa_class/cv_reference_concrete_params.md`)
   §`matmul_primitive` / §`block_tile_alignment` / §`qk_load_form` — the `Mmad` signature,
   `MmadParams` fields, L0A/L0B fractal-layout, Fixpipe `srcStride` per-context units, and the
   16/C0 tile-alignment rule live there. The skeleton + P-P102 give the contract; this file gives
   the code.**
3. **Vec** — online-softmax (P-P101): fp32 contiguous stat ring, precision triad,
   online rescale.
4. **Integrate** — wire the cross-core buffer-sync policy (S-channel `CROSS_CORE_SYNC_BOTH`
   depth-2 reverse-gated; P/O forward-only), the 4-deep staggered Process pipeline (OL-200).
5. **Tiling (op_host)** — **copy the full arch35 op_host multi-head tiling**; do NOT
   hand-roll. (host-tiling is part of the op.) **[large-D component → TODO below]**
6. **Dropout** — only if keep_prob<1.0 in scope: capture the returned (seed, offset),
   replay via public `philox.h` + s2-align-16 counter, apply mask.

### Meta-lessons (carry across FA-class ops)

- **Copy op_host too** — don't hand-roll host tiling; a correct device kernel still
  faults when the host multi-head carrier is only partially ported.
- **Host-tiling is part of the op** — not an optional add-on.
- **Dropout via returned-(seed, offset)** — reproducibility comes from replaying the
  op's own returned Philox state, not from re-deriving a standalone mask.
- **Empirical-ceiling discipline** — run the truth (ground "可抄么 / 试过么") before
  declaring a case impossible; the D>768 ceiling is grounded in the op erroring, not
  assumed.
- **Ground-before-assert** — validate against the vendor source + disk + a runnable
  reference, not against our own derived artifacts.

## Host Tiling — first-class component (peer of cube/vec, not an afterthought)

The host tiling computation is a **first-class component of the FA template**, equal in
weight to the cube and vec blocks — a kernel that copies the device blocks verbatim but
hand-rolls the host tiling will pass small/simple cases and then fault on the regimes the
host is supposed to configure (large-D, GQA, multi-core split). Treat it as a port target,
not glue.

**What the host tiling computes** (the math to faithfully port, by category):
- **K-tiling config** — for head-dim larger than the single-K base, split the contraction
  dim into base-sized chunks and emit the per-chunk loop bounds the device block consumes;
  the device block already K-tiles, so this is *config the block expects*, not new logic.
- **D-bucket selection** — map the actual head-dim to the dispatch bucket.
- **Workspace sizing** — size every GM scratch region to the actual dims (use the real
  type size; a too-small fixed size silently floors then OOBs at large dims).
- **Carrier struct + cross-core scalar blit** — the host populates a ≤128 B POD carrier; the
  cube reads the scalar params via a one-shot cross-core flag → `reinterpret_cast` POD→uint32
  array → copy `sizeof(POD)/4` words from a fixed on-chip scratch base. Mechanism + 7-stage
  host-tiling lifecycle + workspace=`AlignUp`-regions+reserve are carried by
  **`patterns/unverified/candidates.md` → CAND-FA-CARRIER-1** (PIECES A/B/C, generic). **CRITICAL
  repro rule (PIECE D copy-line)**: the carrier byte-layout (field order/widths/bitfield packing)
  CANNOT be shipped in KB or reconstructed from prose — the word-blit needs byte-identical
  host/kernel layout. Author the carrier as ONE shared host+kernel header (defined once, ≤128 B)
  so the layout is internally consistent by construction; do NOT independently re-type it on each
  side. This sidesteps the copy-line — no hard copy-boundary blocks repro.
- **Core-split** — partition output rows / heads across cores; populate both carrier structs.
  The distribution-policy math (round-robin for L2 K/V reuse, symmetric mirror for
  causal/sparse load-balance, boustrophedon snake for varlen TND) + the exact
  offset-arithmetic (deal→unflatten→reflatten region-base table, reverse-partial pivot
  `(fwdPartialLen-1)%cores`, TND ordinal accum-walk) is carried by
  **`patterns/unverified/candidates.md` → CAND-FA-COREDIST-1** (index-arithmetic
  empirically re-derivable from that entry alone — kw-wb5 partition-oracle PASS 2026-06-03,
  so do NOT hand-roll the core-split; read the entry's EXPLICIT-GENERIC offset-arithmetic
  section). It is generic integer math (no vendor primitive) — apply it, adapting only the
  tiling constants to the actual shape.

**Contract-reconstruction discipline**: the target op_host is **framework-coupled** — it reads/writes
through the target host-tiling
context (platform descriptor / shape registry / compile-info), which is NOT the same
interface as a pybind wrapper (which computes a tiling struct from torch tensor shapes).
The host tiling must be reconstructed from the selected arch22 contract. Use target prior art to
enumerate every required category above and to design differential checks, but do not mirror its
arithmetic as the generated body. Adapt to current arch35 public APIs, build the task-owned artifact,
and compare it with selected-source truth. If a field cannot be derived or validated, surface that
contract gap instead of substituting the target implementation.

## Host-tiling LOGIC template (verified — parameterized by op-config)

> Added 2026-06-07 and retained as advisory knowledge. The op_host stages below identify required
> contract categories and review anchors. Re-derive their task-owned equivalents from selected arch22
> source and current arch35 APIs; do not copy the archived target tiling body into `DoTiling`.

The host computes the tiling POD (`FlashAttentionScoreSimplifiedTilingData`) from the op-config (dtype,
B/N/N_kv/S1/S2/D, sparse_mode, pre/next_tokens, keep_prob, mask-present). The LOGIC, in faithful port
order (each a named arch35 op_host stage, parameterized by op-config — NOT a verbatim include):

1. **the sparse-tiling computer** (arch35 op_host sparse-info stage; role-anchors
   `PretokenAndNexttokenAdjustment` / `GetSparseInfo`) — maps user `SparseMode {NO_MASK,ALL_MASK,
   LEFT_UP_CAUSAL,RIGHT_DOWN_CAUSAL,BAND,...}` → internal `SparseEnum {ALL,NONE,CAUSAL,BAND,...}` +
   `AttenMaskCompressMode` + (preTokens, nextTokens, s1/s2SparseValidSize, needCompressMask). KEY rule
   (OL-202 / OL-85): when no explicit `atten_mask` tensor is provided, the op IGNORES sparse_mode → force
   DENSE (`SparseEnum::ALL`, no mask read) to match the reference. Causal-compress engages ONLY with an
   explicit mask. preTokens/nextTokens clamp to int32. RIGHT_DOWN_CAUSAL with S1≠S2 → BAND (nextTokens=S2−S1).
2. **the basic-block computer** (arch35 op_host basic-block stage; role-anchor `CalcS1S2BasicBlock`) —
   `s1BasicBlock` = 128 for D≤256; for D>256: 64 (fp32) / 128 (fp16/bf16). `inputDtypeBytes` = 4 (fp32) /
   2 (fp16/bf16). Parameterized by D and dtype.
3. **the core-fill override** (arch35 op_host total-size stage; role-anchor `CalcTotalSize`) — if
   `totalUnits = B*N_kv*gSize*ceil(S1/s1BasicBlock) < aicNum(28)` AND dtype≠fp32 AND
   `dBasicBlock=AlignUp(D,64) ≤ 256` → drop `s1BasicBlock` to 64 (fills idle cores on small shapes). The
   device kernel for the s1=64 class must use `S1TemplateType::Aligned64` (the launcher must match).
4. **D-bucketing** — `dBasicBlock = AlignUp(D, 64)`; truth-limit `dBasicBlock ≤ 768` (HEAD_DIM_MAX, the
   48/50 ceiling). Round UP to the device template tier (`DTemplateType` Aligned64/128/256/768); non-
   power-of-2 D (160/192→Aligned256; 320/384/640→Aligned768) reach a valid bucket. Aligned256=Dn path;
   Aligned768=splitD path.
5. **GQA wiring** (host-side, the COMPLETE GQA enabling) — `N_kv = k.sizes()[1]`, `gSize = N / N_kv`
   (require `N % N_kv == 0`). Set `ip.n2Size = N_kv`, `ip.gSize = gSize`. The kernel reads K/V at the
   N_kv-strided head (broadcast across gSize query heads); Q/output use query head `n2oIdx*gSize+goIdx`.
   This is COMPLETE GQA — a host-only knob, no kernel change (cross-ref OL-205: FA GQA gaps are host-wiring).
6. **Tiling POD population** — fill `inputParamsRegbase` (bSize, n2Size, gSize, s1Size, s2Size,
   alignedS2=AlignUp(S2,16), dSize/dSizeV, keepProb, scaleValue, preTokens, nextTokens, layoutType,
   attenMaskShapeType/CompressMode/S1Size). `std::memset(&td,0,sizeof(td))` first.

**Core-split distribution** (the multi-core partition math; role-anchors `SetMultiCoreParamsRegbase` +
`SetSplitCoreModeParam`) is NOT hand-rolled — it is carried generically by **CAND-FA-COREDIST-1**
(round-robin / symmetric-mirror / boustrophedon) + **CAND-FA-CARRIER-1 PIECE-B** (the carrier-population of
the split-mode flag + cheap-prefix index by sparse pattern). The host SETS those carrier fields; COREDIST-1
CONSUMES them. This recipe feeds those two.

**Faithful-port discipline (the key rule)** — the vendor op_host is GE-framework-coupled (reads/writes
through the platform descriptor / shape registry); a pybind wrapper computes the same struct from torch
tensor shapes. Port the tiling MATH line-faithfully (every stage above); adapt ONLY the I/O boundary
(GE-tiling-context ↔ pybind). Do NOT hand-roll the math "to fit our host" — that re-introduces the guess-
and-patch churn the verbatim path avoids. RED LINE (port_a3): CPU host C++ only — NO `#include "arch35/"`
device headers, NO aclnn/aclop in `DoTiling`. The launch-side dispatch that consumes this POD is
**CAND-FA-LAUNCH-DISPATCH-1**; the full A3-config→op assembly is **CAND-FA-STITCH-1**.

## Verified whitebox knowledge (K1–K4) — S1Template / core-fill (graybox/blackbox reproduction spec)

> Added 2026-06-07. Codifies the verified knowledge from the FA-A5 S1Template/core-fill faithful-fix
> (origin/main `f94b510a`): `S1Template` parameterized into a rule-driven macro dimension, host dispatch
> became the common `SelectLauncher`/`SelectPseLauncher`, the `==128` host-gate removed. Result: whitebox
> no-dropout precision = **48 bit-exact** (was 34 bit-exact + 12 within-tol + 2 KERNEL-ERR). K1–K4 are the
> spec a future graybox/blackbox assembler reproduces. K1 is stated to match the SHIPPED code
> (`output/a3_to_a5_port/src/kernels/flash_attention_score/op_kernel/pybind11.cpp`), NOT a paraphrase.

- **K1 — core-fill rule (code-exact)**. The host `s1BasicBlock` is computed in two steps (the
  parameterized math is in §"Host-tiling LOGIC" steps 2+3 — not restated here):
  1. **base** (`CalcS1S2BasicBlock`, `pybind11.cpp` L386–391): `s1BasicBlock = 128` when `D ≤ 256`; when
     `D > 256` then `s1BasicBlock = 64` for **fp32** (`inputDtypeBytes == 4`) / `128` for fp16/bf16.
  2. **core-fill override** (`CalcTotalSize`, `pybind11.cpp` L395–401): with
     `totalSize = B·(N_kv·gSize)·CeilDiv(S1, s1BasicBlock)` and `dBasicBlock = AlignUp(D, 64)` —
     **if `totalSize < aicNum(28)` AND `inputDtypeBytes != 4` (i.e. NOT fp32) AND `dBasicBlock ≤ 256`
     → drop `s1BasicBlock` to 64.** The device kernel for the resulting s1=64 class MUST select the
     `S1TemplateType::Aligned64` (`_s64`) launcher (`useAligned64Kernel = (s1BasicBlock == 64) &&
     (dBasicBlock <= 256)`, `pybind11.cpp` L410).
  - **fp32 reality (the precise point the design note got imprecise on)**: fp32 is **NOT** universally
    s1=64. fp32 = 128 for `D ≤ 256` and = 64 **only for `D > 256`** (from the base step). The core-fill
    override **EXCLUDES fp32** entirely (`inputDtypeBytes != 4` guard) — fp32 small shapes do NOT core-fill.
    So the only way fp32 reaches s1=64 is the `D > 256` base branch, never the core-fill path. (The design
    note §④/§③ phrased it as "fp32 默认即 64", which over-generalizes; the code-exact statement is above.)
- **K2 — host-rule ↔ kernel-instance consistency invariant (THE root the fix closed)**. Every
  `(dtype, D-bucket, s1BasicBlock)` the host `DoTiling` can compute MUST have a matching device kernel
  instance + launcher; otherwise the host selects an s1=64 dispatch with no `_s64` kernel → the device
  falls back to the s1=128 kernel → host/device mismatch → 507015 aicore-exception (KERNEL-ERR). The
  pre-fix bug: host computed s1=64 for any `(dtype × bucket≤A256)` small shape, but the kernel TU wired
  the `_s64` variant ONLY for fp16-D128 → every other core-fill combination had no matching instance.
  **Assembly-completeness self-check = this invariant**: when the host rule changes (e.g. widening
  core-fill to all dtypes/buckets), the kernel instance set MUST close over it in the SAME change.
- **K3 — S1Template parameterization pattern**. `s1` is a **cross-cutting parameter dimension** over ALL
  feature families (dense / pse / atten / drop) — core-fill is keyed on `totalSize` and is feature-agnostic,
  so the s1=64 variant is needed for every family. Generate the variants via a **rule-driven parameterized
  macro** (s1Enum lifted to a macro parameter, expanded by an X-list over buckets), NOT per-entry one-off
  kernels (the pre-fix anti-pattern = 8 macros hard-coding `Aligned128` + one hand-written `_s64` one-off,
  violating the host-common-function principle). Each instance is **3-in-lockstep**: device definition
  (`#ifndef __ASC_NPU_HOST__`) + host-pass extern-C declaration (`#else`) + launcher (`<<<>>>` dispatch),
  all driven by the same X-list so they stay synchronized (a hand one-off most often drops one of the three).
  The host side mirrors this: ONE common `SelectLauncher(dtype, dBucket, s1Basic, feature)` /
  `SelectPseLauncher(...)` parameterized function, not per-bucket if-else copies.
- **K4 — test-coverage lesson**. Each `(dtype, D-bucket)` core-fill small-shape case MUST be in the test
  set, else the gap stays hidden. The earlier "64/64" milestone missed c2/c42 (bf16-d128 core-fill) and
  case37 (fp16-d256 core-fill) precisely because those small-shape core-fill paths were never exercised —
  a clean measurement of an incomplete test set hid the host-rule↔kernel-instance gap. The reproduction
  test set must enumerate the core-fill class per dtype × bucket.
- **K5 — dual-pass compile-method (the 2026-06-08 canon-0/21 BLOCKER root)**. build_ascendc.py
  compiles the kernel TU TWICE: a **device pass** (MicroAPI available; `AscendC::MicroAPI` is gated
  `__NPU_ARCH__==3510`, set by the device toolchain) + a **host pass** (build_ascendc.py DEFINES
  `__ASC_NPU_HOST__`, NO MicroAPI). The `#ifndef __ASC_NPU_HOST__` guard (K3) makes the device body an
  EMPTY TU in the host pass — that IS the host pass's job: **EXCLUDE the device body, NOT acquire
  MicroAPI**. Launch stubs are auto-generated by build_ascendc.py from the device object; pybind
  declares `extern "C" uint32_t aclrtlaunch_<name>(uint32_t blockDim, void* stream, ...)`. **FORBIDDEN**:
  `#define __NPU_ARCH__ 3510` at the kernel-TU top to give the HOST pass MicroAPI — it pulls arch35
  DEVICE headers (`PIPE_FIX`@kernel_event.h, `int4x2_t`@vconv_impl, SIMT `cce/dim3/Atomic*`@simt_api)
  into the host pass → host_bisheng cannot compile CANN's arch35 device headers → BLOCKED. Evidence:
  #3 (D=1024) used the guard → RC=0 + DS-verified 3/3; the clean-room canon assembler used
  `__NPU_ARCH__=3510` → BLOCKED 0/21 — SAME build_ascendc.py + SAME wholeport, the difference is purely
  this compile-method. K3 names the guard STRUCTURE; K5 names WHY (dual-pass) + the FORBID a clean
  assembler must know (else it dead-ends on the host-pass MicroAPI question, as the canon did).
  **Backward cross-validation (flash_attention_grad, 2026-06-14)**: K5 generalizes to a white-box FA
  **backward** op (cube `MatmulImpl` AIC + AIV, non-template-assembly — not the forward template-assembly
  path) — the `#ifndef __ASC_NPU_HOST__` guard gives host pass an empty device TU and device pass compiles
  the MicroAPI bodies (gated `__NPU_ARCH__==3510` by the toolchain, NOT set by us) → BUILD RC=0 first try.
  Confirms the dual-pass guard is direction-agnostic (fwd template + bwd hand-written both compile clean).

**Graybox/blackbox evolution**: a graybox assembler takes the arch22 input + K1–K4 → generates the
**complete** s1-variant set from K1+K3 (no missing combination), gates assembly-completeness on K2 (host
rule and kernel instance set close together), and validates with a K4-complete test set. The faithful fix
is the parameterized macro + common `SelectLauncher`, never a per-entry one-off or a `==128` host-gate.

## FA+X extension model — X = stage-delta on the unchanged skeleton

The product value: a new FA-class op = **skeleton (unchanged) + a small X-delta**, made
possible *because* arch35 made the variants pluggable block-types. The delta is localized:

| X | Delta on the skeleton |
|---|---|
| **FA + bias / pse** | vec block-type: **+1 step** (add bias/pse into the score before softmax) |
| **FA + RoPE** | cube/QK-load: **QK-load + rope** (apply rotary embedding at Q/K load time) |
| **FA + quant** | swap to the **existing quant cube block-type variant** (arch35 already ships a quant cube path) |
| **FA-decode** | **swap VecBlockType** (decode-shaped softmax/epilogue, skeleton unchanged) |

**Components also generalize beyond FA**: cube-MIX (**P-P102**) → all cube-class ops;
online-softmax (**P-P101**) → reductions; optional-input gating (**OL-202**) → any
optional-input feature; the staggered MIX pipeline (**OL-200**) → all MIX_AIC ops.

**Honest caveat**: the template is a reproducible *guide* (architecture + public facts +
method). An X-delta still needs either a CANN reference to study (has-ref) or
generate-from-guide (the ref-less frontier) — the template removes the from-scratch cost,
it does not eliminate the per-X engineering.

## BACKWARD section — the FA-grad stitch recipe (parallel to the forward stitch)

> Added 2026-06-20 (distilled from the directive-2 FEASIBILITY DEMO: a MANUAL narrow-slice port of
> CANN `flash_attention_score_grad` arch35 → a self-contained A5 op, build+run+precision MEASURED-YES
> on Ascend950PR_9579/dav-c310 for the **S≤128 fp16/bf16 dense core only**). Source: the whitebox log
> (`workspace/fa_grad_template/whitebox_log.md`) + `CAND-FA-TEMPLATE-GEN-BWD-1`. **Scope-honesty up front:
> this section describes how to stitch the FA-GRAD backward; it is a reusable RECIPE a graybox/orch run
> can FOLLOW, NOT a finished generator. What was demonstrated = ONE config (BNSD, S≤128, fp16/bf16,
> dense, D128, split-axis BN2). The §"What is reusable vs op-specific-gap" table at the end is the
> load-bearing honesty — read it before assuming any piece generalizes.**

### Backward = the SAME forward template machinery, different blocks (the core fact)
CANN `flash_attention_score_grad` arch35 is **structurally isomorphic** to the forward FA template:
SAME CRTP kernel `<CubeBlockType, VecBlockType>`, SAME `KERNEL_TYPE_MIX_AIC_1_2`, SAME raw `<<<coreNum>>>`
launch, SAME host-tiling POD + ASCENDC_TPL axis enumeration, SAME staggered AIC/AIV ping-pong (OL-200).
So the forward stitch RECIPE applies; what changes is (a) **3 launch-phases not 1**, (b) **5 cube GEMMs not
2**, (c) **softmax-GRAD vec not online-softmax**, (d) the backward consumes the forward-saved
`softmax_max / softmax_sum / attention_in` (no recompute of the softmax stats from scratch).

FA-2 backward math the blocks compute (matches CAND-FA-GQA-BWD-1, the validated multi-launch reference):
dV=Pᵀ@dO, dP=dO@Vᵀ, dS=P∘(dP−rowsum(dP∘P)), dQ=(dS@K)·scale, dK=(dSᵀ@Q)·scale. The `rowsum(dP∘P)`
term = the softmax-grad ("sfmg" = SoftMaxGrad) the vec block computes.

### Step 1 — RE-DERIVE the shared `regbase_*` layer from the selected contract
The backward's 5 GEMMs call the SAME cube primitives (`MatmulFull/K/N`, `MMParam`, C310 `LoadDataToL0A/B`,
the Fixpipe) the forward cube uses. Treat the archived `regbase_matmul.h` /
`regbase_fixpipe_out.h` / `regbase_buffer*.h` / `regbase_copyin.h` as API and structure examples.
Emit task-owned shared components from the gradient and saved-tensor contract, and validate them with
CPU fp64 autograd; do not reuse target bodies unchanged.

### Step 2 — MAP target FA-grad prior-art blocks to task-owned backward components
Inspect the archived arch35 blocks only to recover interfaces, sequencing, and completeness hypotheses.
Implement corresponding `wp_fag_*` components from the forward/gradient contract and current public APIs
(`shared/common/...`, no `#include "arch35/"`), without copying target bodies:

| Advisory target block | → task-owned component | Role to re-derive |
|---|---|---|
| `flash_attention_score_grad_common.h` | `wp_fag_common.h` | carriers `commondef::{FagConstInfo,FagRunInfo}` + enums + ARGS_TRAITS feed |
| `flash_attention_score_grad_tiling_data_regbase.h` | `wp_fag_tiling_data.h` | the kernel-read POD `FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase` + sub-params (host-pass-safe: `<cstdint>` only) |
| `flash_attention_score_grad_block_cube.h` | `wp_fag_block_cube.h` | cube block: 5 GEMMs `IterateMmDyV`(dP)/`IterateMmQK`(recompute-S)/`IterateMmDsK`(dQ)/`IterateMmDsQ`(dK)/`IterateMmPDy`(dV); DEFINES the `ARGS_TRAITS` macro |
| `flash_attention_score_grad_block_vec.h` | `wp_fag_block_vec.h` | vec block: softmax-grad (`CalculateCastSoftmaxGrad`) + `BroadcastSubMul`(dS) + `SimpleSoftMax`(recompute-P) |
| `flash_attention_score_grad_kernel_base.h` + `_kernel.h` | `wp_fag_kernel_base.h` + `wp_fag_kernel.h` | the Pre/Base/Post orchestrator + the cv ping-pong `Process()` |
| `flash_attention_score_grad_s1s2_bn2gs1s2_{pre,post}_regbase.h` | `wp_fag_pre.h` / `wp_fag_post.h` | Phase-1 init/cast-prep / Phase-3 fp32-ws→out-dtype reduce |
| `vector_api/{cast_softmax_grad, vf_broadcast_sub_mul, pse_atten_mask_muls_simple_softmax, vf_softmax_grad_front_cast_aligned{256,512,768}_{f16,f32}, vf_muls_sel_simple_softmax*, vf_cast_transdata_deconflict, vf_ds_abs_reduce_max, vf_transdata}.h` | `wholeport/vector_api/*` | the softmax-grad VEC leaves (MicroAPI register-compute) |

Plus the transitive shared-common closure into `shared/common/`: `util.h`, `util_regbase.h` (defines
`regbaseutil` + `DTemplateType`/`S1/S2TemplateType` enums), `buffers_policy/buffer/buffer_manager/
buffer_mix_core.h`, `matmul/FixpipeOut.h`, `const_def.h`, `flash_attention_score_common_regbase.h`, and
the feature headers `attenmask/pse/dropmask.h`.
**TRIM** (out of the dense-core scope → tracked GAPS): quant (`*_quant`, `vf_anti_quant_*`), `presfmg`
(hifp8 pre-pass), deter/tnd/nz/rope/sink variant blocks.

### Step 3 — AUTHOR the 3-phase entry `wp_fag_entry.h` (templated `wp_fag_regbase_impl<IN,float,OUT,s1T,s2T,dT>`)
Modeled on CANN `flash_attention_score_grad_entry_regbase.h` (`INVOKE_FAG_GENERAL_S1S2_BN2_REGBASE_IMPL` for
the BN2 axis / `..._BN2GS1S2_...` for the multi-block axis). The 3 phases in ONE `<<<>>>` launch:
- **Phase 1 PRE** — `FlashAttentionScoreGradS1S2BNGS1S2PreRegbase` (dq/dk/dv init / cast-prep). For
  **BN2 dense (no sink): SKIP** — entry gates Pre on `sinkOptional` (entry L190-197). `pipeIn.Destroy()`.
- **Phase 2 BASE** — CRTP `FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>` via
  `std::conditional<g_coreType==AIC, BlockCube, BlockCubeDummy>` / `<AIV, BlockVec, BlockVecDummy>`. The
  5-GEMM + softmax-grad MIX. `op.Init(key,value,dy,query, ...28 args..., dq,dk,dv, ..., workspace,
  tilingData, &pipe)`; `op.Process()`.
- **Phase 3 POST** — `FlashAttentionScoreGradS1S2BNGS1S2PostRegbase` (fp32-ws→out-dtype cast/reduce). For
  **BN2 dense (no sink): SKIP** (entry L226 gates on `sinkOptional`) — BN2 writes dq/dk/dv DIRECT via the
  `IS_DQ/DK/DV_WRITE_UB` paths in Base. For the BN2GS1S2 axis (S>128), Post runs for non-fp32 dtypes.
- Required head boilerplate: `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` + `SetSysWorkspaceForce(
  workspace)` (the fa-a5-kw-21 carry — bind sys-workspace base under custom `<<<>>>` launch BEFORE
  `GetUserWorkspace`, else `user` lands outside our allocation).

### Step 4 — the **splitAxis DECISION RULE** (THIS was the precision bug — get it right)
The host picks `splitAxis` (CANN `SetSplitAxis`, `op_host/arch35/flash_attention_score_grad_tiling_common_regbase.cpp` L1711):
```
isBn2 = (s1 ≤ 128) && (s2 ≤ 128) && (n1 == n2) && (d ≤ 512) && (dtype ≠ fp32)
        && (d == d1) && !hifp8 && !rope && (tailZeroCount == 0)
splitAxis = isBn2 ? BN2 : BN2GS1S2     // (BN2_MAX_S=128, BN2_MAX_D=512)
```
**BN2** = split by batch-head only, the SIMPLE path: writes dq/dk/dv DIRECT via `IS_*_WRITE_UB`, Pre/Post
gated on sink (dense → just Base). **BN2GS1S2** = the multi-block path: dq/dk/dv go through fp32 workspace
+ Post. **Hardcoding BN2GS1S2 for an isBn2-regime shape → wrong Process branch → garbage/zero output.** The
entry's `SPLIT_AXIS` constant AND the host POD must agree with `isBn2`.

### Step 5 — host tiling is PART OF THE OP (the field-by-field port; stubbing → garbage)
**The single most error-prone piece.** The kernel reads many interdependent POD fields; any left 0 →
garbage. Port the CANN op_host MATH faithfully into the pybind `DoTiling`
(`op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp`, cited):
- **⚠️ THE BACKWARD HAS NO CORE-FILL — do NOT import the forward's** (graybox-found 2026-06-20; this
  was the ONE asset-gap that an autonomous follow-of-this-recipe hit → `dk/dv` garbage on every S≤128
  case, `dq` fine). The FORWARD's step "CalcS1S2BasicBlock core-fill → `s1BasicBlock`=64 for small
  shapes" does NOT exist in the backward. The BACKWARD `GetS1S2TemplateType` (`common_regbase.cpp`
  L870) has NO core-fill: dense fp16/bf16 ALWAYS `s1Template=128`, `CUBE_BASEM=128`; `useS64` /
  `totalUnits` override is NEVER selected for dense fp16/bf16 D128. **This is the single place the
  forward and backward host-tiling DIVERGE** — port the backward `GetS1S2TemplateType`, NOT the
  forward `CalcS1S2BasicBlock` core-fill. (Apply the forward core-fill → s1BasicBlock=64 → wrong
  CUBE_BASEM → dk/dv garbage.)
- **DoSplit (L316-347)**: `FuzzyForBestSplit` → `s1Inner = s1Template/2`, `s2Inner = s2Template`,
  `sfmgdInner = dTemplate` (fp16/bf16 default template = 128/128 → s1Inner=64, s2Inner=128). `s1CvRatio=2,
  s2CvRatio=1` (common_regbase.h L107-108). `s1CvInner=min(s1Inner·2, s1)`, `s1Outer=ceil(s1/s1CvInner)`,
  `s1Tail / s1CvTail` (=Inner when `%`==0). Same shape for s2.
- **Block split (DoBn2MultiBlkSparse dense else-branch L466-493)**: `fusedOuter=b·n2·g`,
  `blockFactor0=ceil(fusedOuter/aicNum)`, `blockOuter=ceil(fusedOuter/blockFactor0)`,
  `blockFactor=blockFactor0·(s1Outer·s2Outer)`, `fusedOuter·=(s1Outer·s2Outer)`,
  **`maxValidBBLen = blockFactor`** (THE field whose 0-value zeroed the output — drives
  `continuousBlockNum`), `blockStarts[i]=blockFactor·i`, `blockEnds[i]=min(blockFactor·(i+1), fusedOuter)`.
- **coreNum is the AIV count**: kernel does `aicCoreNum = coreNum >> 1` (kernel_base.h L696, MIX 1:2). Set
  `bp.coreNum = 2·usedCore`; the `<<<>>>` blockDim launch uses the AIC count = `usedCore`.
- **DoPostTiling (L1069-1126)** — fp16/bf16 NEED Post to cast fp32-ws → out-dtype (when on the BN2GS1S2/
  Post path): `REGBASE_POST_BASE=128·128`, `AICV_RATIO=2`, `qSize=b·n2·g·s1·d`, `kSize=vSize=b·n2·s2·d`;
  per-tensor `{q,k,v}PostBlockFactor = ceil(ceil(size/baseNum) / (blockOuter·AICV_RATIO))`, `BlockTotal=size`,
  `BaseNum=REGBASE_POST_BASE`, `TailNum = size%baseNum ? : baseNum`.
- **Workspace**: `RESERVED 16MB` sys + fp32 dq/dk/dv partials at `dq/dk/dvWorkSpaceOffset` (512B-aligned).
- **POD population**: `s1s2BNGS1S2BaseParams` (b/n2/g/s1/s2/d/d1, scaleValue, keepProb, s1/s2Token=int32-max
  for dense, layout=3 BNSD, pseType=9 PSE_NONE, sparseType=0, enablePreSfmg=false, sinkOptional=0) +
  `SplitCoreParams` (s1/s2 Outer/Inner/CvInner/Tail/CvTail, blockOuter, maxValidBBLen, noNeedDeter=1) +
  `BlockNumList` (blockStarts/Ends) + `PostParams`. `std::memset(&td, 0, sizeof(td))` first.
- POD GM→local read on device: the CANN `GET_TILING_DATA_WITH_STRUCT` macro is **absent in CANN 9.1.0**;
  replace with the manual uint32 word-blit (`for i in sizeof(POD)/4: dst[i]=src[i]`, the forward
  `wp_fa_entry` pattern).

### Step 6 — the dispatcher TU + pybind + canonical entry-points (port_a3 discipline)
- `flash_attention_score_grad_kernels.cpp`: `#include "kernel_operator.h"` UNGUARDED at top; `#ifndef
  __ASC_NPU_HOST__` wraps the device body (`#include "wholeport/wp_fag_entry.h"` + `FAG_LAUNCH(NAME,IN,
  OUT,S1T,S2T,DT)` → `extern "C" __global__ __aicore__` launchers); `#endif`. **K5 dual-pass** (device
  pass gets MicroAPI via the toolchain's `__NPU_ARCH__==3510`; host pass gets an EMPTY device TU via the
  guard). **FORBIDDEN: `#define __NPU_ARCH__ 3510`** (canon-0/21 dead-end). **NO `#include "arch35/"`**
  anywhere in `*_kernels.cpp` / `pybind11.cpp` (the `ARCH35_WRAP_CHEAT` finalize gate, port_a3-scoped).
- `pybind11.cpp`: host-only (no device `kernel_operator.h`); `#include "wholeport/wp_fag_tiling_data.h"`
  (host-pass-safe), the DoTiling above, `SelectLauncher`, `aclrtlaunch_*` raw launch. Returns dq/dk/dv.
- `model_new_ascendc.py` (OL-160 canonical entry, anti-cheat scanners target this name) — `ModelNew.forward(
  q,k,v,gy)`; a forward-stat prelude obtains `softmax_max/sum + attention_in` via the vendor forward (the
  kernel's required inputs). `model.py` — fp64-autograd backward oracle.

### Build-drift fix-class CHECKLIST (the 14 fixes, ALL read from compiler error / source — never guessed)
A future assembler hitting the first V351 compile WILL meet these; fix each FROM the error, not blind:
1. **namespace**: `using namespace commondef;` MUST precede the block includes (blocks open `namespace
   FagBaseApi` + use UNQUALIFIED `FagRunInfo`/`FagConstInfo`).
2. **include order**: block_vec + block_cube BEFORE kernel.h (block_cube DEFINES the `ARGS_TRAITS` macro
   that kernel_base's class-top `ARGS_TRAITS;` expands into the per-axis `using INPUT_TYPE=...; SPLIT_AXIS=
   ...` fields).
3. **`IS_PSE` is `bool` not `PseTypeEnum`** (`CUBE_BLOCK_TRAITS_CONST_FIELDS` has `X(IS_PSE,bool,false)`).
4. **`using namespace optiling::fag;`** + include `wp_fag_tiling_data.h` before kernel (the POD is used
   unqualified by kernel_base).
5. **`GET_TILING_DATA_WITH_STRUCT`** = CANN-9.0.0 macro ABSENT in 9.1.0 → manual GM→local word-blit.
6. **`std::min` rejected on dav-c310 device** → a local `constexpr` min helper, NOT `__aicore__` (it's a
   `static constexpr` evaluated in BOTH host + device passes, so it must be host-callable).
7. **fp32 DESCOPE**: fp32 dV/dQ/dK take the `MatmulK` L0-split-K path whose `GET_DKV_L0_SPLIT_K<float>`
   yields an invalid `baseK` non-type template arg in the trimmed scaffold → fp16/bf16 use `MatmulFull`
   (non-split), unaffected; fp32 is the dtype-floor anyway → descope (tracked GAP).
8. **undefined launcher symbol**: if a dtype is descoped (its `FAG_LAUNCH` removed), also remove its
   `DECL_LAUNCH` + `SelectLauncher` reference in pybind (else link-time undefined symbol).
9-14 (host-tiling, §step 5): `maxValidBBLen=0`→garbage; `coreNum` is AIV (×2); PostParams=0→all-zero
   output; **splitAxis BN2 vs BN2GS1S2** (§step 4, the decisive precision bug).
(Build env drift, NOT kernel): the build-host's `build_ascendc.py` torch path-probe may `import torch_npu`
which can fail on an ABI-mismatched container → use a LOCAL copy with the probe changed to
`importlib.util.find_spec('torch_npu')` (gets the include PATH without loading `_C`) + `TORCH_DEVICE_
BACKEND_AUTOLOAD=0`.

### Clean V351 build + run RECIPE (for a persistent lane, NOT independent prototype-image-dependent)
- **Build toolchain**: a complete arch35 CANN with MicroAPI + RegTensor + dav_l300 (= dav-c310/V351) +
  `Ascend950PR_9579.ini` platform config + bishengir-compile + `tikcpp/ascendc_kernel_cmake` (e.g. a
  full `cann-9.1.0` install). `ASCEND_HOME_PATH = <cann-root>`; `build_ascendc.py -v Ascend950PR_9579`.
- **Run runtime**: own container from a clean base image (e.g. `a5_base`, torch 2.7.1+cpu) + a torch_npu
  that PAIRS with the device-runtime CANN — for the B060 runtime that is **`.post5` torch_npu** (a `.post1`
  build hit a `free()` allocator-ABI crash; install the `.post5` wheel from an on-host package dir, NO
  independent prototype-image dependency) + `--privileged` + `source <full-CANN>/set_env.sh` (with opp) +
  `ASCEND_RT_VISIBLE_DEVICES=<idle NPU>`. Device-init = `chipType=0` means the CANN runtime doesn't match
  the driver/torch_npu — needs the matched complete install, not a build-only/toolkit-only CANN.

### Precision + perf verification
Reuse the fa_gqa_grad fp64-autograd CDV harness (`model.py` oracle): forward inputs (q,k,v) + gy → fp64
`autograd.grad` over the standard SDPA forward → cast to in-dtype. CDV tol fp16 3e-2 / bf16 5e-2. Compare
mine vs oracle AND vendor `npu_fusion_attention_grad` vs oracle (bit-equivalence = err(mine-ora) ≈
err(vendor-ora)). Determinism ≥5 identical md5. Perf: msprof device-exclusive vs vendor (NOT wall-clock —
OL-201; a per-call fwd-stat prelude inflates wall-clock).

### What is REUSABLE vs op-specific-GAP (the load-bearing honesty)
| Piece | Reusable? |
|---|---|
| The "backward = forward template machinery + different blocks" structural mapping | **REUSABLE** — proven isomorphic by source. |
| The stitchable block list + rename + self-contained-include rewrite (steps 1-2) | **REUSABLE** — mechanical, any FA-grad config. |
| The 3-phase Pre/Base/Post entry pattern + K5 dual-pass dispatcher (steps 3, 6) | **REUSABLE** — structure is config-independent. |
| The build-drift fix-class checklist (the 14) | **REUSABLE as a checklist** — a graybox hitting the first compile will meet these; each is a named, source-resolved fix, not a guess. |
| The splitAxis decision RULE (step 4) | **REUSABLE** — it's the CANN host rule, stated as a formula. |
| The host-tiling field math (step 5) | **REUSABLE-WITH-CARE** — the formulas are the CANN op_host math (cited), but I only VERIFIED the **BN2 / S≤128 / fp16-bf16 / dense / BNSD / D128** instantiation. The DoPostTiling + BN2GS1S2 path are ported-but-NOT-precision-verified (S>128 FAILs the current build). A graybox must port the math faithfully AND verify per new config — do NOT assume the S≤128 values transfer. |
| **GAP — S>128 (BN2GS1S2 axis)** | the host axis-switch + the BN2GS1S2 Process branch + Post are wired but the S=256 case FAILs precision → unresolved; needs its own debug pass. |
| **GAP — fp32** | descoped (MatmulK L0-split `baseK` in the trimmed scaffold). |
| **GAP — mask / pse / dropout / other D-buckets / non-BNSD layouts** | not wired (dense-core scope only). |
| **① autonomous generation — CONFIRMED for the verified slice** | `flash_attention_score_grad` kw-1 (2026-06-20, Ascend950PR_9579/dav-c310) executed steps 1-6 via the ORCHESTRATOR (no human port), built clean FIRST TRY (0 compile-fix iters — the substrate pre-obeys the 14-fix catalog, e.g. the graybox `std::min` fix #6 already applied), then PASSed precision 8/8 (fp16/bf16, S≤128, BN2 dense, BNSD, D128, bit-equivalent to vendor, deterministic). So the ① autonomy claim HOLDS for the BN2 dense slice — the fix-class checklist is a SUFFICIENT pre-load. |
| **② full scope — still GAP** | the autonomous run delivered only the BN2 slice the manual demo had; S>128 + fp32 were NOT reached autonomously (see the S>128 / fp32 GAP rows below — architectural additions, not compile-fixes). |
| **GAP — perf** | only a 0.85× wall-clock proxy measured; msprof device-exclusive + perf-tune not done. |

**Architecture-choice note (which FA-grad path to pick) — applies to ANY dense / small-S FA-grad target.**
Before chasing this fused path's S>128/fp32 GAPs, declare the architecture against the multi-launch sibling.
The decisive, MEASURED axis is PRECISION COVERAGE: for a DENSE / small-S FA-grad op, the proven MULTI-LAUNCH
approach (**CAND-FA-GQA-BWD-1**: AIC-only cube + AIV-only vec, GM-staged) is **precision-core-complete** —
it covers the FULL scope (fp16/bf16/**fp32**, **S>128**) at precision 45/45 (incl fp32 24/25), whereas the
fused single-launch stitch THIS section describes is, as the verified slice, **BN2-only (S≤128, no fp32)**.
That precision-coverage gap is the architecture decider. **PERF is mixed for BOTH and is NOT the
differentiator** (and is a separate tuning lever): the multi-launch's "1.32×" is a MEDIAN dominated by
small-S — its per-shape rows include sub-1× at S=256 (`[2,4,2,256,128]`→0.69×); the fused stitch runs
~0.20× on the small BN2 shapes (vendor 1.5–6µs vs our 13–16µs fixed launch/sync overhead, OL-200/OL-201).
So at the S>128 core shape NEITHER clears 1.0×. **HYPOTHESIS (UNMEASURED — do NOT cite as established):** the
fused single-launch MIX *should* amortize its fixed overhead only when per-block compute is large, so its
plausible value is a large-S perf lever, NOT small-shape coverage — BUT this was NOT measured (the
flash_attention_score_grad run's forward_spec topped at S=512 and the fused path was still overhead-bound
there; a large-S sweep, ~S=2k–8k, is needed to confirm or refute any fused-path perf win). Bottom line
(MEASURED): for dense/small-S, the multi-launch is the precision-core-complete path; the fused stitch's
distinct value is an UNPROVEN large-S hypothesis. (This is the C19 cross-check — evaluate the fused path's
ceiling against the multi-launch sibling, not in isolation; and keep the large-S claim a flagged hypothesis
until a large-S sweep measures it.)

Cross-ref: **CAND-FA-TEMPLATE-GEN-BWD-1** (the candidate entry with the measured result + the full drift
catalog), **CAND-FA-GQA-BWD-1** (the non-template multi-launch FA-bwd MATH reference / oracle),
**CAND-FA-STITCH-1** + **CAND-FA-LAUNCH-DISPATCH-1** (the forward stitch recipe this parallels), **OL-160**
(canonical entry-points), **OL-200** (MIX pipeline), **OL-201** (perf measurement caveat).

## Cross-references

- **P-P101** — de-scalarize flash online-softmax (the vec block-type); `patterns/domains/`
  via PATTERN_INDEX. Precision triad + fp32-contiguous stat.
- **P-P102** — cube-class A5 MIX (the cube block-type + cross-core sync contract);
  `patterns/domains/cube_vector_fusion.md`.
- **OL-200** — MIX cube/vec pipeline stagger (the 4-deep Process pipeline; correct ≠
  perf-normal).
- **OL-202** — optional-input gating (the `hasAttenMask` → sparse-inert fact).
- **OL-201** — kernel≠wall measurement caveat (perf claims must decompose wall, not assume
  kernel-win reaches e2e).
- **CAND-FA-COREDIST-1** (`patterns/unverified/candidates.md`) — the core-split / multi-core
  distribution math (round-robin + symmetric mirror + boustrophedon snake + the
  deal→unflatten→reflatten offset-arithmetic). Index-arithmetic empirically re-derivable from
  the entry alone (kw-wb5 partition-oracle PASS 2026-06-03). Read it for the §Core-split detail
  instead of hand-rolling.
- **CAND-FA-CARRIER-1** (`patterns/unverified/candidates.md`) — AIV↔AIC scalar-params carrier:
  the ≤128 B POD word-blit MECHANISM + host-tiling lifecycle/workspace STRUCTURE + offset
  arithmetic (PIECES A/B/C, generic). PIECE D (the exact carrier byte-layout) is a copy-line —
  sidestepped by authoring the carrier as ONE shared host+kernel header. Read it for §Host-Tiling
  carrier detail. Orthogonal to COREDIST-1 (this SETS the split-mode/prefix fields COREDIST-1 consumes).
