---
applies_to: soc=Ascend910_9382
reason: This template teaches an op-gen worker to GENERATE (not copy) an a3/arch22
  (Ascend910_9382, Atlas A3) cube+vector MIX_AIC_1_2 attention op — kernel AND host.
  The MIX dispatch (NO KERNEL_TASK_TYPE_DEFAULT macro) and the AIC↔AIV cross-core
  handshake are device-proven on Ascend910_9382 (dav-2201, arch22), never on a5.
  It is the a3 COUNTERPART of the a5-only cube_vector_fusion.md (P-P102) /
  fa_class_template.md (P-P103). Do NOT hand this template to an a5/Ascend950PR FA
  worker — a5 has its own native-MIX template and a different (arch35 §4 mode-4)
  sync recipe.
---

# FlashAttention-class MIX Template Domain (a3 / arch22) — GENERATE a cube+vector attention op (kernel + host)

> ## ⚠ ARCH SCOPE — this template is `arch22` / `220x` / Atlas A3 **ONLY**
> `arch_scope: arch22 · 220x · soc=Ascend910_9382 · product=Atlas A3 · dav-2201 · cann=9.0.0` — the greppable restatement of the machine-readable `applies_to: soc=Ascend910_9382` frontmatter line (which is what main's template-scope gate — `kb_index_audit.check_domain_template_scope`, DEBT-222 — reads).
> **NOT for an `arch35` / Ascend950PR / Atlas A5 FA target.** The a5 side has its OWN templates — `cube_vector_fusion.md` (**P-P102**) and `fa_class_template.md` (**P-P103**). Their recipes (arch35 §4 sync mode-4, the `KERNEL_TASK_TYPE_DEFAULT` macro, C310 Fixpipe, MicroAPI regbase softmax) are verified on Ascend950PR and are WRONG for a3.
>
> **P-P116** body (domain file). Index rows: `PATTERN_INDEX.md` + `KB_INDEX.md` (orphan-free invariant). `op_class=attention-fwd / CUBE_MIX (flash-attention family, a3 hand-authored)`.

## What this template is (and how to use it)

This is a **REUSABLE, GENERATION-TEACHING** template, not a copy source. It teaches the *pattern* of an a3 cube+vector MIX attention op — the structure, the load-bearing exact lines, the axes you generalize along, and the host side — so a worker GENERATES its own kernel + host for its own op. It is deliberately the OPPOSITE of a whole-body copy: there is no liftable op body here.

**How a worker uses it:**

1. **READ** this template to learn the MIX pattern + the host-generation guidance.
2. **BUILD + RUN** the minimal a3 MIX SYNC-witness (see §6) once, to *witness the AIC↔AIV handshake close deadlock-free* — proof beats prose before you author.
3. **GENERATE** your op's kernel + host from the knowledge here, **using shipped-library primitives** (see the rule below). Do NOT copy the witness body.

### ★ Reference SHIPPED libraries; generate only the op-glue

The customer's build environment has the CANN library and catlass. **USE them — do not reinvent the wheel:**

| need | USE this SHIPPED primitive (encouraged) | do NOT hand-roll |
|---|---|---|
| cube matmul (`S=Q@Kᵀ`, `O=P@V`) | CANN `MatmulImpl<>` + `IterateAll<sync=true>` (`#include "lib/matmul_intf.h"`) | a hand-written GEMM, or a pure-vector emulation (OL-188, forbidden) |
| row-wise softmax | CANN library `AscendC::SoftMax<T>(...)` + `SoftMaxTiling` / `GetSoftMaxMinTmpSize` (memory-base-vector lib) | our `MixSoftmax` + `RowReduceMax`/`RowReduceSum` wrappers |
| low-level row reduce (if you hand-fold) | library intrinsics `WholeReduceMax` / `WholeReduceSum` (fp32) | a bespoke reduce tree you present as reusable |
| AIC↔AIV handshake | CANN `CrossCoreSetFlag<MODE,PIPE>()` / `CrossCoreWaitFlag()` (catlass `catlass/arch/cross_core_sync.hpp`) | a hand-rolled scratch-flag polling loop |

**BAN (the customer will NOT have these):** CANN *internal* `.cpp`, headers NOT in the shipped CANN library, and **OUR example's hand-written helpers** — `RowReduceMax` / `RowReduceSum` / `Align8` / `FloorPow2` / `MakeMixCfg` / the `MixCubeQK` / `MixCubePV` / `MixSoftmax` wrappers. These are **op-glue you GENERATE yourself** (thin wrappers over the library APIs above), NOT liftable library code.

> **The agent host needs NO CANN.** Compilation happens on the NPU server (which has CANN + catlass). Your generated code freely `#include`s CANN / catlass headers (`lib/matmul_intf.h`, the softmax lib, `catlass/arch/cross_core_sync.hpp`); they resolve at BUILD time on the server. Reference library headers without hesitation.

## verified_on

`verified_on: soc=Ascend910_9382; cann=9.0.0; 2026-07-18; DS famix (single-head FA core) + famix_mh (multi-head)`. Device re-confirmed on the CANN **9.0.0** container: MIX builds + runs deadlock-free + precision-passes (cosine 0.99999905, `allclose` True, deterministic).

---

## 1. Arch scope (restate in your generated op)

- Target `arch22` / `dav-2201`, `SOC_VERSION=Ascend910_9382`, Atlas A3.
- Reusable TEMPLATE, NOT a copy — you generate the op-specific kernel + host.
- This closes the a3 half of **DEBT-222** (the a3 FA-class starting knowledge the KB previously lacked).

## 2. The MIX pattern — PSEUDOCODE (the structure you generate)

One `__global__ __aicore__` entry. The default MIX launch reaches BOTH the AIC and the AIV cores; `if ASCEND_IS_AIC` / `if ASCEND_IS_AIV` partition the work inside it. The two cores rendezvous through two cross-core flags:

```
__global__ mix_attn(q, k, v, o, s_scratch, p_scratch, seq, d, scale):

  if AIC:                                        # cube core
      cube1:  S = Q @ Kᵀ            via library MatmulImpl, IterateAll<sync=true>
      CrossCoreSetFlag(FLAG_S)      # publish S — BROADCAST, releases both AIVs
      CrossCoreWaitFlag(FLAG_P)     # block until AIV has written P
      cube2:  O = P @ V             via library MatmulImpl, IterateAll<sync=true>

  if AIV:                                         # vector core (1:2 paired to the AIC)
      CrossCoreWaitFlag(FLAG_S)     # block until S ready
      P = softmax(scale * S) row-wise, fp32 accumulate   via library SoftMax
      CrossCoreSetFlag(FLAG_P)      # publish P — see §3(a): BOTH subblocks must set
```

- `S`,`P` are `[seq,seq]` scratch tensors in **GM** (allocated host-side, passed in). Softmax works one row at a time in UB.
- The forward flag (`FLAG_S`, AIC→AIV) is **broadcast**; the reverse flag (`FLAG_P`, AIV→AIC) is **per-subblock-counted** — this asymmetry is the whole trap (§3a).

## 3. Minimal NON-NEGOTIABLE snippets ONLY

These exact lines are **load-bearing** — use the CANN/catlass library APIs shown; everything around them you generate. Grounded verbatim in the device-proven `famix` probe / the `examples/a3_mix_fa_min` witness.

**(a) PB-55 — BOTH AIV subblocks set the reverse flag (uniquely correct; single-setter DEADLOCKS).** The reverse AIV→AIC handshake in `MIX_AIC_1_2` is per-subblock-COUNTED: the single AIC `CrossCoreWaitFlag(FLAG_P)` requires a set from EVERY AIV subblock of the 1:2 pair. Both subblocks compute the (identical → benign) softmax and BOTH set the flag:

```cpp
if ASCEND_IS_AIV {
    CrossCoreWaitFlag(FLAG_S);                                  // forward flag is broadcast: one AIC set releases both AIVs
    /* ...softmax P = softmax(scale*S), via library SoftMax... */
    CrossCoreSetFlag<MIX_SYNC_MODE2, PIPE_MTE3>(FLAG_P);        // BOTH subblocks reach here — NO `if (GetSubBlockIdx()==0)` guard
}
```

Raising `FLAG_P` from only subblock 0 → AIC waits forever (measured DEADLOCK, no fault code). `PIPE_MTE3` because the producer of P is the `DataCopy` that drains it to GM.

**(b) `MatmulImpl<>` + `IterateAll<sync=true>` — the genuine (non-async) library cube invocation shape.** Both matmuls use this; NEVER the async KfcServer `Iterate()`/`GetTensor()` path (PB-34 deadlock):

```cpp
MatmulImpl<AT, BT, CT, BiasT, MM_CFG> mm;      // AT/BT/CT = MatmulType<GM, ND, half, /*ISTRANS=*/false>
mm.Init(&tiling, &pipe);
mm.SetTensorA(aG, /*isTransposeA=*/false);
mm.SetTensorB(bG, /*isTransposeB=*/true);      // S=Q@Kᵀ: transpose is a RUNTIME bool on K; the static ISTRANS flag stays false on all three operands
mm.SetSingleShape(M, N, K);
mm.template IterateAll<true>(outG, 0, false, false, false);   // sync=true, no atomic
mm.End();
```

(`MM_CFG` / `tiling` are op-glue you fill from your shapes — generate them, don't lift our `MakeMixCfg`.)

**(c) arch22 dispatch — NO task-type macro.** Do NOT emit `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`. That macro is arch35-only and is REJECTED on Ascend910_9382 → `ACL_ERROR_RT_PARAM_INVALID = 107000` at `RegisterAscendBinary` (PB-28). The default MIX dispatch already reaches both cores:

```cpp
extern "C" __global__ __aicore__ void mix_attn(/* ... */) {
    if ASCEND_IS_AIC { /* cubes + FLAG_S set / FLAG_P wait */ }
    if ASCEND_IS_AIV { /* softmax + FLAG_S wait / FLAG_P set */ }
}
```

> **Macro coupling (load-bearing — the GDR 507014 trap, PB-57).** The "NO task-type macro"
> rule above is correct **for THIS template's path** — a *scoped* `CrossCoreSetFlag`/`WaitFlag`
> MODE2 handshake (§(d) below), which is built around the 1:2 default asymmetry. It does NOT
> generalize to a *whole-device* `SyncAll<false>()` barrier: that path REQUIRES the 1:1 macro
> `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)`, because with the no-macro 1:2 default the
> idle AIV cores skip the per-work-item loop and never issue their `SyncAll` → 507014 deadlock
> (PB-57). So `KERNEL_TYPE_MIX_AIC_1_1` **IS a valid arch22 macro** (PB-28 declares it
> out-of-scope of the 107000 ban); only `MIX_AIC_1_2` / `*_ONLY` are arch35-only. Decision rule:
> **scoped CrossCore MODE2 → no macro (this template); whole-device `SyncAll` → 1:1 macro
> (PB-57 / P-P117). NEVER mix "no macro (1:2)" with whole-device `SyncAll`.**
>
> **⚠️ Generated scoped-1:2 determinism caveat (PB-57 (b')).** The "scoped MODE2 →
> 1:2, no macro" rule above is DEADLOCK-correct, but a generated kernel using
> `MIX_AIC_1_2` plus many per-id `CrossCoreSetFlag<0x2>` handshakes became run-to-run
> nondeterministic above roughly 12 concurrent MIX blocks. For generated FA-class kernels that
> need determinism, select the balanced `KERNEL_TYPE_MIX_AIC_1_1` template. No whole-device
> `SyncAll` is needed: device evidence (N=30) showed all 16 cases deterministic after changing
> only the AIC:AIV pairing balance. Keep the per-id flags and WAR barriers unchanged; see PB-57
> (b') for the full evidence.

**(d) Distinct flag ids, NEVER 0 (PB-35).** `event_t(0)` collides with the cube-internal pipe-sync chain → silent hang. Use the user range 1..7, one id per direction:

```cpp
constexpr uint8_t  MIX_SYNC_MODE2 = 2;   // mode 2 suffices on arch22 (NOT the arch35 §4 mode-4 disjoint-id recipe)
constexpr uint16_t FLAG_S = 4;           // AIC→AIV
constexpr uint16_t FLAG_P = 5;           // AIV→AIC
```

The AIC raises `FLAG_S` with `PIPE_FIX` (its producer is the cube fixpipe): `CrossCoreSetFlag<MIX_SYNC_MODE2, PIPE_FIX>(FLAG_S);`.

## 4. Generalization — the axes you extend along

The proven skeleton is single-head, `seq≤384`, `d=64`, fp16, non-flash, no mask. Your op generalizes along these axes (prose + tiny illustrative pseudocode — you write the full impl):

- **Multi-head / batch** — a device-side loop over `nheads = B*H`, per-head Q/K/V/O offset `bh*seq*d`, reusing ONE `[seq,seq]` S/P scratch pair across all heads. The `FLAG_S`/`FLAG_P` chain serializes access to that shared scratch, so it is safe for any `nheads` (iter *i+1*'s cube#1 overwrites S only after AIV read it; AIV overwrites P only after cube#2(*i*) consumed it):
  ```
  for bh in 0..nheads:  qh=q+bh*seq*d; ...;  # same AIC/AIV chain per head, one reused scratch pair
  ```
- **Causal / `sparse_mode`** — add the mask INSIDE the softmax (before rowmax): set masked positions to `-inf` (or a large negative), then the library `SoftMax` / your reduce handles the rest. Not wired in the proven skeleton.
- **Layouts (BSH / SBH / BSND / BNSD)** — differ only in the index math that computes each head's base offset and the row/col strides you pass to `SetGlobalBuffer` / the matmul shapes. The compute pattern is identical; generate the offset arithmetic per layout.
- **Dtypes (fp16 / bf16)** — I/O dtype is the `MatmulType` `T`; softmax ALWAYS accumulates in **fp32** (Cast in → reduce/exp in fp32 → Cast out with `CAST_RINT`), independent of I/O dtype. PV accumulate stays fp32 in the cube.
- **Shapes** — single-pass **non-flash** softmax is correct while a full `[seq]` fp32 row fits UB (fine at moderate seq, since S lives in GM and only one row is UB-resident). For very long seq, switch to **flash online-softmax + KV-block tiling** (running-max / running-sum / expMax rescale) — codify that delta from **P-P101**, not here.

## 5. HOST-side generation guidance (generate this too, with standard CANN APIs)

Author the host with **standard CANN / torch_npu APIs** — no exotic wiring. Pattern, not a copyable file:

- **Tiling struct** — the op-specific scalar args the kernel needs (`seq`, `d`, `scale`, plus `nheads` / layout strides / mask flags for the general op). For the `IterateAll<sync=true>` cube path, the `TCubeTiling` is filled from device-side scalar args (`M/N/Ka/Kb`, `singleCoreM/N/K`) — so the host does NOT need the matmul host-tiling lib on its link line (OL-235). If you use the library `SoftMax`, generate its `SoftMaxTiling` (via `GetSoftMaxMinTmpSize`/`GetSoftMaxMaxTmpSize` for the tmp-buffer size).
- **Scratch allocation** — allocate the `[seq,seq]` (or `[nheads,seq,seq]`, or one reused `[seq,seq]`) S / P GM scratch tensors host-side and pass their pointers in (they are NOT op outputs).
- **Launch** — the standard `aclrtlaunch_<kernel>(blockDim, stream, ...ptrs..., scalars)`. `blockDim = 1` (one AIC block; the runtime pairs the AIV(s) for the 1:2 MIX). **The standard `aclrtlaunch` runtime already supplies the FFTS descriptor** (via `rtGetC2cCtrlAddr`) — do **NOT** hand-wire FFTS; that is what makes the user `CrossCore*` handshake work under the default launch.
- **pybind marshaling** — keep the pybind shim PURE: device/dtype/dim `TORCH_CHECK`s, `.contiguous()`, `torch::empty` for output + scratch, compute `scale = 1/√d`, call the `extern "C" aclrtlaunch_<kernel>` shim, return the output. NO torch math in the shim (all compute is in the kernel).
- **Build** — `SOC_VERSION=Ascend910_9382`, the standard AscendC `aclrtlaunch` flow (separate AIC+AIV device objects; pybind links the auto-generated `extern "C" aclrtlaunch_<kernel>`). Compiles on the NPU server; the generated code `#include`s CANN/catlass headers resolved there.

## 6. Witness step — BUILD + RUN the SYNC-witness BEFORE you author

Before authoring your kernel, build and run the minimal a3 MIX SYNC-witness at:

`src/skills/references/target/ascendc/examples/a3_mix_fa_min/`

(main is slimming this into a handshake-only witness — **main owns its final form**; reference it by that path.) Build it on an a3 CANN 9.0.0 container and run it once: the point is to watch `torch.npu.synchronize()` **return** (not hang) — that is the device proof the AIC↔AIV handshake closes deadlock-free, which beats any prose assurance. Do NOT author or fork the witness (main's lane) and do NOT construct your own; READ + RUN it, then GENERATE your op.

## 7. Cross-references

- **PB-55** — the `MIX_AIC_1_2` AIC↔AIV handshake is direction-ASYMMETRIC (reverse per-subblock-counted; single-setter deadlocks); §3(a) implements it.
- **PB-34** — `MatmulImpl<>` + async KfcServer path deadlocks in this MIX mode → §3(b) mandates `IterateAll<sync=true>`.
- **PB-35** — `event_t(0)` collides with the cube-internal pipe-sync chain → §3(d) forbids flag id 0.
- **DEBT-222** — the a3 FA-class domain-template gap; this file closes its a3 half.
- **PB-28** — `KERNEL_TASK_TYPE_DEFAULT` is arch35-only (`107000` on arch22) → §3(c) omits it.
- **P-P101** — de-scalarized online-softmax; the delta to add for flash KV-block tiling on long seq (§4).
- **P-P102 / P-P103** — the a5/arch35 native-MIX counterparts; disjoint arch scope (this a3 template is their sibling, not a substitute).
- **OL-188** — pure-VEC for a cube-required op = forbidden architectural hack (§ library-rule).
- **OL-235** — the harness pybind link line has no host matmul-tiling lib → `IterateAll<sync=true>` is the buildable cube path (§5).

## 8. HONEST SCOPE

**Device-proven** (`Ascend910_9382`, CANN 9.0.0, 2026-07-18): the MIX **pattern** — dispatch, the 2-flag AIC↔AIV handshake, genuine library-cube matmuls, fp32 softmax — at seq ≤ 384, d = 64, fp16 I/O, single-head + multi-head (device loop, one reused scratch pair). cosine 0.99999905, deterministic.

**NOT proven (do not claim as covered):** flash online-softmax + KV-block tiling (long seq — use P-P101); causal / attention mask; **perf** (single-pass non-flash, not tuned — but a3 **DOES** have a vendor baseline: `npu_fusion_attention` runs ~102µs, and `cross_core_sync.md` §5 / PB-56 measure this template's single-core at 0.0146× and a per-head-independent multi-core extension at 0.186× vendor. **Corrects the earlier "a3 has no vendor FA baseline" claim — it does; perf is measurable, just uncompetitive until a flash rework**); **no claim on a5** (`unverified_on: soc=Ascend950PR` — a5 uses P-P102 / P-P103).

Each op generated from this template is verified SEPARATELY on device (owner-scheduled); this template's existence enables generation, it does not certify any specific generated op.
