---
applies_to: soc=Ascend950PR
reason: This template teaches building the A5/V351 (Ascend950PR) MIX kernel — its
  own body scopes itself so: "Cube-class A5 ports use native MIX" (Principle), inline
  "applies_to: ...; A5/V351", and "verified_on: flash_attention_score A5". The 220x/a3
  side is only the SOURCE algorithm being ported FROM; the device code + cross-core
  handshake this template stitches is A5. Delivered via the CUBE_MIX-tagged port_a3
  Layer-2 block (_port_a3_cube_class_mix_block), whose target is always a5.
---

# Cube-Vector Fusion (MIX) — port_a3 cube-class canonical pattern

> **P-P102** body. Index row: `PATTERN_INDEX.md`. KB_INDEX row present (orphan-free invariant).
>
> Design source: `docs/design/PORT_A3_CUBE_CLASS_MIX_ENFORCEMENT_DESIGN.md` §3.4.
> Purpose: so the NEXT cube-class port_a3 op starts from knowledge, not from
> scratch (amortize the DEBT-110 "8 spawns for FA" cost). This is the
> worked-example a CUBE_MIX-tagged worker is pointed at (via the Layer-2 brief
> `_port_a3_cube_class_mix_block`), **reached through KB — never by reading
> another op's `output/` archive** (own-dir-only guard, commit `b96508ec`).

## Principle (the WHAT)

**Cube-class A5 ports use native MIX (Mmad cube + vector epilogue + WorkspaceQueue
cross-core sync), never pure-vec.** A pure-VEC kernel generated for a cube-required
op is a HACK (OL-188, same anti-cheat tier as CPU fallback) — the finalize gate
`_check_architecture_class` rejects it as `ARCHITECTURAL_HACK` (PR #316).

- **applies_to**: `port_a3_to_a5; op_class CUBE_MIX (matmul / attention / conv / rnn /
  gmm / ffn families); A5/V351`.
- **verified_on**: `flash_attention_score A5` (`output/a3_to_a5_port/src/kernels/
  flash_attention_score/op_kernel/`, decision_manifest.jsonl 2026-05-29). The concrete
  anchors below are distilled from that verified archive so a worker reads them HERE,
  not from the archive.

## Concrete anchor (the worked example, distilled)

- **File split**: `<op>_cube.h` (cube class) + `<op>_vec.h` (vec class) + `<op>_kernel.h`/
  `.cpp` orchestrator. Class names contain the literal `Cube`/`Vec` (the finalize
  pre-build regex + arch-class classifier look for cube markers in these files).
- **Cube primitive**: manual `AscendC::Mmad`. FA used 9 `Mmad` calls;
  `MmadParams.cmatrixInitVal=true` for a single K-tile, `cmatrixSource=false`.
  **EXECUTABLE params (this anchor is contract-level; the code-level detail lives in
  [`../../fa_class/cv_reference_concrete_params.md`](../../fa_class/cv_reference_concrete_params.md)):**
  the `Mmad` signature, full `MmadParams` with `cmatrixInitVal=(ki==0)` for the K-tile accumulate
  loop (not always-true — that's the single-tile case only), the L0A/L0B Nz-fractal layout (16×C0,
  C0=16 fp16), the QK-load `ifTranspose` form, the Fixpipe `srcStride` PER-CONTEXT units (`/C0` for
  L0C→workspace vs ELEMENTS for L0C→GM), and the 16/C0 block-tile-alignment rule. An agent writing
  the cube needs that file — this anchor stops short of the code.
  `matmul::Matmul<>` / `MatmulImpl<>` is the library alternative, but FA chose `Mmad`
  after V220 evidence that `matmul::Matmul<>` was numerically wrong ~500× — the A5
  path picks the safer verified `Mmad`. (Cross-ref: the standalone `matmul::Matmul<>`/
  KFC path is separately blocked on V220 — CAND-KFC-standalone-bootstrap-teardown,
  rendezvous deadlock — so `Mmad` is also the unblocked choice.)
- **Vec epilogue**: the op's vector reduction/activation. FA: `SoftmaxFlashV2` (online
  softmax). General cube-class epilogue = the op's reduction/elementwise tail.
- **Cross-core sync**: `WorkspaceQueue<T, RING_SLOTS=3>` ring buffer, **ONE per
  producer↔consumer direction** (FA: 3 — S cube→vec, P vec→cube, O cube→vec), with
  paired flag IDs and **raw `PIPE_FIX`/`PIPE_MTE3`/`PIPE_MTE2` literals** (CANN 9.0.0
  forbids templated `pipe_t`). **NOT** file-scope inline `CrossCoreSetFlag` in the loop
  (Antipattern A → 507015). (Cross-ref OL-190 WorkspaceQueue.)
- **Task type**: `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` — V220-native
  (PB-28 FALSIFIED 2026-05-25). Do NOT arch-guard it out.
- **Tiling**: regbase per-AIV sub-block (`vec_num=2` for MIX_AIC_1_2 attention), under
  the V351 248 KiB UB budget.

## The WHY (so a worker can ADAPT, not copy)

Per the study-CANN-understand-WHY direction: the MIX structure exists because cube
(AIC) and vector (AIV) are **separate cores** that must be kept both busy. The cross-
core sync is a **counting-semaphore** handshake (`WorkspaceQueue` flag IDs) pairing one
AIC producer with the AIV consumer per data direction; getting the AIC/AIV pairing and
the producer-PIPE↔consumer-PIPE matching right is what avoids both deadlock (507014-
class rendezvous) and the lockstep pipeline-collapse (OL-200: cube/vec each ~half idle
when scheduled in lockstep instead of `i%K`-staggered). A worker adapting this to a new
cube op must reproduce the *contract* (one queue per direction, correct PIPE pairing,
staggered schedule), not the FA-specific tile sizes.

## Generation methodology — understand→KB→research→regenerate

(NOT line-port V220, NOT pure-vec fallback. Codified in the Layer-2 brief + here.)

1. **Understand**: read the A3/V220 algorithm source (`op_kernel/*.h` top-level +
   `op_host/*.h`) for the algorithm — what cube op, what epilogue, what dataflow.
2. **KB**: consult this pattern + L-tier RegBase MicroAPI KB (OL-143/144) for the A5
   mapping.
3. **Research (only if you cannot generate)**: read the A5 reference for UNDERSTANDING
   — allowed per the study-CANN direction; the `ARCH35_WRAP_CHEAT` gate + copy-shape
   scanners remain the anti-copy net (understand, never copy verbatim into the kernel).
4. **Generate from understanding** using the scaffold above.
5. **Regenerate** on gate rejection / precision-perf gap by looping back to
   understand/KB/research — **NEVER** to a pure-VEC fallback (the gate rejects it).

## Anti-pattern cross-refs

- **OL-188** — pure-VEC for a cube-required op = HACK (the finalize gate this pattern
  satisfies).
- **OL-190** — WorkspaceQueue ring-buffer cross-core sync.
- **OL-200** — MIX pipeline collapse: lockstep cube/vec leaves each ~half idle; stagger
  by `i%K` (precision-correct ≠ perf-correct).
- **EC-57 / EC-58** — per-source `SPLIT_CORE_CUBE`/`SPLIT_CORE_VEC` defines +
  `matmul_intf.h` include discipline for the MIX compile.
- **PB-35** — `event_t(0)` cross-Path deadlock.
- **PB-28 FALSIFIED** — `KERNEL_TYPE_MIX_AIC_1_2` is V220-native; do not arch-guard out.
