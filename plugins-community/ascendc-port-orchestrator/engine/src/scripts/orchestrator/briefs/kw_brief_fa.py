# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""FA-class (attention) brief builders for aog-kernel-worker.

Extracted verbatim from kw_brief.py (DEBT-201 god-file decomposition,
2026-07-06). This is the cohesive FA-class cluster: attention/backward
predicates + the arch35 template-assembly recipe blocks + the P-P103 backward
stitch / multi-launch recipes + the GE op_host generation block.

Depends only on `_common` and `plugins.base` (imported inline where used) — a
LEAF module. The parent `kw_brief` re-imports the four externally-referenced
symbols (`_is_fa_class_backward`, `_fused_fa_backward_requested`,
`_fa_class_backward_stitch_block`, `_fa_class_backward_multilaunch_block`) so
`from briefs.kw_brief import ...` keeps working for the BackwardPlugin.

Behavior is BYTE-IDENTICAL to the pre-split functions (prompt-template refactor;
golden-locked by the FA-cluster goldens + existing FA emit tests).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from briefs.kb_scope import (
    kb_entry_applies_to_target,
    kb_file_applies_to_target,
    kb_section_applies_to_target,
)


def _is_fa_class_backward(op: str, op_class: str, workspace: Optional[Path]) -> bool:
    """True iff the op is an FA-class (attention) op AND a backward (gradient) op.

    Used by both the kw_brief forward-block short-circuit guard and the
    BackwardPlugin override to route an FA-grad op to the P-P103 BACKWARD
    stitch recipe (NOT the forward template-assembly, NOT the analytic-derive
    backward). Belt-and-suspenders on BOTH axes: FA = name OR tag; backward =
    GRADIENT/BACKWARD tag OR *_grad/_backward/_bwd name OR opgen_mode==backward.
    """
    try:
        from plugins.base import is_attention_named as _is_fa_named
        from plugins.base import is_fa_class as _is_fa_tag
        from plugins.base import is_backward_class as _is_bw_tag
    except Exception:
        return False
    if not (_is_fa_named(op) or _is_fa_tag(op_class)):
        return False
    name_l = (op or "").lower()
    is_bw_named = (
        name_l.endswith("_grad") or name_l.endswith("_backward")
        or name_l.endswith("_bwd") or "_grad_" in name_l
    )
    is_bw_mode = False
    if workspace is not None:
        try:
            import json as _json
            st_p = workspace / ".opgen_state.json"
            if st_p.is_file():
                st = _json.loads(st_p.read_text())
                is_bw_mode = (st.get("opgen_mode") == "backward")
        except Exception:
            is_bw_mode = False
    return bool(_is_bw_tag(op_class) or is_bw_named or is_bw_mode)


def _fused_fa_backward_requested(op: str, workspace: Optional[Path]) -> bool:
    """True iff the FUSED single-launch FA-grad stitch is the REQUESTED architecture.

    Architecture-default decision (2026-06-20, owner via the C19 finding): for a
    DENSE / small-S FA-grad op the DEFAULT is the proven MULTI-LAUNCH approach
    (CAND-FA-GQA-BWD-1 — precision-core-complete: 45/45 incl S>128 + fp32), NOT the
    fused single-launch stitch (which is BN2-only + ~0.2× on small shapes; its large-S
    perf value is an UNMEASURED hypothesis). So the fused-stitch recipe is gated behind
    an EXPLICIT opt-in, never the default. Opt-in signals (any one):
      - `.opgen_state.json` `fa_backward_arch == "fused"` (explicit caller choice), OR
      - `.opgen_state.json` `fa_backward_large_s` truthy (the regime where the fused
        MIX *may* amortize its fixed overhead — the only place it's plausibly worth it;
        still hypothesis-flagged in ③ until a large-S sweep measures it).
    Default (no signal) → False → the kw follows the multi-launch default brief.
    """
    if workspace is None:
        return False
    try:
        import json as _json
        st_p = workspace / ".opgen_state.json"
        if not st_p.is_file():
            return False
        st = _json.loads(st_p.read_text())
    except Exception:
        return False
    if str(st.get("fa_backward_arch", "")).lower() == "fused":
        return True
    return bool(st.get("fa_backward_large_s"))


def _fa_class_template_assembly_block(
    op: str, workspace: Path, target: str = "a5"
) -> Optional[str]:
    """If op is FA-class (attention-named OR FA-tagged), return the
    **template-assembly recipe** brief: the FA-class kw worker assembles a
    self-contained arch35 FA operator (host + kernel) from the arch22 algorithm
    spec + the codified KB templates (P-P103). Else None.

    `target` (DEBT-208) selects which KB cards the MIX cross-core-sync block
    carries, by each card's own `applies_to: soc=`. Defaults to `a5` to match the
    `kb_manifest_block(..., target="a5")` convention; `build_worker_brief` passes
    `env.target`.

    The FA paradigm is template-assembly. FA-class ops route to the standard
    kw worker, which does
    the assembly directly from the arch22 spec + KB templates (no separate
    design stage).

    Source-mode-agnostic (FA is op-class, not mode): fires for port_a3 and
    backward when the op is attention-named or FA-tagged.

    Input-provenance (§15.2): arch22 source-NPU capture remains the migration
    truth. A target archive, verified prestage manifest, or prior-art scan may
    be consulted only as logged advisory context and may never replace fresh
    source capture or target-NPU verification. Content source =
    `FA_KW_BRIEF_TEMPLATE_ASSEMBLY_CONTENT.md` (P-P103 §9/§14/§15).
    """
    op_cls_path = workspace / "op_classification.json"
    if not op_cls_path.is_file():
        return None
    try:
        import json as _json
        tags = _json.loads(op_cls_path.read_text()).get("op_class_tags") or []
    except Exception:
        return None
    op_class = " ".join(tags) if isinstance(tags, list) else str(tags)
    # task#33 / task#31: gate on the NAME-based predicate (is_attention_named),
    # SAME as the routing gate — the tag-based is_fa_class (FUSED+SOFTMAX)
    # mis-fires on pure-vector fused ops with a softmax (hc_split_sinkhorn).
    # Name backstop OR the narrowed structural ATTENTION tag; a Sinkhorn-shaped
    # op matches NEITHER → authors normally.
    try:
        from plugins.base import is_attention_named as _is_fa_named
        from plugins.base import is_fa_class as _is_fa_tag
    except Exception:
        return None
    if not (_is_fa_named(op) or _is_fa_tag(op_class)):
        return None
    # FA-class BACKWARD route (2026-06-20): when the FA-class op is a *backward*
    # (gradient) op, the FORWARD template-assembly recipe below is the WRONG
    # recipe (forward config-space, wp_fa_regbase_impl forward entry, 2-GEMM
    # online-softmax phase-order). RETURN None here so this forward block does NOT
    # short-circuit — the BackwardPlugin.kw_brief_phase_block override (which
    # carries the B3.3b verify/finalize SCHEMA CONTRACT + the cannbot precision
    # judge) runs and PREPENDS the FA-grad ARCHITECTURE brief. The architecture is
    # chosen there (C19 finding, 2026-06-20): DEFAULT = the proven MULTI-LAUNCH
    # approach (`_fa_class_backward_multilaunch_block` → CAND-FA-GQA-BWD-1,
    # precision-core-complete incl S>128+fp32); the fused single-launch stitch
    # (`_fa_class_backward_stitch_block` → P-P103 BACKWARD) is GATED behind an
    # explicit opt-in (`_fused_fa_backward_requested`: fa_backward_arch=="fused" /
    # fa_backward_large_s), never the default. Keeping the GENERATION recipe + the
    # VERIFY/finalize contract composed in ONE place (the plugin) avoids losing
    # the schema contract that a bare short-circuit here would drop.
    # Backward signal: GRADIENT/BACKWARD tag OR a *_grad/_backward/_bwd op name OR
    # opgen_mode==backward in .opgen_state.json — any one suffices (belt-and-
    # suspenders, same robustness principle as the name+tag FA gate above).
    if _is_fa_class_backward(op, op_class, workspace):
        return None
    return (
        _fa_assembly_intro_block(op, op_class)
        + _fa_assembly_recipe_block()
        + _fa_assembly_deadlock_warning_block(target)
        + _fa_assembly_compile_block()
        + _fa_assembly_verify_hard_block()
        + _fa_ge_host_gen_block()
        + "Reference: `kb/target/ascendc/patterns/domains/fa_class_template.md`\n"
        "(P-P103) + design doc §9/§14/§15."
    )


def _fa_assembly_intro_block(op: str, op_class: str) -> str:
    """Header + input-provenance prose for the forward FA assembly brief.

    Extracted from `_fa_class_template_assembly_block` (DEBT-164 god-fn split);
    returns the exact same leading string the composed brief used to inline.
    """
    return (
        "# PHASES (FA-class template-assembly — owner 2026-06-07 paradigm)\n"
        "\n"
        f"op `{op}` is FA-class (attention-named / FA-tagged: {op_class!r}). Paradigm =\n"
        "**template-assembly**. Assemble a self-contained\n"
        "arch35 FA operator (host + kernel) from the arch22 algorithm spec + the codified\n"
        "KB templates (P-P103). Do NOT author from scratch; do NOT use any IL.\n"
        "\n"
        "## INPUT AND PROVENANCE (§15.1/§15.2):\n"
        "- **arch22 FA source** (algorithm spec being ported): `op_kernel/arch22/*.h`\n"
        "  (`flash_attention_score_s1s2_bn2gs1.h` main + variants + `_common.h` /\n"
        "  `_tiling.h` / `_template_tiling_key.h`) + `op_host/arch22/` (host-tiling) +\n"
        "  the op proto (`flash_attention_score_def.cpp`) + the test config\n"
        "  (dtype/shape/head_num/layout). [The arch22 source is provided/bind-mounted\n"
        "  in graybox mode — read it where it is staged; in port_a3 it is the upstream\n"
        "  arch22 source and remains the migration truth.]\n"
        "- **codified KB**: P-P103 `patterns/domains/fa_class_template.md` (skeleton +\n"
        "  FA+X delta table + block inventory + host-tiling logic) + design doc §9/§14.\n"
        "\n"
        "A target archive, `.prior_art_scan.json`, DEBT203 branch base, or SHA-verified\n"
        "`.upstream_prestaged.json` entry may be read only as provenance-logged, read-only\n"
        "advisory context. Never copy an untracked target file or treat advisory context as\n"
        "truth. Reconstruct from arch22 semantics + KB, then reverify with a fresh arch22\n"
        "source-NPU capture and the generated arch35 operator on target NPU.\n"
        "\n"
    )


def _fa_assembly_recipe_block() -> str:
    """The §Recipe assembly steps (config-space → per-combo → phase-order).

    Extracted from `_fa_class_template_assembly_block` (DEBT-164 god-fn split).
    """
    return (
        "## RECIPE (assemble — P-P103 §Recipe / §14.3):\n"
        "1. **Enumerate the FULL declared config-space from arch22 `template_tiling_key.h`**\n"
        "   (the `ASCENDC_TPL_UINT_DECL` block = the op's COMPLETE capability surface, NOT\n"
        "   the test cases): DataType {fp16,bf16,fp32,fp8×3 per op proto}, Layout\n"
        "   {BSND,SBH,BSH,BNSD,TND}, HasDropOut{0,1}, HasAttenMask{0,1}, HasPse{0,1},\n"
        "   Sparse{0..9}, ImplMode{0,1,2}, S1/S2/DTemplateType (D up to 768). Respect the\n"
        "   `ASCENDC_TPL_SEL` validity rules. **THIS — not the test config — defines WHAT to\n"
        "   assemble.** The test fixtures are for VERIFICATION ONLY; assembling only what the\n"
        "   tests show = the 49/64 extraction-failure (2026-06-08, A5≢A3). Assemble the FULL\n"
        "   SEL-valid space so the A5 op is **functionally equivalent to the A3 source**. Any\n"
        "   declared combo that fails to assemble/build/verify → **DEBUG-ON-TOP FIRST, do NOT\n"
        "   fast-stop**: you ARE a debugging worker, not template-only. Attempt to extend/fix the\n"
        "   template+host from arch22 + KB (e.g. wire `keepProb` + select the dropout kernel path\n"
        "   for dropout; adjust tiling / register-allocation for larger D; fix the build/precision\n"
        "   error using KB knowledge — the same Edit/Build/Verify loop you use for any op). Record\n"
        "   an **explicit evidence-backed GAP in analysis.md ONLY if the fix is genuinely\n"
        "   un-derivable from arch22 + KB after a real debug attempt** (a fundamentally-new\n"
        "   algorithm in NO source). Fast-stop/GAP is the LAST resort, not the first response to a\n"
        "   no-exact-template case. **NEVER silently drop a declared combo** (silent-drop IS the\n"
        "   A5≢A3 bug). Ground-truth list: `docs/design/FA_CLASS_DESIGN_NOTES.md#fa-a3-source-param-space`.\n"
        "   **SOURCE-AVAILABILITY = check BOTH arch22 AND the KB templates** before declaring\n"
        "   a combo no-source: a combo absent from arch22 may still be in the KB block-templates\n"
        "   (`fa_class/templates/op_kernel/wholeport/`). E.g. fp8/mxfp8 has 0 arch22 hits BUT the\n"
        "   KB has `vf_basic_block_fullquant_*`(fp8) + `*_mx.h`(mxfp8) + large-D blocks → it is\n"
        "   **KB-sourceable, NOT no-source** (the 2026-06-08 v2 GAP-1 wrongly grepped only arch22).\n"
        "   Declare `no-source` ONLY if NEITHER arch22 NOR the KB templates carry it.\n"
        "   **USER-NL EXTENSIONS**: if a structured param-space-extension is provided (e.g.\n"
        "   `{head_dim:[256,512,768], dtype:[fp8_e5m2,...]}` from the current entry request),\n"
        "   merge it into the target space — assemble those too (sourced from arch22 ∪ KB).\n"
        "   THEN per resolved combo:\n"
        "1b. **Extract per-combo params from arch22**: dtype→{INPUT_T,T,OUTPUT_T}; layout;\n"
        "   B/N/N2(KV-head)/G/S1/S2/D/DV; sparse_mode; features (hasPse/hasAtten/hasDrop);\n"
        "   scaleValue/keepProb. Source = op proto + arch22 structure (NOT test config).\n"
        "2. **Select kernel template + fill params**: instantiate\n"
        "   `wp_fa_regbase_impl<INPUT_T,T,OUTPUT_T, implMode, layOutType, S1Template,\n"
        "   S2Template, DTemplate, DVTemplate, pseType, hasAtten, hasDrop, hasRope>`.\n"
        "   dtype → {INPUT_T,T,OUTPUT_T} (fp16→<half,float,half>; bf16→<bfloat16_t,\n"
        "   float,bfloat16_t>; fp32→<float,float,float>); D → AlignUp(D,64) → DTemplate\n"
        "   bucket (≤64 Aligned64 / ≤128 Aligned128 / ≤256 Aligned256 / >256 Aligned768);\n"
        "   features → {pseType,hasAtten,hasDrop}; S1Template = Aligned128 (core-fill\n"
        "   → Aligned64).\n"
        "3. **Fill host-tiling**: dBasicBlock=AlignUp(D,64); s1BasicBlock per\n"
        "   CalcS1S2BasicBlock (fp32→64 else 128; core-fill totalSize<28 & dBasicBlock\n"
        "   ≤256 → 64); sparseType (dense=ALL / causal=CAUSAL); totalSize=B·N2·G·\n"
        "   s1OuterSize; coreNum=min(totalSize,28); splitCoreMode (s2≥threshold→1).\n"
        "   Fill td (InputParamsRegbase + MultiCoreParamsRegbase); workspace = 16MB +\n"
        "   (bmm2+vec2)·3·coreNum.\n"
        "   **Workspace + tiling = use the COMMON reusable cache** `wp_fa_host_cache::get(&td,\n"
        "   sizeof(td), totalWsBytes, dev)` (include `op_host/wp_fa_host_cache.h`) → returns\n"
        "   {workspace_ptr, tiling_ptr}. **Do NOT per-call `torch::empty(workspace)` + tiling\n"
        "   memcpy+H2D** — that's the OL-201/DEBT-147 per-call host overhead (custom-launch has no\n"
        "   framework pre-alloc). The cache persists workspace + caches device-tiling keyed by\n"
        "   (tiling-POD,ws-size). WHITEBOX-VERIFIED 2026-06-08: wall −36..−73%, precision 40/40\n"
        "   within-T1 unchanged. Outputs (attention_out/softmax_*) stay per-call (results).\n"
        "4. **Select launcher**: `wp_fa_do_<dtype>_bnsd[_<feature>]_d<bucket>` (core-fill\n"
        "   fp16-d128 → `_s64`).\n"
        "5. **Assemble self-contained op**: kernel TU = instantiate\n"
        "   `wp_fa_regbase_impl<args>` entry + `<<<>>>` launcher; host = DoTiling per\n"
        "   config + dispatch + output alloc; **reuse block templates** (P-P103:\n"
        "   wp_block_cube / wp_block_vec_base / wp_attenmask / wp_pse / wp_dropmask /\n"
        "   wp_mc_* / wp_fixpipe_out / wp_fa_entry) via include. **Self-contained**: NO\n"
        "   `#include \"arch35/\"`; canonical entry-points (`model_new_ascendc.py` / `model.py`).\n"
        "6. **Phase-order** (kernel, §9): IterateBmm1(S=Q·Kᵀ) → GetBmm1Result →\n"
        "   CopyInAttenMask → Muls(scaleValue) → PseCompute(if hasPse) → ComputeAttenMask\n"
        "   → SoftMaxCompute → IterateBmm2(O=P·V) → Bmm2ResultMul/Div(online rescale) →\n"
        "   Bmm2DataCopyOut + SoftmaxDataCopyOut; cube:vec = 1:2 MIX.\n"
        "\n"
    )


def _fa_assembly_deadlock_warning_block(target: str = "a5") -> str:
    """MIX cube+vec (`KERNEL_TYPE_MIX_AIC_1_2`) cross-core-sync brief for the
    FORWARD FA-class template-assembly path, SCOPED BY EACH KB ENTRY'S OWN
    `applies_to: soc=` (DEBT-208, 2026-07-17).

    Why this block exists (PB-34 recall-fix, 2026-07-16): the forward FA-assembly
    recipe hands the worker a 1:2 MIX cube:vec template (recipe step 6, "cube:vec
    = 1:2 MIX") but carried ZERO deadlock warning, while the BACKWARD briefs
    (`_fa_class_backward_stitch_block` / `_fa_class_backward_multilaunch_block`)
    DO cite the PB-34/35 cross-core-sync risk. A forward worker targeting V220
    had no steer away from the MatmulImpl + manual-CrossCore FFTS-slot deadlock
    and could walk into a silent first-launch hang for days.

    Why it is now SoC-scoped (DEBT-208): the recall-fix composed the PB-34 warning
    UNCONDITIONALLY, with the scope carried only as prose inside the injected text
    ("V351/A5 scope bound — do NOT over-apply") — LLM compliance, not structure.
    On A5 that INVERTED the KB: PB-34 is `applies_to: soc=Ascend910_9382` (V220)
    with two `verified_does_not_reproduce_on: Ascend950PR` witnesses (micro-probe
    2026-05-23; GDN full-op light-port 2026-06-15, 122/122 T1 PASS), and its own
    Consequence line makes the light-port the DEFAULT A5 route — exactly what the
    injected warning steered A5 workers away from (an FA worker retreated to
    vector-only). Each sub-block below is now emitted iff the KB entry it carries
    declares a scope covering `target`, read from the entry itself via
    `briefs.kb_scope` (which reuses `kb_index_audit`'s SoC parser). Adding a SoC
    bound to a KB entry now bounds this composer with no composer edit.

    The A5 and V220 halves are NOT variants of one warning — they are different
    knowledge. A5 gets the two mutually-exclusive proven routes (Path B / Path A);
    V220 gets the deadlock that is confirmed there. PB-35 and the DEBT-210
    compile-path gap reach BOTH (PB-35's `applies_to` names both SoCs; DEBT-210 is
    a harness gap with no SoC scope → fail-open).

    Args:
        target: build target (`a5` / `a3` / `a2`), i.e. `env.target`. Defaults to
            `a5` to match the `kb_manifest_block(..., target="a5")` convention.
    """
    parts = [_fa_mix_sync_intro_block(target)]
    # A5 recipe — anchored on the two entries that carry it: OL-220 (the light-port
    # build recipe = Path B) and cross_core_sync.md §4 (the runnable handshake =
    # Path A). Both are `soc=Ascend950PR`, so this is the A5 route selector.
    if _mix_sync_a5_recipe_applies(target):
        parts.append(_fa_mix_two_paths_block())
    # a3 positive recipe (DEBT-222 delivery): the device-proven a3 FA-class STARTING
    # SKELETON P-P116 + the PB-55 reverse-handshake rule. The a3 mirror of the a5
    # two-paths block above — where an a5 worker gets its proven MIX routes, an a3
    # worker gets P-P116. Gated on `fa_class_a3_mix_template.md`'s own header
    # `applies_to: soc=Ascend910_9382` (a3/a2 → 220x); `unverified_on: soc=Ascend950PR`,
    # so it is suppressed on a5 and widens NO a5 scope (DEBT-208 discipline).
    if kb_file_applies_to_target(_A3_MIX_TEMPLATE_MD, target):
        parts.append(_fa_a3_mix_skeleton_block())
    if kb_entry_applies_to_target("PB-34", target):
        parts.append(_fa_mix_pb34_v220_block())
    if kb_entry_applies_to_target("OL-275", target):
        parts.append(_fa_mix_v220_shipped_cube_block())
    if kb_entry_applies_to_target("PB-35", target):
        parts.append(_fa_mix_pb35_pattern_a_block())
    if kb_entry_applies_to_target("PB-45", target):
        parts.append(_fa_mix_pb45_library_cube_block())
    parts.append(_fa_mix_debt210_sync_base_block())
    parts.append(_fa_mix_fix_cards_block(target))
    return "".join(parts)


# `cross_core_sync.md` §4 — the RUNNABLE A5 handshake. Path A's anchor; its
# `applies_to: soc=Ascend950PR (V351 / A5, Ascend950PR_9579)` is what scopes the
# A5 recipe below.
_CROSS_CORE_SYNC_MD = "target/ascendc/fa_class/cross_core_sync.md"

# P-P116 — the a3/arch22 hand-authored cube+vector MIX attention STARTING SKELETON.
# Its own header `applies_to: soc=Ascend910_9382` (a3/a2 → 220x) + `unverified_on:
# soc=Ascend950PR` is what scopes the a3 delivery block below to a3 only.
_A3_MIX_TEMPLATE_MD = "target/ascendc/patterns/domains/fa_class_a3_mix_template.md"


def _mix_sync_a5_recipe_applies(target: str) -> bool:
    """Does the A5 two-path recipe apply to `target`? Read from the KB entries.

    Path B is anchored on OL-220 (`soc=Ascend950PR` — the GDN light-port build
    recipe, 122/122) and Path A on `cross_core_sync.md` §4 (`soc=Ascend950PR` —
    the PUBLIC-API-runnable handshake). Both must apply for the two-path choice to
    be a real choice, so this is an AND, not an OR: surfacing one path without its
    exclusive alternative is what produced the blend this block exists to stop.
    """
    return kb_entry_applies_to_target("OL-220", target) and kb_section_applies_to_target(
        _CROSS_CORE_SYNC_MD, "4", target
    )


def _fa_mix_sync_intro_block(target: str) -> str:
    """Header naming the target + the pick-exactly-one-path framing."""
    return (
        f"## MIX cube+vec CROSS-CORE SYNC (target `{target}`) — the sync discipline is a\n"
        "## WHOLE-PACKAGE CHOICE, not a menu of tips. Read this before you write a flag:\n"
        "Your `cube:vec = 1:2 MIX` (`KERNEL_TYPE_MIX_AIC_1_2`) handoff from recipe step 6 is\n"
        "the cross-core-sync surface where FA-class ops hang. Everything below is scoped by the\n"
        "KB's own `applies_to: soc=` — if a card is not here, it does NOT apply to your target,\n"
        "and you should not go looking for it.\n"
    )


def _fa_mix_two_paths_block() -> str:
    """The A5 MIX cube↔vec recipe: TWO MUTUALLY EXCLUSIVE proven paths.

    Scoped by `_mix_sync_a5_recipe_applies` (OL-220 ∧ cross_core_sync.md §4, both
    `soc=Ascend950PR`). Presented as an exclusive CHOICE because PB-34's Fix
    section defines Pattern A and Pattern B as exclusive and BLENDING THEM IS THE
    BUG: `3_FusionAttention`'s `fa_fused_mixed_fp16` pairs `MatmulImpl<>` with
    hand-rolled `CrossCoreSetFlag<0x2>` and its own comment at `:28` copies the
    warning it then violates. A menu ("library cube + §4 handshake") is an
    invitation to re-create exactly that structure.

    POINTS at PB-35 + §4 rather than copying them: a copy drifts, and a paraphrase
    strips the bounds (§4's 64/64 witness is the `Buffer<CROSS_CORE_SYNC_FORWARD>`
    abstraction NOT a hand-roll; (C)'s `PIPE_V` holds only for a UB-resident
    result). Both bounds are stated here as REASONS TO READ §4, not as a substitute
    for it.
    """
    return (
        "\n"
        "### PICK EXACTLY ONE PATH — they are MUTUALLY EXCLUSIVE (PB-34 Fix defines them so)\n"
        "**Blending them IS the bug.** `MatmulImpl<>` (Path B's cube) + Path A's manual §4\n"
        "handshake is the failing structure of our own `3_FusionAttention` `fa_fused_mixed_fp16`\n"
        "— which pulls in `MatmulImpl<>` AND hand-rolls `CrossCoreSetFlag<0x2>`\n"
        "(`fusion_attention_fused_kernels.cpp:105/116/142/152`) while its OWN comment at `:28`\n"
        "already states the hazard ('MatmulImpl::IterateAll INTERNALLY uses CrossCoreSetFlag.\n"
        "User flag IDs 0-3 may collide with internal IDs'). It copied the warning and mixed anyway.\n"
        "Take ONE path WHOLE. Do NOT take 'the good parts of both'.\n"
        "\n"
        "#### PATH B — library cube + KFC-IMPLICIT sync (**the A5 DEFAULT — full-op PROVEN**)\n"
        "Keep `MatmulImpl<>`. Let KFC do the cube↔vec sync. **ZERO manual `CrossCoreSetFlag` /\n"
        "`CrossCoreWaitFlag` on EITHER side — not one call.**\n"
        "- **Witness (full-op, not a micro-probe)**: GDN `chunk_gated_delta_rule` light-port —\n"
        "  8 `matmul::MatmulImpl<>` instances ×3 cube stages + `MIX_AIC_1_2`, compiled first-try on\n"
        "  bisheng dav-c310, no hang, **122/122 T1 PASS** (A5, CANN 9.1.T500). This is PB-34's own\n"
        "  `verified_does_not_reproduce_on (FULL-OP scale)` bullet — read it in `PLATFORM_BUGS.md`.\n"
        "- **This is what PB-34 tells an A5 worker to do**: 'for a V220 cube-MIX fused op, the\n"
        "  DEFAULT A5 route is a LIGHT PORT (keep `MatmulImpl<>` + the manual flag chain; adapt only\n"
        "  the ACLRT_LAUNCH entry + host tiling), NOT a hand-rolled tile-Mmad rewrite.'\n"
        "- **Build recipe: OL-220** (`ascendc_library` cube+vec MIX — non-empty `CMAKE_BUILD_TYPE`,\n"
        "  post-`ascendc_library` include scoping, where `MultiCoreMatmulTiling` lives).\n"
        "- **`cross_core_sync.md` §4 DOES NOT APPLY TO THIS PATH. Do NOT add it.** §4 is the\n"
        "  hand-rolled handshake; bolting it onto a KFC cube re-creates the PB-34 blend above.\n"
        "\n"
        "#### PATH A — non-KFC library cube + the §4 manual handshake\n"
        "Drop `MatmulImpl<>`/KFC entirely. Use a **library** block-level GEMM that owns no KFC — the\n"
        "catlass tile→block→epilogue layers (`Gemm::Block::BlockMmadTla`, wrapped as one\n"
        "layout-parameterized `RunGemm` helper; design + working single-chunk GDN in\n"
        "`docs/design/FA_CLASS_DESIGN_NOTES.md#gdn-catlass-composable-primitives-design`). No KFC\n"
        "means no FFTS-slot contention, which is what makes manual flags SAFE here.\n"
        "- **Then take the handshake from `target/ascendc/fa_class/cross_core_sync.md` §4 (`:209`,\n"
        "  verdict PUBLIC-API-runnable)** — **(A)** SYNC MODE 4, not mode 2 (`:226`); **(B)** disjoint\n"
        "  per-sub-block flag ids `id` / `id+16`, and **BOTH must be Set** — sending one leaves the\n"
        "  second AIV with no happens-before (`:233`); **(C)** the consumer `Wait`s on **`PIPE_V`**, NOT\n"
        "  the producer's `PIPE_FIX` (`:243`).\n"
        "- **READ §4 ITSELF — do not work from this summary. Two bounds a paraphrase strips:**\n"
        "  1. §4's `verified_on` 64/64 reference achieves the handshake via the BaseApi\n"
        "     **`Buffer<CROSS_CORE_SYNC_FORWARD>` abstraction, NOT a hand-roll**; the hand-rolled form\n"
        "     is public-API-runnable ONLY WITH (C). **Prefer the abstraction** — it picks the pipe +\n"
        "     managed id for you, which is why it is bit-exact and deadlock-free.\n"
        "  2. **(C) is bounded**: `PIPE_V` is correct only for a **UB-RESIDENT** result. cube→vec\n"
        "     **via GM** needs more (`:317`, kw-gb5): `PIPE_V` gates the AIV vector pipe but leaves the\n"
        "     MTE2 GM-read un-ordered vs the cube's Fixpipe retire → reads GM too early (init data) or\n"
        "     mid-write (`507015` aivec OOB). Cleanest fix: route the result Fixpipe L0C→**UB**.\n"
        "- **Witness**: `workspace/gdn_catlass/` single-chunk GDN — A5/CANN-9.1.T500, 3/3 vs fp64\n"
        "  oracle @4e-2, **deterministic ×3**. Sync discipline = `AscendC::SyncAll` (single-chunk) or\n"
        "  CrossCore **mode-4** flag rotation over a static collision-free flag-id space.\n"
        "- **Determinism must be tested fresh-PROCESS ×≥3** (`:329`) — warm in-process re-runs mask the\n"
        "  race; the first launch fixes a scheduling path that repeats within that process.\n"
    )


def _fa_a3_mix_skeleton_block() -> str:
    """The a3/arch22 FA-class STARTING SKELETON: P-P116 + the PB-55 handshake rule.

    The a3 mirror of the a5 `_fa_mix_two_paths_block` recipe (DEBT-222 delivery):
    where an a5 FA worker is handed its two proven MIX routes, an a3 worker is
    handed the device-proven hand-authored cube+vector skeleton P-P116
    (`fa_class_a3_mix_template.md`, DS `famix`/`famix_mh`, `Ascend910_9382`,
    cann=9.0.0) as the place to START — not a from-scratch author. It is the a3
    COUNTERPART of the a5-only P-P103 `fa_class_template.md` / P-P102
    `cube_vector_fusion.md`, and delivering the a5 template to an a3 worker (or this
    a3 skeleton to a5) is exactly the mis-scoped-knowledge delivery DEBT-208/DEBT-222
    guard against.

    Emitted iff `fa_class_a3_mix_template.md`'s own header `applies_to:
    soc=Ascend910_9382` covers the target (via `kb_file_applies_to_target`, the same
    DEBT-208 discipline the other cards use). P-P116 is `unverified_on:
    soc=Ascend950PR`, so this is SUPPRESSED on a5 and widens NO a5 scope — the a5
    brief stays byte-identical.

    POINTS at the file rather than copying it (a copy drifts; a paraphrase strips
    the bounds). Carries the two load-bearing facts a worker cannot get wrong: the
    PB-55 reverse-handshake rule (BOTH AIV subblocks set `FLAG_P`, else DEADLOCK) and
    the HONEST SCOPE (seq≤384, d=64, fp16, single-pass NON-flash, no causal/mask, no
    tuned-perf) so it is not mistaken for a finished flash-attention.

    Carries the STRICT-BAR THREE-STAGE discipline (owner 2026-07-18, post #188/#189
    rework): (1) READ the reworked template P-P116 as reusable pattern KNOWLEDGE —
    pseudocode + the one handshake snippet + compute-as-knowledge + host guidance +
    the shipped-library steering table (USE CANN `MatmulImpl` / `AscendC::SoftMax` /
    catlass `CrossCore`; our `MixSoftmax`/`RowReduce*`/`MakeMixCfg` = op-glue you
    GENERATE, not lift); (2) BUILD + RUN the SYNC-WITNESS `examples/a3_mix_fa_min/`
    (#189 thinned it to a handshake-only demonstrator — genuine cubes + PB-55
    handshake PRESERVED, AIV compute now a zero-liftable PLACEHOLDER identity copy)
    to watch the AIC↔AIV handshake close deadlock-free on device — see-it-work beats
    prose, NOT a copyable op; (3) GENERATE your own kernel + host from the template
    knowledge using SHIPPED-library primitives (`#include` CANN/catlass), lifting
    neither the witness body nor our hand-written helpers nor any non-shipped source.
    Pointed at (not restated: DEBT-223); DEBT-215 scanner + the witness's
    zero-liftable helpers enforce the no-copy bar. a3-only. This REPLACES the stale
    "materialize the full compilable reference" framing (#186 / #177).
    """
    return (
        "\n"
        "### a3 FA-CLASS STARTING SKELETON — P-P116 "
        "(`fa_class_a3_mix_template.md`): START HERE, do NOT author from scratch\n"
        "For a **220x / a3** (`Ascend910_9382`, arch22) cube+vector MIX attention op the KB now carries a\n"
        "**device-proven hand-authored starting skeleton — P-P116**\n"
        "(`src/skills/references/target/ascendc/patterns/domains/fa_class_a3_mix_template.md`;\n"
        "`verified_on: Ascend910_9382; cann=9.0.0; DS famix (single-head) + famix_mh (multi-head)`). It is the a3\n"
        "COUNTERPART of the a5-only P-P103 `fa_class_template.md` / P-P102 `cube_vector_fusion.md` — **use P-P116\n"
        "for a3, NOT the a5 templates** (their arch35 §4 mode-4 sync / MicroAPI regbase softmax are WRONG on a3).\n"
        "#### THREE-STAGE DISCIPLINE — READ the template → BUILD+RUN the witness "
        "→ GENERATE your own with shipped libraries\n"
        "The STRICT bar: you GENERATE the op from KB knowledge + the customer's SHIPPED libraries — you do NOT copy a\n"
        "liftable artifact. Follow three stages, in order:\n"
        "1. **READ the template P-P116** (`fa_class_a3_mix_template.md`, path "
        "above) as reusable pattern KNOWLEDGE — the\n"
        "   MIX pseudocode, the one unavoidable handshake snippet, the "
        "compute-as-KNOWLEDGE, the host-generation guidance,\n"
        "   and — load-bearing — its **shipped-library steering table**: USE CANN `MatmulImpl` / `AscendC::SoftMax` /\n"
        "   catlass `CrossCore` (`catlass/arch/cross_core_sync.hpp`) — **the customer HAS these**; our hand-written\n"
        "   `MixSoftmax` / `RowReduce*` / `MakeMixCfg` helpers are **op-glue you "
        "GENERATE yourself, do NOT lift**. The\n"
        "   template POINTS at the pattern; it holds NO liftable op body.\n"
        "2. **BUILD + RUN the SYNC-WITNESS** "
        "`src/skills/references/target/ascendc/examples/a3_mix_fa_min/` on YOUR a3\n"
        "   container — watch `torch.npu.synchronize()` RETURN (not hang) and witness the AIC↔AIV MIX handshake close\n"
        "   **deadlock-free** on device (the empirically-decisive see-it-work step; "
        "PB-55's both-AIV-set reverse handshake\n"
        "   is PROVEN, not a hope). Its compute is a **PLACEHOLDER** (identity copy "
        "— the real op-helpers were DELETED,\n"
        "   zero-liftable): it is a **sync DEMONSTRATOR, NOT a copyable op**.\n"
        "3. **GENERATE your own kernel + host** from the template knowledge, **using shipped-library primitives** —\n"
        "   `#include` the CANN / catlass headers (`lib/matmul_intf.h`, the SoftMax "
        "lib, `catlass/arch/cross_core_sync.hpp`),\n"
        "   which resolve at BUILD time on the NPU server. Do **NOT** lift the "
        "witness body, our example's hand-written\n"
        "   helpers, or any non-shipped source. The customer-portability test is: "
        "**does the customer's own CANN/catlass\n"
        "   BUILD it?** The DEBT-215 scanner + the witness's zero-liftable-helpers backstop enforce this. a3-only\n"
        "   (arch22, `Ascend910_9382`); it makes NO claim on a5.\n"
        "**READ the file — this is the map, not the territory. The load-bearing structure:**\n"
        "- **Dispatch**: ONE MIX_AIC_1_2 kernel via the standard `aclrtlaunch` runtime; `if ASCEND_IS_AIC` /\n"
        "  `if ASCEND_IS_AIV` partition the work inside one `__global__ __aicore__` entry. **On arch22 do NOT emit\n"
        "  `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`** — that macro is arch35-only and is REJECTED on\n"
        "  `Ascend910_9382` (`ACL_ERROR_RT_PARAM_INVALID=107000` at `RegisterAscendBinary`). This is the\n"
        "  load-bearing a3↔a5 divergence: P-P102/P-P103 tell an a5 worker to KEEP that macro; an a3 worker OMITS it.\n"
        "- **AIC** (`if ASCEND_IS_AIC`): two GENUINE cubes via `MatmulImpl` **`IterateAll<sync=true>` + `End()`**\n"
        "  (NEVER the async KfcServer `Iterate()`/`GetTensor()` path — it grabs the AIC↔AIV FFTS sync slots and\n"
        "  deadlocks in this MIX mode; that is PB-34). cube#1 `S = Q@Kᵀ` (`SetTensorB(k, isTransposeB=true)` — the\n"
        "  transpose is a RUNTIME bool, the static `ISTRANS` template flag stays false) → `CrossCoreSetFlag(FLAG_S)`\n"
        "  → `CrossCoreWaitFlag(FLAG_P)` → cube#2 `O = P@V`. A pure-vector emulation of either matmul is the\n"
        "  OL-188 architectural hack (cube-required op) and is SHIP-BLOCKED — do NOT fall back to it.\n"
        "- **AIV** (`if ASCEND_IS_AIV`): `CrossCoreWaitFlag(FLAG_S)` → row-wise fp32 softmax in UB\n"
        "  (Cast fp16→fp32 → Muls by `scale=1/√d` → `WholeReduceMax` → Adds(-max) → Exp → `WholeReduceSum` →\n"
        "  Muls(1/sum) → Cast fp32→fp16) → publish `FLAG_P`.\n"
        "#### PB-55 — the REVERSE (AIV→AIC) handshake is per-subblock-COUNTED; "
        "a single setter DEADLOCKS (READ before you write FLAG_P)\n"
        "The `MIX_AIC_1_2` cross-core handshake is **DIRECTION-ASYMMETRIC**. The FORWARD flag (AIC→AIV, `FLAG_S`)\n"
        "is BROADCAST — one AIC `CrossCoreSetFlag` releases BOTH AIV subblocks. The **REVERSE flag (AIV→AIC,\n"
        "`FLAG_P`) is per-subblock-COUNTED**: the single AIC `CrossCoreWaitFlag(FLAG_P)` requires a set from\n"
        "**EVERY AIV subblock of the 1:2 pair**. **BOTH AIV subblocks MUST `CrossCoreSetFlag(FLAG_P)`** — raising\n"
        "it from only subblock 0 → AIC waits FOREVER (measured DEADLOCK: hang at `synchronize`, no fault code).\n"
        "Both subblocks recompute the softmax redundantly (identical values → benign) and both set `FLAG_P`.\n"
        "Bisect evidence in PB-55: `NO_REVERSE`=SYNC_OK, single-setter reverse=DEADLOCK, both-set=PASS.\n"
        "Sync discipline: `MIX_SYNC_MODE2` (mode 2) SUFFICES on arch22 (the arch35 §4 mode-4 / `id`,`id+16`\n"
        "disjoint-id recipe is NOT needed here). Distinct flag ids per handshake (`FLAG_S=4` / `FLAG_P=5`, user\n"
        "range 1..7) — **NEVER flag id 0** (PB-35: `event_t(0)` collides with the cube-internal pipe-sync chain).\n"
        "#### HONEST SCOPE — P-P116 is a STARTING skeleton, NOT a finished flash-attention (do NOT overclaim):\n"
        "- **Device-verified**: seqlen ≤ 384, d = 64, fp16 I/O (fp32 softmax accumulate), **single-pass NON-flash**\n"
        "  softmax (the whole `[seq,seq]` S row is in GM, softmax reduces one full row at a time), multi-head\n"
        "  (device-side `nheads` loop, one reused `[seq,seq]` S/P GM scratch pair), cosine 0.999999 /\n"
        "  `max_abs_diff` 4.9e-4, deterministic bit-exact across fresh processes.\n"
        "- **NOT yet proven — DEBUG-ON-TOP, do NOT claim as covered**: flash online-softmax with KV-block tiling\n"
        "  (add the P-P101 running-max/sum/expMax delta ONLY when a long-seq op needs it — not a wall at seq≤384),\n"
        "  causal / attention mask (not wired in the skeleton), tuned perf (a3 has NO vendor FA baseline; the\n"
        "  1.21 ms figure is a functional-correctness datapoint, not a perf claim).\n"
        "- **`unverified_on: soc=Ascend950PR`** — P-P116 makes NO claim on a5; this block is a3-only by that scope.\n"
    )


def _fa_mix_pb34_v220_block() -> str:
    """PB-34: `MatmulImpl<>` + manual CrossCore + `MIX_AIC_1_2` → silent hang.

    Emitted iff PB-34's `applies_to: soc=Ascend910_9382` covers the target — i.e.
    V220 (a3/a2) only. On A5 this is SUPPRESSED: PB-34 carries two
    `verified_does_not_reproduce_on: Ascend950PR` witnesses and its Consequence
    line makes the light-port the A5 default, so injecting it there inverted the
    card's own advice. Text preserved from the 2026-07-16 recall-fix; the prose
    "V351/A5 scope bound — do NOT over-apply" trailer is GONE because the bound is
    now the `if` in the composer, not an instruction the reader must obey.
    """
    return (
        "\n"
        "### PB-34 — MIX cube+vec SILENT-HANG (V220; the CONFIRMED cause of our own 507014)\n"
        "**On V220** (`soc=Ascend910_9382`, arch22 / port_a3 target): a kernel that pairs\n"
        "`MatmulImpl<>` (or `MatmulClient`/KFC) with **manual** `CrossCoreSetFlag<0x2>` /\n"
        "`CrossCoreWaitFlag<0x2>` under `MIX_AIC_1_2` builds + register-binaries CLEANLY, then HANGS\n"
        "FOREVER on the FIRST launch — the host sits at `torch.npu.synchronize()` past ~45s with NO\n"
        "aicore exception (error `507014` AICore timeout / `507015` MTE workspace-OOB / `507035`\n"
        "vec-abnormal, whichever engine starves first).\n"
        "**Root cause — KFC slot contention.** `MatmulImpl<>`'s KFC internal flag-count state machine\n"
        "collides with the user-owned FFTS slots: KFC consumes the same flag-ID slots that\n"
        "`CrossCoreSetFlag<0x2>(0..7)` allocates from. Our failing kernel mixes BOTH halves in ONE\n"
        "kernel — `3_FusionAttention`'s `fa_fused_mixed_fp16` pulls in `MatmulImpl<>` (via\n"
        "`lib/matmul/matmul.h`) AND hand-rolls `CrossCoreSetFlag<0x2>`\n"
        "(`fusion_attention_fused_kernels.cpp:105/116/142/152`); its AIC branch does both, and the\n"
        "kernel's own comment at `:28` already flags the hazard. PB-34's `verified_on` names this very\n"
        "kernel. **If your kernel does both, this is your bug — fix it here, not in the sync base.**\n"
        "**Fix = Pattern A XOR Pattern B, never mixed:**\n"
        "- **Pattern B**: keep `MatmulImpl<>`, use KFC-implicit sync ONLY — remove ALL manual\n"
        "  `CrossCoreSetFlag<0x2>`/`WaitFlag`. Viable only when the cross-stage handoff fits inside one\n"
        "  `Iterate()` boundary (multi-stage FA often does NOT).\n"
        "- **Pattern A**: drop `MatmulImpl<>` for tile-MMAD primitives and KEEP the manual flag chain.\n"
        "  **NOT a free escape — read the PB-35 card below BEFORE you hand-roll it.**\n"
        "- **cube-only** (AIC_ONLY, no vec epilogue → no MIX sync surface): sidesteps the `507014` KFC\n"
        "  wall entirely — anchor DEBT-206 (1_BatchMatmul, Ascend910_9382), **SHIPPED end-to-end; see\n"
        "  the dedicated section below for the recipe + its bounds**. catlass is the vendor-origin\n"
        "  composable-cube variant of this route — **UNPROVEN on V220 (no execution witness in the KB):\n"
        "  a candidate to evaluate, NOT a recipe to follow.**\n"
        "**Intra-AIC cube pipe sync on V220 — do NOT hand-roll it.** Pattern A with user-owned\n"
        "cube-internal pipe sync remains **UNSOLVED in canonical KB** on V220: the 'use event ids ≥ 4'\n"
        "fix was **empirically falsified** (3 distinct schemes — raw `event_t(2..7)`, canonical\n"
        "`GetTPipePtr()->FetchEventID()` — ALL reproduce the same silent hang; `PLATFORM_BUGS.md:934`).\n"
        "**Use a LIBRARY cube.** This paragraph bounds the **hand-rolled** intra-AIC pipe sync ONLY —\n"
        "do NOT read it as 'cube is impossible on V220, fall back to vector'. A non-KFC library cube\n"
        "has ALREADY SHIPPED here (DEBT-206); the next section is that route and its bounds.\n"
    )


def _fa_mix_v220_shipped_cube_block() -> str:
    """OL-275: a non-KFC LIBRARY cube HAS shipped end-to-end on V220 (DEBT-206).

    Why this block exists (2026-07-17): the PB-34 V220 block closed with "Use a
    library cube, or stay on the AIV-only VEC fallback **until the canonical V220
    cube workflow lands**." The second clause was FALSE — the workflow HAD landed —
    and the block contradicted itself: its own `cube-only` bullet already named
    DEBT-206 four lines earlier. The false clause was the one written in plain
    imperative language, so a V220 FA worker read the section as "cube is
    unavailable here, go vector", which is how an attention op ended up pure-vector.
    Two facts refute the trailer, BOTH already in the KB this brief cites:
      - **OL-275 carries the DEBT-206 witness** (`OPERATIONAL_KNOWLEDGE.md:11016`):
        the FIRST SHIPPED `verified_on:a3` cube op — a real archived end-to-end
        deliverable on the standalone `build_ascendc.py`, not a micro-probe.
      - **The vector fallback it offered is a SHIP-BLOCKED state.** For a
        cube-required op a pure-vec kernel is an OL-188 anti-cheat HACK:
        `_check_architecture_class` (`finalize_checks_structural.py:447/478`)
        returns ARCHITECTURAL_HACK and blocks ship. `kw_brief_pa3_phases:85` tells
        the SAME worker "**NEVER** to a pure-VEC fallback (the gate will reject
        it)". The FA brief was steering into the gate the port_a3 brief warns about.

    Gated on **OL-275's own** `applies_to: soc=Ascend910_V220` — the entry that
    carries the DEBT-206 witness — per the DEBT-208 rule that each sub-block is
    emitted iff the KB entry it carries covers `target`. This widens NO A5 scope:
    the A5 recipe stays anchored on OL-220 ∧ `cross_core_sync.md` §4, and OL-275 is
    already in the V220 fix-card trailer, so the a5 brief is byte-unchanged.

    Deliberately NOT written here: a catlass/`BlockMmadTla` V220 recipe. The vendor
    ships it as its default-arch (`CATLASS_ARCH=2201`) V220 example, but the KB
    carries NO V220 execution witness for it — writing an unproven recipe into a
    brief is the same defect this block fixes. It is labelled unproven, not
    recommended.
    """
    return (
        "\n"
        "### A non-KFC LIBRARY CUBE HAS SHIPPED ON V220 — 'cube is impossible, use vector' is FALSE\n"
        "**The V220 cube workflow HAS landed** — OL-275 (`OPERATIONAL_KNOWLEDGE.md:11016`):\n"
        "`1_BatchMatmul` (DEBT-206, 2026-07-13, Ascend910_9382 / **CANN 9.1.0**) is the **first\n"
        "SHIPPED `verified_on:a3` cube op**, a real archived end-to-end deliverable — NOT a probe:\n"
        "- direct `MatmulImpl<>` with **NON-KFC synchronous `IterateAll<sync=true>` + `End()`** — no\n"
        "  `REGIST_MATMUL_OBJ`, no msg-ring → **no KFC msg-ring means no FFTS slot contention, which is\n"
        "  precisely why PB-34 above does not fire on it**;\n"
        "- cube body guarded by **`ASCEND_IS_AIC`** (`AIC_ONLY`) so only the AIC runs the matmul;\n"
        "- **builds + runs clean on the standalone `build_ascendc.py`** — the same harness you are\n"
        "  using; no GE op-build, no bishengir;\n"
        "- **deterministic 5/5 bit-identical**; ships PARTIAL_PERSIST 46/51 strict (the 1 FAIL is an\n"
        "  fp32 reference-ub cancellation case, OL-276 — not a sync bug).\n"
        "**BOUNDS — carry these, do NOT overclaim:**\n"
        "1. **DEBT-206 is CUBE-ONLY.** `AIC_ONLY` = no vec epilogue = **no MIX cross-core surface at\n"
        "   all**. It proves a non-KFC library cube RUNS on V220. It does **NOT** prove the cube↔vec\n"
        "   MIX half your FA op needs. Never cite it as an FA MIX solution.\n"
        "2. **It was verified on CANN 9.1.0.** On a **9.0.0** box treat it as UNCONFIRMED and\n"
        "   re-verify before relying on it.\n"
        "3. Its `ASCEND_IS_AIC` body-guard and a constexpr `MatmulApiStaticTiling` on-stack\n"
        "   `TCubeTiling` were applied TOGETHER and never A/B-isolated — read them as cube-only\n"
        "   hygiene, not one attributed fix.\n"
        "**Honest state of the V220 cube+vec MIX question (no invented recipe — this is ALL we have):**\n"
        "- **PROVEN — ascendc backend, CUBE-ONLY**: the DEBT-206 `AIC_ONLY` route above. Prefer it\n"
        "  wherever your op decomposes into cube-only work with the vector work in a SEPARATE launch.\n"
        "- **NOT PROVEN — the genuine gap**: a HAND-ROLLED ascendc cube+vec MIX on V220 through\n"
        "  `build_ascendc.py`. **No such recipe exists in the KB.** If your op truly needs one, that is\n"
        "  a real GAP — record it with evidence and escalate. Do not invent one, and do not paper over\n"
        "  it with vector.\n"
        "- **UNPROVEN CANDIDATE (labelled, not recommended)**: catlass `Gemm::Block::BlockMmadTla`\n"
        "  ships as the vendor's default-arch (`CATLASS_ARCH=2201`) V220 example, but the KB holds\n"
        "  **no V220 execution witness** for it. Evaluate it if you choose; do not follow it as a\n"
        "  proven recipe.\n"
        "**The AIV-only VEC fallback is NOT a safe default — it is SHIP-BLOCKED.** For a cube-required\n"
        "op a pure-vec kernel is an OL-188 anti-cheat HACK: the finalize gate `_check_architecture_class`\n"
        "(`finalize_checks_structural.py:447`) returns **ARCHITECTURAL_HACK** and blocks ship (the\n"
        "port_a3 brief says the same: '**NEVER** to a pure-VEC fallback'). Take it ONLY as a measured\n"
        "last resort with the reason recorded — **never because 'the canonical workflow has not\n"
        "landed'. It has.**\n"
    )


def _fa_mix_pb35_pattern_a_block() -> str:
    """PB-35: `event_t(0)` cube-internal pipe sync collides with the cross-core chain.

    Emitted iff PB-35's `applies_to: soc=Ascend910_9382,Ascend950PR_9579` covers
    the target — i.e. BOTH SoCs, so this reaches an A5 worker. That is the point:
    PB-35 `attacks Pattern A itself` (`op_class=mixed_aic_aiv_pattern_a_tile_mmad`)
    and is `confirmed_on` V351/A5, so on A5 it is the mode that actually bites once
    PB-34 is (correctly) suppressed. Until 2026-07-17 PB-35's `applies_to` said
    V220-only while its own `confirmed_on` said A5 — honoring `applies_to` would
    have SUPPRESSED it on the one SoC where it is confirmed. That contradiction is
    fixed on main and `kb_index_audit.check_soc_scope` now hard-fails on the shape.
    """
    return (
        "\n"
        "### PB-35 — the Pattern-A trap (`PLATFORM_BUGS.md:883`; applies to V220 **and** V351/A5)\n"
        "`applies_to: op_class=mixed_aic_aiv_pattern_a_tile_mmad` — this card attacks **Pattern A\n"
        "itself**, and it is **`confirmed_on: Ascend950PR_9579` (V351/A5)**: a user-owned-Mmad +\n"
        "hand-rolled-cross-core-flag cube-MIX FA **DEADLOCKED on A5** (kw-gb2 hermetic graybox\n"
        "2026-06-03). Two named traps:\n"
        "- **Never use `event_t(0)` for cube-internal pipe sync** — it collides with cross-core flag\n"
        "  id 0 (`FLAG_CANON_DONE` in FA-class kernels). Cube-internal pipe events and cross-core flags\n"
        "  share the user-owned id range, so they MUST be allocated disjointly. (Note the card's own\n"
        "  'use ids ≥ 4' Fix was falsified on V220 — the collision is real, that remedy is not.)\n"
        "- **A SYNC-MODE-2 (1:2 broadcast) hand-roll sharing ONE flag id across both AIV sub-blocks\n"
        "  deadlocks.** This is exactly what §4's (A)+(B) correct.\n"
        "**Never hand-pick event-id literals** — use a managed rotation pool; a literal like `6` can\n"
        "alias a managed id.\n"
    )


def _fa_mix_pb45_library_cube_block() -> str:
    """PB-45: arch35 `TPipe::Reset()` frees the GLOBAL event pool → use a library cube.

    Emitted iff PB-45's `applies_to: soc=Ascend950PR (V351/A5)` covers the target.
    Carries the A5 half of "do not hand-roll intra-AIC cube pipe sync" (the V220
    half rides in the PB-34 block, from `PLATFORM_BUGS.md:934`).
    """
    return (
        "\n"
        "### PB-45 — intra-AIC cube pipe sync on A5: use a LIBRARY cube, do not hand-roll\n"
        "`TPipe::Reset()` frees the **global** `g_tpipeImpl` event pool + buffer cursor on arch35, so a\n"
        "multi-stage MIX kernel cannot carry persistent cross-call sync state across a Reset boundary\n"
        "(persistent `Buffer<>` credits primed at Init are correct ONLY when Init runs once for the\n"
        "kernel lifetime — the library `MatmulImpl` / Path-B case). If a hand-roll IS required, drive\n"
        "cube-internal L0 fences with Reset-safe back-to-back `FetchEventID` Set+Wait, no persistent\n"
        "credits, no hardcoded ids (**OL-223**).\n"
        "**Why this pushes you to a library cube**: even after the Reset-safe fix, the GDN regbase\n"
        "hand-rolled cube left the M-tail cross-core handshake **non-deterministic below event-id\n"
        "granularity** (the irreducible PB-35 wall) — all 122 cases RUN clean, but the pass count is\n"
        "NOT bit-stable run-to-run. The deterministic answers are the two library paths above: the\n"
        "light-port (Path B) runs 122/122, the catlass composition (Path A) is deterministic ×3.\n"
    )


def _fa_mix_debt210_sync_base_block() -> str:
    """DEBT-210: the FFTS sync-base compile-path gap — a FACT, not a proven cause.

    SoC-INDEPENDENT and always emitted: DEBT-210 is a harness/compile-path gap in
    `build_ascendc.py`'s ACLRT stub (shared by every target), not a KB entry with
    an `applies_to` scope — so `kb_scope` fails open on it and it reaches both
    briefs. Kept as its own block precisely so it is NOT re-welded to PB-34: the
    2026-07-17 retraction below was a deliberate, evidence-backed correction and
    must not be lost when PB-34 is scoped away on A5.
    """
    return (
        "\n"
        "### DEBT-210 — FFTS sync base not emitted (a COMPILE-PATH gap; **NOT** a proven cause of a hang)\n"
        "`CrossCoreSetFlag<0x2>`/`WaitFlag` only route between cores once the FFTS C2C control address\n"
        "is installed as the kernel's sync base — a **three-line host↔kernel idiom**:\n"
        "    host   `rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);`\n"
        "    launch pass `fftsAddr` as the kernel's FIRST arg\n"
        "    kernel `AscendC::SetSyncBaseAddr(fftsAddr);`\n"
        "A GE op-build supplies it automatically, and so does bishengir — but a standalone non-GE host\n"
        "CAN supply it too; GE is NOT required. `build_ascendc.py`'s ACLRT stub does **not currently\n"
        "emit** these lines (DEBT-210(d′), an open harness gap — not an impossibility).\n"
        "**RETRACTED 2026-07-17 — an unset sync base is NOT known to cause a hang.** The earlier\n"
        "'unset sync base → silent hang' arrow was correlation only, and a single-variable flip REFUTED\n"
        "the expected direction: deleting BOTH `SetSyncBaseAddr` calls from a known-good catlass MIX op\n"
        "left it PASSING — no 507014, no hang — with the instrument check confirming that op is a REAL\n"
        "cross-core op (9 `<0x2>` flags in its main compute loop). **Do not chase this**, and do not\n"
        "hand-patch the build: the three lines do not compile from our host TU today (the CANN runtime\n"
        "include sits on the DEVICE target and CMake `PRIVATE` does not propagate, while\n"
        "`rtGetC2cCtrlAddr` is a HOST API). A missing-header error here is EXPECTED — **escalate to\n"
        "DEBT-210(d′), do not work around it.**\n"
    )


def _fa_mix_fix_cards_block(target: str) -> str:
    """Trailing fix-card list, each card filtered by its OWN `applies_to: soc=`.

    The pre-DEBT-208 list was a fixed string — `PB-34, OL-275, OL-220, EC-68` —
    handed to every target regardless of scope. That mixed V220 cards (PB-34,
    OL-275: `soc=Ascend910_V220`, whose own `unverified_on` says "do not assume
    transfer" to A5) with A5 cards (OL-220, EC-68: `soc=Ascend950PR`) in one
    breath. Filtering each card by its own declared scope is the same one-line
    predicate as the blocks above — that is what makes DEBT-208 a class fix rather
    than a PB-34 patch.
    """
    cards = [
        ("PB-34", "root cause + Pattern A/B exclusivity"),
        ("PB-35", "the Pattern-A trap — event-id collision"),
        ("PB-45", "arch35 TPipe::Reset frees the global event pool"),
        ("PB-55", "reverse AIV→AIC handshake is per-subblock-COUNTED — BOTH subblocks must set FLAG_P"),
        ("OL-220", "cube+vec MIX `ascendc_library` build recipe"),
        ("OL-223", "Reset-safe cube-internal L0 fences"),
        ("OL-275", "managed-cube KFC lifecycle"),
        ("EC-68", "`507015` `SetSysWorkspaceForce` on ACLRT_LAUNCH MIX"),
    ]
    live = [f"{cid} ({why})" for cid, why in cards
            if kb_entry_applies_to_target(cid, target)]
    return (
        "\n"
        f"**Fix cards in scope for `{target}`** (each filtered by its own KB `applies_to: soc=`; a card\n"
        "absent here is scoped to another SoC — do not go pull it in): "
        + ", ".join(live)
        + ".\nCross-ref CAND-FA1.\n"
        "\n"
    )


def _fa_assembly_compile_block() -> str:
    """The §COMPILE MIX dual-pass host/device-split prose (the canon-0/21 fix).

    Extracted from `_fa_class_template_assembly_block` (DEBT-164 god-fn split).
    """
    return (
        "## COMPILE — MIX dual-pass host/device-split (#3-proven RC=0; THE canon-0/21 BLOCKER fix):\n"
        "build_ascendc.py compiles the kernel TU TWICE — a **device pass** (cce/dav-c310, MicroAPI\n"
        "available) AND a **host pass** (host_bisheng, NO MicroAPI). Guard device bodies with\n"
        "`#ifndef __ASC_NPU_HOST__` so the host pass sees an EMPTY device TU:\n"
        "  #include \"kernel_operator.h\"      // top, UNGUARDED — both passes need it\n"
        "  #ifndef __ASC_NPU_HOST__            // ↓ device-only: wholeport includes + MicroAPI/SIMD\n"
        "  #include \"wholeport/wp_fa_entry.h\"\n"
        "  extern \"C\" __global__ __aicore__ void wp_fa_do_<suffix>(...){ wp_fa_regbase_impl<...>(...); }\n"
        "  FA_DO_LIST                          // X-macro instantiates all entries\n"
        "  #endif  // __ASC_NPU_HOST__\n"
        "build_ascendc.py DEFINES `__ASC_NPU_HOST__` in the host pass → the guard makes the device\n"
        "body an empty TU (host needs no MicroAPI); the device pass (guard undefined) compiles the\n"
        "MicroAPI bodies (`AscendC::MicroAPI` is gated `__NPU_ARCH__==3510`, set by the device\n"
        "toolchain — do NOT set it yourself). Launch stubs are AUTO-GENERATED by build_ascendc.py\n"
        "from the device object; pybind11.cpp declares + calls them:\n"
        "  extern \"C\" uint32_t aclrtlaunch_<name>(uint32_t blockDim, void* stream, /*args*/);\n"
        "then SelectLauncher dispatches by layout/dtype/D.\n"
        "**FORBIDDEN (the canon-0/21 dead-end)**: do NOT `#define __NPU_ARCH__ 3510` at the kernel-TU\n"
        "top to give the HOST pass MicroAPI — that pulls arch35 DEVICE headers (PIPE_FIX@kernel_event.h,\n"
        "int4x2_t@vconv_impl, SIMT cce/dim3/Atomic*@simt_api) into the host pass → host_bisheng cannot\n"
        "compile CANN's arch35 device headers → BLOCKED. The host pass must EXCLUDE the device body\n"
        "(guard), NOT acquire MicroAPI. Evidence: #3(D=1024) used the guard → RC=0 + DS-verified 3/3;\n"
        "canon(clean) used `__NPU_ARCH__=3510` → BLOCKED 0/21 (same build_ascendc.py + same wholeport).\n"
        "\n"
    )


def _fa_assembly_verify_hard_block() -> str:
    """The §VERIFY + §DECLARED-COMBOS-HARD + §SPARSE-COMPRESSED-MASK prose.

    Extracted from `_fa_class_template_assembly_block` (DEBT-164 god-fn split).
    """
    return (
        "## VERIFY: self-contained (no external arch35 include) + bit-exact / within-T1-tol\n"
        "vs vendor-oracle. Spec-map (stitch-class per dtype×D-bucket×layout): §14.3.\n"
        "\n"
        "## DECLARED COMBOS THAT ARE HARD (NOT licenses to skip): d256/D>256 core-fill\n"
        "(parameterize S1Template common-fn), dropout (keep_prob<1, must wire keepProb —\n"
        "ignoring it → wrong dense output, a FAIL not an OOS), fp32, sparse, pse. These are\n"
        "DECLARED in tiling_key → you MUST assemble them, and **DEBUG-ON-TOP** when they fail\n"
        "(extend the template+host from arch22 + KB via Edit/Build/Verify — wire keepProb, fix\n"
        "tiling/registers, etc.). Record an explicit evidence-backed GAP (the failing combo +\n"
        "the hardware/impl reason) **ONLY if genuinely un-derivable from arch22 + KB after a real\n"
        "debug attempt** — **never silently omit a declared combo, never call it 'out-of-scope',\n"
        "never fast-stop on first failure** (the 2026-06-08 49/64 came from exactly that).\n"
        "\n"
        "## SPARSE 2/3/4 COMPRESSED-MASK (2026-06-09 pp-graybox root-cause): for sparse_mode\n"
        "2/3/4 (leftUpCausal/rightDownCausal/band) the vendor passes a COMPRESSED 2048×2048\n"
        "causal mask TEMPLATE, NOT an (S1,S2) per-element mask. The host/wrapper MUST detect\n"
        "`atten_mask.shape != (S1,S2)` and NOT slice/reshape it as an (S1,S2) mask (taking the\n"
        "first S1 rows of a 2048² template = garbage — this was the false 'masked-row-sentinel\n"
        "GAP'). Instead derive the causal band from pre_tockens/next_tockens + sparse_mode and\n"
        "let the kernel compute the band (the kernel's NO_COMPRESS path is correct). Band\n"
        "convention (canonical, per product model_new 235bc228): masked ⇔ j > i+next_tockens\n"
        "OR j < i−pre_tockens, True=masked. Generic rule: atten_mask shape ≠ (S1,S2) →\n"
        "band-from-tokens, NEVER first-rows-slice. Evidence:\n"
        "pp-graybox 2/3/4 FAIL(mad~2-3, mislabeled masked-row-sentinel) → PASS(mad 3-6e-4) after\n"
        "this fix; the kernel was always correct.\n"
        "\n"
    )


def _fa_class_backward_stitch_block(op: str, op_class: str) -> str:
    """FA-class BACKWARD template-stitch worker brief (2026-06-20).

    Routes an attention-family BACKWARD (gradient) op to the P-P103 BACKWARD
    section (`fa_class_template.md` L519-709 — the FA-grad stitch recipe), NOT
    the forward `_fa_class_template_assembly_block` (wrong: forward config-space
    + wp_fa_regbase_impl forward entry + 2-GEMM online-softmax phase-order) and
    NOT the generic analytic-derive BackwardPlugin brief: this route preserves
    the fused architecture, but the implementation is still authored
    independently from interface and algorithm semantics.

    The fused arch35 FA-grad is structurally isomorphic to the forward FA
    template (SAME CRTP <Cube,Vec>, SAME KERNEL_TYPE_MIX_AIC_1_2, SAME <<<>>>
    raw launch, SAME host-tiling POD) but with DIFFERENT blocks: 3-phase
    Pre/Base/Post, 5 cube GEMMs (dV/dP/dS-driven dQ/dK), softmax-GRAD vec, and
    consumes the forward-saved softmax_max/sum/attention_in. The brief encodes
    the 6 stitch steps + the splitAxis BN2-vs-BN2GS1S2 DECISION RULE (the
    precision bug) + the 14-fix build-drift catalog (each read from the compiler
    error, never guessed) + the host-tiling field math + the graybox-found
    "backward has NO core-fill" clarification (importing the forward core-fill →
    dk/dv garbage on S>64). CANN arch35 is a trusted behavioral and interface
    reference, not an implementation donor: block-by-block and line-by-line
    copying are forbidden; self-contained also means NO `#include arch35/`.

    Full-scope DEBUG-ON-TOP discipline (same as the forward block): the verified
    slice is BN2/S≤128/fp16-bf16/dense/BNSD/D128; S>128 (BN2GS1S2 + host
    axis-switch), fp32, and mask/pse/dropout/other-D/non-BNSD are DECLARED GAPs
    the worker must implement from source semantics + KB (Edit/Build/Verify), NOT
    fast-stop. Record an explicit evidence-backed GAP only if genuinely
    un-derivable after a real debug attempt.
    """
    return (
        "# PHASES (FA-class BACKWARD template-stitch — P-P103 BACKWARD section)\n"
        "\n"
        f"op `{op}` is an FA-class **BACKWARD** (gradient) op (tags: {op_class!r}).\n"
        "Paradigm = **independent fused-template implementation** guided by the forward\n"
        "operator contract, autograd oracle, and codified algorithm semantics, producing a\n"
        "self-contained A5/V351 op. This is NOT the analytic-derive backward path and NOT\n"
        "the forward FA template-assembly — it is the FA-GRAD STITCH RECIPE in P-P103's\n"
        "**BACKWARD section**. Do NOT hand-derive the grad math from scratch; do NOT author\n"
        "a multi-launch MatmulImpl kernel. Reconstruct the fused single-launch design without\n"
        "copying source blocks or lines.\n"
        "\n"
        "## THE GUIDE — read FIRST, follow EXACTLY:\n"
        "`kb/target/ascendc/patterns/domains/fa_class_template.md`\n"
        "→ **BACKWARD section** ('the FA-grad stitch recipe', the §after the FA+X table) +\n"
        "`CAND-FA-TEMPLATE-GEN-BWD-1` (candidates.md). The recipe = 6 stitch steps + the\n"
        "splitAxis DECISION RULE + the 14-fix build-drift checklist + the host-tiling field\n"
        "math + the REUSABLE-vs-GAP honesty table. It is a semantic recipe, not copy authority.\n"
        "\n"
        "## TRUSTED INPUTS (read for contracts and evidence):\n"
        "- **Forward specification + autograd oracle**: derive the backward interface, phase\n"
        "  ordering, and invariants from the declared forward math and measured CPU-fp64 truth.\n"
        "  Provenance-tracked target/prior artifacts are advisory only and cannot replace it.\n"
        "- **the forward FA template's shared `regbase_*` layer**: use an existing shared\n"
        "  project component through a minimal structural adapter where signatures differ;\n"
        "  the five backward GEMMs call the same cube primitives. Relevant interfaces are\n"
        "  `regbase_matmul.h` / `regbase_fixpipe_out.h` / `regbase_buffer*.h` / `regbase_copyin.h`,\n"
        "  plus the documented CANN common-component contracts. Do not transplant vendor bodies.\n"
        "- **codified KB**: P-P103 BACKWARD + the forward §Recipe (for the shared machinery).\n"
        "- **the autograd oracle + CDV harness**: the `fa_gqa_grad` pattern (`model.py` fp64\n"
        "  autograd backward = the precision truth).\n"
        "Record every trusted-source read in `reference_manifest.jsonl` using an allowed\n"
        "category. Self-contained = NO `#include \"arch35/\"` in any TU;\n"
        "the ARCH35_WRAP_CHEAT finalize gate rejects it. The implementation must also pass an\n"
        "allowed-input provenance review that logs every target/prior advisory read and\n"
        "rejects raw untracked copying or use of such context as truth.\n"
        "\n"
        "## RECIPE (the 6 stitch steps — P-P103 BACKWARD):\n"
        "1. **INTEGRATE the project's shared `regbase_*` layer** through includes or a minimal\n"
        "   structural adapter in the backward op's `shared/`; do not duplicate its bodies.\n"
        "2. **INDEPENDENTLY AUTHOR the fused FA-grad core** from the documented contracts:\n"
        "   `wp_fag_common.h` (carriers) /\n"
        "   `wp_fag_tiling_data.h` (POD) / `wp_fag_block_cube.h` (the 5 GEMMs: IterateMmDyV/\n"
        "   IterateMmQK/IterateMmDsK/IterateMmDsQ/IterateMmPDy) / `wp_fag_block_vec.h`\n"
        "   (softmax-grad: CalculateCastSoftmaxGrad + BroadcastSubMul(dS) + SimpleSoftMax) /\n"
        "   `wp_fag_kernel_base.h` + `wp_fag_kernel.h` (Pre/Base/Post + cv ping-pong Process)\n"
        "   / `wp_fag_pre.h` / `wp_fag_post.h` + independently authored `vector_api/*`\n"
        "   softmax-grad VEC leaves. Start with the declared dense-core scope; quant /\n"
        "   presfmg(hifp8) / deter / tnd / nz / rope / sink remain tracked GAPs. Preserve\n"
        "   behavior through tests, not textual transplantation; no `#include arch35/`.\n"
        "3. **AUTHOR the 3-phase entry `wp_fag_entry.h`** (templated\n"
        "   `wp_fag_regbase_impl<IN,float,OUT,s1T,s2T,dT>`): Phase-1 PRE / Phase-2 BASE (the\n"
        "   CRTP `<CubeBlockType,VecBlockType>` 5-GEMM+softmax-grad MIX) / Phase-3 POST, in ONE\n"
        "   `<<<>>>` launch. `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` +\n"
        "   `SetSysWorkspaceForce(workspace)` (bind sys-workspace base under custom `<<<>>>`\n"
        "   BEFORE GetUserWorkspace). For **BN2 dense (no sink): SKIP Pre+Post** (entry gates\n"
        "   them on sinkOptional; BN2 writes dq/dk/dv DIRECT via IS_DQ/DK/DV_WRITE_UB).\n"
        "4. **the splitAxis DECISION RULE (THIS is the precision bug — get it right)**: the host\n"
        "   picks `splitAxis` per CANN SetSplitAxis:\n"
        "     isBn2 = (s1≤128)&&(s2≤128)&&(n1==n2)&&(d≤512)&&(dtype≠fp32)&&(d==d1)&&!hifp8&&\n"
        "             !rope&&(tailZeroCount==0);  splitAxis = isBn2 ? BN2 : BN2GS1S2\n"
        "   **BN2** = the SIMPLE path (Base-only, dq/dk/dv DIRECT). **BN2GS1S2** = the multi-\n"
        "   block path (dq/dk/dv via fp32 workspace + Post). Hardcoding BN2GS1S2 for an isBn2\n"
        "   shape → wrong Process branch → garbage/zero. The entry's SPLIT_AXIS constant AND the\n"
        "   host POD MUST agree with isBn2.\n"
        "5. **host tiling is PART OF THE OP** (field-by-field port; stubbing → garbage). Port\n"
        "   the CANN op_host MATH faithfully into the pybind `DoTiling`: DoSplit/FuzzyForBestSplit\n"
        "   (s1Inner=s1Template/2, s2Inner=s2Template, CvRatio 2:1), DoBn2MultiBlkSparse dense\n"
        "   block-split (**maxValidBBLen** — a 0 here zeroes the output), coreNum=2·usedCore\n"
        "   (kernel does aicCoreNum=coreNum>>1, MIX 1:2), DoPostTiling (fp16/bf16 Post casts\n"
        "   fp32-ws→out-dtype on the BN2GS1S2/Post path), workspace (16MB sys + fp32 dq/dk/dv\n"
        "   partials 512B-aligned), and the POD population (`std::memset(&td,0,sizeof(td))`\n"
        "   first). POD GM→local read: the `GET_TILING_DATA_WITH_STRUCT` macro is ABSENT in\n"
        "   CANN 9.1.0 → use the manual uint32 word-blit.\n"
        "   **⚠️ THE BACKWARD HAS NO CORE-FILL — do NOT import the forward's** (the single most\n"
        "   likely autonomous-gen trap, graybox-confirmed 2026-06-20). The FORWARD's\n"
        "   'CalcS1S2BasicBlock core-fill → s1BasicBlock=64 for small shapes' does NOT exist in\n"
        "   the backward: `GetS1S2TemplateType` (CANN common_regbase.cpp) has NO core-fill —\n"
        "   dense fp16/bf16 ALWAYS s1Template=128, CUBE_BASEM=128; useS64/totalUnits override is\n"
        "   NEVER selected for dense fp16/bf16 D128. Importing the forward core-fill → s1=64 →\n"
        "   wrong CUBE_BASEM → dk/dv garbage on every S>64 case (dq stays fine, masking the bug).\n"
        "   Port the backward `GetS1S2TemplateType`, NOT the forward `CalcS1S2BasicBlock`.\n"
        "6. **the dispatcher TU + pybind + canonical entry-points** (port_a3 K5 dual-pass\n"
        "   discipline): `<op>_kernels.cpp` = `#include \"kernel_operator.h\"` UNGUARDED at top;\n"
        "   `#ifndef __ASC_NPU_HOST__` wraps the device body (`#include \"wholeport/wp_fag_entry.h\"`\n"
        "   + the FAG_LAUNCH macro → `extern \"C\" __global__ __aicore__` launchers); `#endif`.\n"
        "   **FORBIDDEN: `#define __NPU_ARCH__ 3510`** (the canon-0/21 dead-end — it pulls arch35\n"
        "   DEVICE headers into the host pass). NO `#include \"arch35/\"` in `*_kernels.cpp` /\n"
        "   `pybind11.cpp`. `pybind11.cpp` = host-only (no device kernel_operator.h); the DoTiling\n"
        "   above + SelectLauncher + raw `aclrtlaunch_*`; returns dq/dk/dv. `model_new_ascendc.py`\n"
        "   (OL-160 canonical entry, anti-cheat scanners target this name) — `ModelNew.forward(\n"
        "   q,k,v,gy)` with a forward-stat prelude obtaining softmax_max/sum + attention_in via\n"
        "   the vendor forward (the kernel's required inputs); `model.py` = fp64-autograd oracle.\n"
        "\n"
        "## BUILD-DRIFT FIX-CLASS CHECKLIST (the 14 — read each from the compiler error, NEVER\n"
        "guess; the catalog is a sufficient PRE-LOAD so you OBEY them in the authored code and\n"
        "they mostly never fire — the graybox hit only 1 of 14 (std::min) by pre-obeying the rest):\n"
        "  #1 `using namespace commondef;` MUST precede the block includes (blocks use\n"
        "     UNQUALIFIED FagRunInfo/FagConstInfo).\n"
        "  #2 include order: block_vec + block_cube BEFORE kernel.h (block_cube DEFINES the\n"
        "     ARGS_TRAITS macro kernel_base's class-top `ARGS_TRAITS;` expands).\n"
        "  #3 `IS_PSE` is `bool` not `PseTypeEnum` (CUBE_BLOCK_TRAITS_CONST_FIELDS X(IS_PSE,bool)).\n"
        "  #4 `using namespace optiling::fag;` + include `wp_fag_tiling_data.h` before kernel.\n"
        "  #5 `GET_TILING_DATA_WITH_STRUCT` absent in 9.1.0 → manual GM→local word-blit.\n"
        "  #6 `std::min` rejected on dav-c310 device → a local `constexpr` min helper (NOT\n"
        "     `__aicore__` — it's host+device-evaluated, must be host-callable).\n"
        "  #7 fp32 dV/dQ/dK take the MatmulK L0-split-K path whose GET_DKV_L0_SPLIT_K<float>\n"
        "     yields an invalid baseK in the trimmed scaffold → fp16/bf16 use MatmulFull (non-\n"
        "     split, unaffected). fp32 is the dtype-floor → if it blocks, descope as a tracked GAP\n"
        "     (do NOT fast-stop the whole op).\n"
        "  #8 if a dtype is descoped (FAG_LAUNCH removed), also remove its DECL_LAUNCH +\n"
        "     SelectLauncher reference (else link-time undefined symbol).\n"
        "  #9-14 host-tiling (§step 5): maxValidBBLen=0→garbage; coreNum is AIV(×2);\n"
        "     PostParams=0→all-zero output; splitAxis BN2-vs-BN2GS1S2 (§step 4, the decisive\n"
        "     precision bug); the 'no core-fill' clarification (§step 5).\n"
        "  (env, NOT kernel): build_ascendc.py's torch path-probe may `import torch_npu` which\n"
        "  fails on an ABI-mismatched container → use a LOCAL build_ascendc copy with the probe\n"
        "  changed to `importlib.util.find_spec('torch_npu')` + TORCH_DEVICE_BACKEND_AUTOLOAD=0.\n"
        "\n"
        "## FULL SCOPE — DEBUG-ON-TOP (NOT fast-stop; the verified slice is the START, not the end):\n"
        "The VERIFIED slice (manual+graybox) is **BN2 / S≤128 / fp16-bf16 / dense / BNSD / D128**.\n"
        "These are DECLARED GAPs you MUST EXTEND (Edit/Build/Verify from CANN source + KB), not\n"
        "skip:\n"
        "- **S>128 (BN2GS1S2 axis)**: wire the BN2GS1S2 Process branch + the host axis-switch\n"
        "  (splitAxis flips to BN2GS1S2 when isBn2 is false) + Post (fp32-ws→out-dtype reduce).\n"
        "  This closes the S=256 fail. Port the BN2GS1S2 DoSplit/block-split/Post math faithfully.\n"
        "- **fp32**: resolve the MatmulK L0-split baseK in the trimmed scaffold (or widen the\n"
        "  scaffold to carry the split-K path). fp32 is the dtype-floor (err≈vendor floor).\n"
        "- **mask / pse / dropout / other D-buckets / non-BNSD layouts**: the feature blocks are\n"
        "  COPIED (wp_fag attenmask/pse/dropmask + the layout offset compute); wire the gates +\n"
        "  the host params (per the forward FA+X delta model: each X = a stage-delta on the\n"
        "  unchanged skeleton). Dropout = capture the vendor-returned (seed,offset) + replay\n"
        "  Philox (do NOT rewrite the math).\n"
        "**NEVER silently drop a declared combo.** Record an explicit evidence-backed GAP in\n"
        "analysis.md ONLY if a combo is genuinely un-derivable from CANN source + KB after a real\n"
        "debug attempt — fast-stop/GAP is the LAST resort, not the first response.\n"
        "\n"
        "## VERIFY (precision) + PERF (the C2 backward-perf block below carries the FA-grad A/B):\n"
        "Reuse the fa_gqa_grad fp64-autograd CDV harness (model.py oracle): forward inputs\n"
        "(q,k,v) + gy → fp64 autograd.grad over the standard SDPA forward → cast to in-dtype.\n"
        "Compare MINE vs oracle AND vendor `npu_fusion_attention_grad` vs oracle — bit-\n"
        "equivalence = err(mine-ora) ≈ err(vendor-ora) (the dtype FLOOR; ceiling-vs-bug =\n"
        "ceiling, NEVER loosen tol to fake a pass). Determinism ≥5 identical md5. Perf: msprof /\n"
        "torch_npu.profiler device-exclusive vs the vendor (NOT wall-clock — a per-call fwd-stat\n"
        "prelude inflates wall-clock; time ONLY the backward, fwd-stat precomputed OUTSIDE the\n"
        "timed region). The B3.3b finalize SCHEMA CONTRACT (verify_<op>.py stdout summary object\n"
        "with tier1_pass/total/status + performance block) is in the backward-mode brief that\n"
        "composes around this block — honor it.\n"
        "\n"
        "Reference: `kb/target/ascendc/patterns/domains/fa_class_template.md`\n"
        "(P-P103 BACKWARD section) + `CAND-FA-TEMPLATE-GEN-BWD-1` + `CAND-FA-GQA-BWD-1` (the\n"
        "FA-2-backward MATH reference / oracle) + OL-200 (MIX pipeline) + OL-201 (perf caveat)."
    )


def _fa_class_backward_multilaunch_block(op: str, op_class: str) -> str:
    """FA-class BACKWARD DEFAULT brief — the proven MULTI-LAUNCH approach.

    Architecture default (2026-06-20, owner via the C19 finding): for a DENSE FA-grad
    op the multi-launch approach (CAND-FA-GQA-BWD-1) is the PRECISION-CORE-COMPLETE,
    demonstrably-better path — it covers the full scope (fp16/bf16/fp32, S>128) at
    precision 45/45, whereas the fused single-launch stitch is BN2-only (S≤128, no fp32)
    and ~0.2× on small shapes. So an FA-grad op DEFAULTS here; the fused-stitch recipe
    (`_fa_class_backward_stitch_block`) is gated behind an explicit opt-in
    (`_fused_fa_backward_requested`: `fa_backward_arch=="fused"` / `fa_backward_large_s`).

    This brief points the kw at the multi-launch MATH + the OL-200 MIX-pipelining KB;
    the analytic-derive Phase A-D prose that composes after it (BackwardPlugin body) is
    multi-launch-compatible (derive the per-grad math, REAL AscendC, fp64 autograd verify).
    """
    return (
        "# PHASES (FA-class BACKWARD — MULTI-LAUNCH default, the precision-core-complete path)\n"
        "\n"
        f"op `{op}` is an FA-class **BACKWARD** (gradient) op (tags: {op_class!r}).\n"
        "ARCHITECTURE = the proven **MULTI-LAUNCH** FA-grad approach (CAND-FA-GQA-BWD-1), NOT the\n"
        "fused single-launch stitch. RATIONALE (the C19 finding, MEASURED): for a dense FA-grad op\n"
        "the multi-launch is **precision-core-complete** (fp16/bf16/fp32, S>128 — 45/45), while the\n"
        "fused single-launch stitch is BN2-only (S≤128, no fp32) + ~0.2× on small shapes (its\n"
        "large-S perf value is an UNMEASURED hypothesis). So default to multi-launch; the fused\n"
        "stitch is opt-in only (`.opgen_state.json fa_backward_arch=\"fused\"` / `fa_backward_large_s`).\n"
        "\n"
        "## THE GUIDE — read FIRST:\n"
        "`CAND-FA-GQA-BWD-1` (candidates.md — the validated FA-2-backward MATH reference: AIC-only\n"
        "cube `MatmulImpl` GEMMs + AIV-only vec, GM-staged, multi-launch host-serialized; sidesteps\n"
        "the MIX cross-core-sync PB-34/35) + **OL-200** (the MIX_AIC cube/vec software-pipelining\n"
        "perf knowledge, in the C2 backward-perf block below) + the repository's backward\n"
        "verification and performance contracts. Generated deliverables are not references.\n"
        "\n"
        "## FA-2 BACKWARD MATH (the 5 grads — dense N1==N2 is the G=1 special case of GQA):\n"
        "dV = Pᵀ@dO ; dP = dO@Vᵀ ; dS = P∘(dP − rowsum(dP∘P)) ; dQ = (dS@K)·scale ; dK = (dSᵀ@Q)·scale.\n"
        "The `rowsum(dP∘P)` term = softmax-grad. Forward-saved softmax_max/sum + attention_in feed it\n"
        "(the model_new_ascendc.py forward-stat prelude obtains them via the vendor forward).\n"
        "\n"
        "## RECIPE (multi-launch — per CAND-FA-GQA-BWD-1):\n"
        "1. Each of the 5 GEMMs = a SEPARATE AIC-only `MatmulImpl` launch (cube), host-serialized;\n"
        "   the softmax-grad + casts = SEPARATE AIV-only vec launches. NO MIX_AIC_1_2 single-launch,\n"
        "   NO manual CrossCoreSetFlag (that's the fused path's PB-34/35 risk). This is why the\n"
        "   multi-launch covers the full scope cleanly: no cross-core-sync architectural barrier.\n"
        "2. fp32 = the L0-split-K cube path (full scope — multi-launch carries it; 24/25 measured).\n"
        "3. S>128 = no special axis-switch — the multi-launch GM-stages every tile, so large S is the\n"
        "   same kernel with more tiles (the BN2-vs-BN2GS1S2 split that traps the fused path does NOT\n"
        "   exist here).\n"
        "4. Emit the canonical `model_new_ascendc.py` (OL-160, `ModelNew.forward(q,k,v,gy)` + the\n"
        "   forward-stat prelude) + `verify_<op>.py` (the B3.3b schema + cannbot judge, in the body\n"
        "   below) + `perf_ab_profiler.py` (vs `npu_fusion_attention_grad`, P97 profiler, OL-200/201).\n"
        "\n"
        "## DEBUG-ON-TOP: the multi-launch is the proven full-scope path — close the core scope\n"
        "(S>128 + fp32 + dense BNSD/D128) on it. If a combo genuinely can't be derived from\n"
        "CAND-FA-GQA-BWD-1 + the fa_gqa_grad pattern after a real attempt, record an evidence-backed\n"
        "GAP — do NOT silently drop it, do NOT fall back to the fused stitch (that's the wrong arch\n"
        "for dense small-S; only opt into fused for an explicit large-S perf study).\n"
        "\n"
        "Reference: `CAND-FA-GQA-BWD-1` (the multi-launch MATH/oracle) + `CAND-FA-CROSS-BWD-1` /\n"
        "`CAND-FA-SWA-BWD-1/2` (sibling FA-bwd variants) + OL-200 (MIX pipeline) + OL-201 (perf caveat)\n"
        "+ the fa_class_template.md BACKWARD §architecture-choice note (fused vs multi-launch)."
    )


def _fa_ge_host_gen_block() -> str:
    """port_a3 FA GE op_host generation step (flash_attention_score-pbh-1,
    2026-06-11, owner mandate). For port_a3 FA the worker must SHIP a GE
    op_host (`op_host/<op>_def.cpp` / `_infershape.cpp` / `_tiling.cpp`) — and
    it MUST be GENERATED by following GE_HOST_TRANSFORM_RECIPE.md, NOT
    byte-copied from CANN arch35 source.

    Background: 13 port_a3 FA archives shipped GE op_host files byte-for-byte
    identical to CANN source — customers without CANN source can't reproduce
    them. The finalize gate GE_OPHOST_RAW_CANN_COPY now REJECTS such archives
    when the tiling unit lacks the KB `wfh::` shared-layer structure. This
    brief block makes the recipe-driven
    GE-host-gen an explicit deliverable step so the worker assembles it right.

    Returned as a trailing block of the FA-class template-assembly brief
    (which already fires for port_a3 FA). Input provenance §15.2 still
    applies: derive the GE op_host from the arch22 GE op_host input + KB;
    target/prior artifacts, when present, remain logged advisory context.
    """
    return (
        "## GE OP_HOST GENERATION (port_a3 FA — REQUIRED deliverable, flash_attention_score-pbh-1):\n"
        "When this op ships a GE op_host (`op_host/<op>_def.cpp` / `_infershape.cpp` /\n"
        "`_tiling.cpp`), you MUST GENERATE it by FOLLOWING the recipe — do NOT byte-copy CANN\n"
        "arch35 source (the 13-archive regression: byte-identical GE op_host → customer with no\n"
        "CANN source can't reproduce). Recipe:\n"
        "`kb/target/ascendc/patterns/domains/fa_class/templates/op_host/`\n"
        "→ `GE_HOST_TRANSFORM_RECIPE.md` (the arch22→arch35 per-file assembler) + `GE_HOST_TEMPLATE.md`\n"
        "+ skeletons (`flash_attention_score_{def,infershape,tiling}.cpp`) + `wp_fa_host_tiling.h`\n"
        "(the shared `wfh::`/`wp_fa_host::` arch35 tiling logic) + `ge_host_shim.h` (Tier-1 shell).\n"
        "Per-file transform (RECIPE §three transform classes):\n"
        "1. **`infershape.cpp` = CARRY** the A3 (arch22) infershape verbatim (shape inference is\n"
        "   arch-invariant; only the dtype-relation hook gains the fp8→bf16-out branch if the op\n"
        "   declares fp8). Treat any target/prior version as advisory, not source truth.\n"
        "2. **`def.cpp` = CARRY + PATCH** the A3 op IR (input/output/attr names + order + dtype\n"
        "   rows already present), then PATCH ONLY the SOC config string (A5 `ascend910_95`); add\n"
        "   fp8 dtype rows ONLY if the A3 def lacks them AND the target space declares fp8.\n"
        "3. **`tiling.cpp` = REPLACE-HOOK** — this is the load-bearing one. The A3 (arch22) and A5\n"
        "   (arch35 regbase) tiling are DIFFERENT architectures, NOT a line-transform. ASSEMBLE\n"
        "   tiling.cpp from the KB skeleton + the shared layer: `#include \"wp_fa_host_tiling.h\"`\n"
        "   and compute EVERY tiling VALUE by CALLING `wfh::Calc*` (alias `wp_fa_host::Calc*`) — the\n"
        "   SAME shared functions the pybind launch host uses (dBasicBlock←wfh::CalcDBasicBlock,\n"
        "   effSparseMode←wfh::CalcEffSparseMode, SparseTiling←wfh::ComputeSparseTiling,\n"
        "   MultiCoreParams←wfh::SetMultiCoreParamsRegbase, totalWsBytes←wfh::CalcWorkspaceSize,\n"
        "   …). Do NOT inline raw arch35 arithmetic; do NOT byte-copy the arch35 tiling .cpp.\n"
        "RED LINE: host C++ only — NO `#include \"arch35/\"` device headers, NO aclnn/aclop.\n"
        "Section 15.2 holds: derive from arch22 input + this KB recipe; provenance-tracked\n"
        "target/prior context is advisory only and must be independently reverified.\n"
        "FINALIZE GATE: GE_OPHOST_RAW_CANN_COPY inspects only workspace structure and rejects\n"
        "the archive when GE `tiling.cpp` lacks `#include \"wp_fa_host_tiling.h\"` or has zero\n"
        "`wfh::`/`wp_fa_host::` calls. Target comparison, if performed, is provenance-logged\n"
        "advisory analysis and cannot satisfy the source- or target-NPU truth gates.\n"
        "\n"
    )
