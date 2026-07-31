# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Op-class taxonomy — DEPRECATED v3.

Status: this module is being retired in favor of LLM-driven classification
via `/aog-op-classify` skill (P0aak, 2026-05-07). The bench-name-keyed
`OP_TAGS` dict + regex `_SOURCE_SCAN_SIGNATURES` were both Python-side
hardcoded heuristics that didn't generalize beyond benchmark ops — silent
KB-load regression for cross-generation and custom operators.

What replaces it:
- `phase_o17_classify.py` runs `/aog-op-classify` skill in isolated
  subprocess, writes `workspace/<op>/op_classification.json`
- Brief consumers read the JSON instead of calling `lookup()`
- `DEFAULT_KB_SECTIONS` (still used) provides bookshelf-level baseline
- See `src/skills/aog-op-classify/SKILL.md` for procedure

Transition shim: `lookup()` still exists; it now reads
`op_classification.json` if present, else falls back to `DEFAULT_KB_SECTIONS`
only (no more bench-name-keyed dict, no regex source-scan).

The original `OP_TAGS` and `TAG_KB_SECTIONS` dicts are retained below
for emergency rollback only — NOT consulted by `lookup()` in v3.

Codex review C1 (2026-05-04) recommended deterministic Python lookup for
brief construction. The recommendation was sound for the time but did not
account for the project's broader scope (cross-generation + custom ops). The
LLM-driven approach in v3 preserves the "no LLM in brief construction
itself" rule (the orchestrator's brief generation is still pure Python)
while moving the upstream classification to where LLM-knowledge is
actually needed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Tag → KB sections mapping
# ---------------------------------------------------------------------------
# Each tag points to a list of relative paths (under kb/)
# that briefs should ALWAYS load when a tagged op is being processed.
TAG_KB_SECTIONS: dict[str, list[str]] = {
    # === Compute structure ===
    "elementwise": [
        # No special concerns; default safe set is sufficient
    ],
    "broadcast": [
        "patterns/domains/memory_access.md",
    ],
    "reduction": [
        "OPERATIONAL_KNOWLEDGE.md#OL-110",   # cross-N reduction fail-floor
        "OPERATIONAL_KNOWLEDGE.md#OL-114",   # 2-pass tile loop for D > UB
        "OPERATIONAL_KNOWLEDGE.md#OL-102",   # P0acp-followup 2026-05-10: fp16/bf16
                                             # composite ops must replicate CPU's
                                             # per-op rounding chain — was missing
                                             # from reduction tag, caused DS Cumsum
                                             # 28/51 PARTIAL because worker re-derived
                                             # the precision diagnosis instead of
                                             # finding it via brief KB inject.
        "patterns/domains/reduction_quant.md",
    ],
    "segmented-reduction": [
        "OPERATIONAL_KNOWLEDGE.md#OL-110",
        "OPERATIONAL_KNOWLEDGE.md#OL-102",   # same rationale as reduction tag above
        "patterns/domains/reduction_quant.md",
    ],
    "sort-select": [
        "ERROR_CORRECTIONS.md#EC-33",         # Sort 6-arg overload VMS 343
        "OPERATIONAL_KNOWLEDGE.md#OL-83",     # 1-ULP boundary mismatch
        "patterns/unverified/candidates.md#CAND-PP79",  # bf16 mantissa-collision tie cluster
        "patterns/unverified/candidates.md#CAND-PP80",  # T1-vs-CPU triage
    ],
    "scatter-gather": [
        "OPERATIONAL_KNOWLEDGE.md#OL-67",
        "OPERATIONAL_KNOWLEDGE.md#OL-90",
        "patterns/PATTERN_INDEX.md",          # P-P67 atomic-add patterns
    ],

    # === Op families ===
    "normalization": [
        "OPERATIONAL_KNOWLEDGE.md#OL-110",   # output-dtype reduction-order drift
        "OPERATIONAL_KNOWLEDGE.md#OL-111",   # on-device pilot mandate
        "OPERATIONAL_KNOWLEDGE.md#OL-120",   # fp32-internal + CAST_RINT canonical path
        "patterns/unverified/candidates.md#CAND-PP82",  # Direction-4 anti-pattern
    ],
    "softmax": [
        "OPERATIONAL_KNOWLEDGE.md#OL-110",
        "patterns/domains/reduction_quant.md",
    ],
    "fft": [
        "OPERATIONAL_KNOWLEDGE.md#OL-83",     # 1-ULP boundary
        "OPERATIONAL_KNOWLEDGE.md#OL-103",   # transcendental Tier-2
        # DEBT-052 (fp32 unit-ULP limit) — not in KB yet, refer to TECHNICAL_DEBT.md
    ],
    "fused": [
        # Fused ops have multiple compute primitives; tag with their constituents
        "patterns/PATTERN_INDEX.md",          # general pattern lookup
        # FA-class architectural blueprint (user-signed
        # 2026-05-10 via Discord directive "make sure FA can work on the guide of new KB"):
        "patterns/unverified/candidates.md#CAND-FA1",  # cross-core AIC↔AIV pipeline (CrossCoreSetFlag/WaitFlag chain)
        # PB-34 MIX cube+vec (KERNEL_TYPE_MIX_AIC_1_2) V220 deadlock recall-fix (2026-07-16):
        # CAND-FA1 above hands a fused/FA-class op the Pattern-A cross-core-sync recipe but
        # WITHOUT the deadlock root-cause + fix cards, so a CUBE_MIX fa_class op could walk into
        # the MatmulImpl+manual-CrossCore FFTS-slot hang (507014/507015, silent first-launch hang
        # at torch.npu.synchronize()) on V220. Anchor the fix cards alongside CAND-FA1.
        # NOTE (mechanism-honesty): this TAG_KB_SECTIONS dict is RETAINED-FOR-ROLLBACK-ONLY —
        # the live `lookup()` (P0aak, v3) reads op_classification.json `kb_recommendations`
        # (written by the /aog-op-classify skill) + default_kb_sections, and does NOT consult
        # TAG_KB_SECTIONS (see lookup() docstring "OP_TAGS / TAG_KB_SECTIONS ... are NOT
        # consulted"). These entries therefore document the intended fused/attention anchoring
        # (and feed coverage_report / emergency rollback), but the LIVE surfacing path is:
        # (1) the kw forward-FA brief warning block `_fa_assembly_deadlock_warning_block`
        #     (kw_brief_fa.py) which delivers the same cards inline to every forward FA worker;
        # (2) KB_INDEX.md §By Symptom (the MIX cube+vec hang row); (3) default_kb_sections which
        # always loads PLATFORM_BUGS.md + OPERATIONAL_KNOWLEDGE.md. To also drive the LLM
        # classifier's kb_recommendations, the aog-op-classify SKILL.md tag→KB table (fused /
        # attention rows) is the live edit-point — left untouched here (outside this fix's file
        # scope; flagged for review).
        "PLATFORM_BUGS.md#PB-34",             # root cause + Pattern A/B fix (V220-only; benign on V351)
        "OPERATIONAL_KNOWLEDGE.md#OL-275",    # managed-cube KFC lifecycle (hand-rolled multi-cube self-poison)
        "OPERATIONAL_KNOWLEDGE.md#OL-220",    # cube+vec MIX ascendc_library build recipe
        "ERROR_CORRECTIONS.md#EC-68",         # 507015 SetSysWorkspaceForce on ACLRT_LAUNCH MIX workspace
        "patterns/unverified/candidates.md#CAND-FA2",  # online-softmax row-state recurrence (avoid S² materialization)
        "patterns/unverified/candidates.md#CAND-FA3",  # GM ping-pong via modulo-(D+1) slot rotation
        "patterns/unverified/candidates.md#CAND-FA4",  # tree-reduce wide rows via BlockReduceMax/Sum
        "patterns/unverified/candidates.md#CAND-FA5",  # multi-output workspace contract (FA-class auxiliary outputs)
        # multi-CANN cross-learning batch 2026-05-10 (Discord directive: cross-op-evidence
        # accumulation, NOT generate-blind-then-empirically-learn). Codex review on 2026-05-11
        # left these as NEEDS_REVISION pending a5_ops shipped evidence — i.e. an op-gen run
        # that consumes them. They are exposed in kw brief here so the next fused op can
        # cite them; the finalize hook then writes verified_on back when kernel passes:
        "patterns/unverified/candidates.md#CAND-FAG-1",  # 3-kernel split for fp32-workspace gradient accumulation
        # Deterministic backward via coordinate-partitioned core dispatch.
        "patterns/unverified/candidates.md#CAND-FAG-2",
        "patterns/unverified/candidates.md#CAND-FAG-3",  # saved-tensor restore contract for fused bwd
        "patterns/unverified/candidates.md#CAND-FAG-4",  # multi-output backward dispatch (shared recompute, ladder)
        # latent-projection prolog (down × pre-norm × up, cube-vec-cube ladder)
        "patterns/unverified/candidates.md#CAND-MLA-1",
        "patterns/unverified/candidates.md#CAND-MLA-2",  # paged-attention scatter-cache via single DataCopy per token
        "patterns/unverified/candidates.md#CAND-MLA-3",  # whole-VEC barrier for cross-partition stage transitions
        # interleaved-pair RoPE via GatherMask + symmetric Mul(cos)+Mul(sin)
        "patterns/unverified/candidates.md#CAND-MLA-4",
        # matmul-library-driven AIC/AIV co-iteration (complement to CAND-FA1)
        "patterns/unverified/candidates.md#CAND-NSA-1",
        "patterns/unverified/candidates.md#CAND-NSA-2",  # power-of-2 tree-reduce across head-pack axis via strided Add
        "patterns/unverified/candidates.md#CAND-NSA-3",  # overlapping-window weighted-sum (1-D convolution shape)
        # Phase-multiplexed single UB region via typed GetWithOffset views.
        "patterns/unverified/candidates.md#CAND-NSA-4",
        "patterns/unverified/candidates.md#CAND-RAU-1",  # online-softmax 2-input symmetric merge (ring/tree fold)
        # packed-variable-length (TND/varlen) traversal w/ cumulative-offset
        "patterns/unverified/candidates.md#CAND-RAU-2",
        # Per-row scale broadcast via non-zero src1Stride in BinaryRepeatParams.
        "patterns/unverified/candidates.md#CAND-RAU-3",
        # Concatenated-pair UB layout for 2-input binary-reduction kernels.
        "patterns/unverified/candidates.md#CAND-RAU-4",
    ],
    "stateful-cache": [
        "OPERATIONAL_KNOWLEDGE.md#OL-88",     # PA_BLK reference-side race
    ],

    # W12 (2026-05-12, ROADMAP §1.5): arch22→arch35 port mode. Tag emitted by
    # phase_o17_classify when the op-class is `a3_to_a5_port` (i.e. when
    # /ascendc-op-gen was invoked with --port-a3 — see W1). Loads the 4
    # KB entries written in W8-W11 commit `267667a`.
    "a3_to_a5_port": [
        "ops_nn_layout/ops_nn_a5_artifact_layout.md",            # W8 — artifact schemas + build
        "OPERATIONAL_KNOWLEDGE.md#OL-131",                       # W9 — cross-op router pattern
        "patterns/domains/platform_compat.md#P-P90",             # W10 — V220→V351 surgical strip
        "hardware/target/ascend950pr.md",                        # W11 — ToFloat<> restriction + IsRegbase()
    ],

    # P0aai (2026-05-06): explicit transcendental tag — ops that call AscendC's
    # transcendental primitives (Tanh, Sigmoid, Erf, Exp, Log, Sin, Cos) directly.
    # Default `elementwise` tag has empty KB-sections and missed the per-primitive
    # precision evidence (PB-24/25 = bimodal Tanh, 2-ULP Sigmoid). Without this
    # tag, kw on op#1 1_GELU loaded only PLATFORM_BUGS.md broadly without
    # PATTERN_INDEX, and would have missed P-P88 (Cephes-form rewrite). Tag pulls
    # the full chain: per-primitive measurement → recipe → cross-validation.
    "transcendental": [
        "OPERATIONAL_KNOWLEDGE.md#OL-103",   # transcendental Tier-2 (refined 2026-05-06)
        "PLATFORM_BUGS.md",                   # PB-24/25 measurement (already default; explicit here for emphasis)
        "patterns/PATTERN_INDEX.md",          # P-P88 Cephes-form rewrite recipe
    ],

    # === Workflow class ===
    "loss-bwd": [
        "OPERATIONAL_KNOWLEDGE.md#OL-89",     # Python decomposed truth ref
        "OPERATIONAL_KNOWLEDGE.md#OL-110",
        # Mode2-FINALIZED (main 2026-06-18): weight-grad-fp32 is a BACKWARD rule (mixed
        # weight/activation params, weight_type=fp32: SSM/scan/attention bwd). KEPT on loss-bwd —
        # TAG_KB_SECTIONS has no separate GRADIENT/backward tag, loss-bwd is the only backward-
        # semantic tag and selective_scan-bwd carries it; adding a new tag would need classifier
        # wiring (out of scope for a KB-injection entry). Narrowing is via the entry's own
        # applies_to (op_class=scan/SSM/linear-recurrent backward) — coarse tag selects, applies_to
        # scopes; an unrelated loss-bwd op that receives it reads applies_to and skips (token cost
        # only, not a correctness risk).
        # weight_type=fp32 grads returned fp32, not activation dtype
        "patterns/unverified/candidates.md#CAND-SSM-BWD-WEIGHTGRAD-FP32",
        # Scan/SSM bwd: cooperative parallel-prefix SIMT (not naive-SIMT/SIMD),
        # applies_to=scan/SSM/linear-recurrent bwd.
        "patterns/unverified/candidates.md#CAND-SSM-BWD-COOPSIMT-PERF",
        "OPERATIONAL_KNOWLEDGE.md#OL-227",   # gated-backward fp16 dtype-range edge classification
        "OPERATIONAL_KNOWLEDGE.md#OL-228",   # verify hand-derived backward vs pure-fp64 forward
    ],
    "optimizer-update": [
        "OPERATIONAL_KNOWLEDGE.md#OL-68",     # case A — torch_npu deprecated
        "OPERATIONAL_KNOWLEDGE.md#OL-118",   # output-dtype rule
    ],

    # === Reference-modeling concerns ===
    "reference-ub": [
        "OPERATIONAL_KNOWLEDGE.md#OL-85",     # logic-first anti-overfit
        "OPERATIONAL_KNOWLEDGE.md#OL-104",   # CPU-serial alignment
    ],
    "path-a-cpu-truth": [
        "OPERATIONAL_KNOWLEDGE.md#OL-68",
        "OPERATIONAL_KNOWLEDGE.md#OL-89",
        "OPERATIONAL_KNOWLEDGE.md#OL-118",
    ],
}

# Default safe set always loaded for every op (regardless of tags).
#
# P0aaj (2026-05-06): widened from 4 → 7 entries after self-retrospective on
# P0aai. Original 4-entry default + benchmark-name-hardcoded OP_TAGS combined
# meant non-benchmark ops (cross-generation, custom user ops, sparse-attention,
# anything outside the original static dictionary fell into untagged fallback and
# got NEITHER the per-class KB sections (transcendental, normalization,
# reduction, etc.) NOR the pattern library. Silent degradation for the
# project's cross-generation use case.
#
# Fix here: pull the load-bearing pattern library + per-domain pattern files
# + ASCENDC_API_CATALOG into the default. Now any op — whether explicitly
# tagged or untagged-fallback — gets enough KB to find pattern entries by
# Glob/grep when the per-class tag isn't set.
#
# Per-tag KB-sections still useful for FOCUSED loads (e.g. `transcendental`
# → OL-103 + PB-24/25 specifically), but now ride on top of a base that's
# already pattern-aware.
# Target-INDEPENDENT base — same on every backend.
# P88 (2026-05-15) KB reorg: paths updated to new shared/ + target/ascendc/ layout.
_DEFAULT_KB_SECTIONS_BASE: list[str] = [
    "KB_INDEX.md",                                         # navigation index (stays at root)
    "shared/ALWAYS_LOADED_RULES.md",                       # OL-* always-loaded rules
    "target/ascendc/OPERATIONAL_KNOWLEDGE.md",             # generalist scan (workers grep)
    "target/ascendc/PLATFORM_BUGS.md",                     # always relevant (PB-24/25 etc.)
    "target/ascendc/API_CATALOG.md",                       # primitive catalog
    "target/ascendc/patterns/PATTERN_INDEX.md",            # pattern library — P-P88 etc.
]

# P0abj (2026-05-08): target-aware hardware-spec dispatch. Pre-fix, the
# default manifest hardcoded `hardware/target/ascend950pr.md` regardless of
# TARGET — A3/A2 op-gen on DS env was loading A5 specs (UB size, AIV count,
# atomics, register file all wrong for V220). DS-flagged when reviewing
# whether ascend950pr.md could be removed.
TARGET_HW_SPEC_MAP: dict[str, str] = {
    "a5": "hardware/target/ascend950pr.md",   # Ascend950PR — V351 / arch35
    "a3": "hardware/target/ascend910c.md",    # Ascend910 V220 single-die (910C)
    "a2": "hardware/target/ascend910b.md",    # Ascend910 V220 single-die (910B)
}


def default_kb_sections(target: str = "a5") -> list[str]:
    """Return the default KB-manifest list for the given target chip.

    target: "a5" | "a3" | "a2" (case-insensitive). DS-env variants like
    "a3-ds" / "a2-ds" normalize via `.rstrip("-ds")` per AscendCEnv
    convention (DS isolation suffix doesn't change hardware family).

    Falls back to A5 for unknown targets (warn-don't-error policy:
    op-gen on a brand-new chip should still proceed with the closest
    spec rather than miss an entry from the manifest).
    """
    norm = (target or "a5").lower()
    if norm.endswith("-ds"):
        norm = norm[:-3]
    hw_path = TARGET_HW_SPEC_MAP.get(norm, TARGET_HW_SPEC_MAP["a5"])
    return list(_DEFAULT_KB_SECTIONS_BASE) + [hw_path]


# P0abj followup (2026-05-09): fail-fast on missing KB-manifest files.
# User flagged: if user runs op-gen on TARGET=a3 but `ascend910c.md` is
# missing on disk, the agent silently 404s mid-Read; orchestrator gets
# unhelpful agent-side error after $$$ already spent. Better: raise at
# brief-construction time with a precise actionable message.
# Resolution: kb/<path> for each manifest entry.

# op_taxonomy.py lives at <plugin_root>/engine/src/scripts/orchestrator/briefs/.
# 2026-07-05: KB relocated to <plugin_root>/kb/; parents[5] == plugin_root.
_REFERENCES_ROOT = Path(__file__).resolve().parents[5] / "kb"


class KBManifestMissingError(FileNotFoundError):
    """KB-manifest references a file that doesn't exist on disk.

    Raised at brief-construction time so failures surface before any
    LLM agent spawns and before any A5/A3/A2 deploy is attempted.
    Message includes the missing path AND the references-root so the
    user can either restore the file, fix a typo, or update the
    manifest mapping.
    """


# P88 (2026-05-15) KB reorg: legacy bare-name path mapping. Many OP_TAGS
# entries use bare names like `OPERATIONAL_KNOWLEDGE.md#OL-110` that
# pre-date the KB reorganization. Rather than rewrite 50+ entries inline,
# resolve via this map at validation/render time. New entries SHOULD
# use the full new path; legacy entries stay valid.
_LEGACY_PATH_REWRITE: dict[str, str] = {
    "OPERATIONAL_KNOWLEDGE.md": "target/ascendc/OPERATIONAL_KNOWLEDGE.md",
    "PLATFORM_BUGS.md": "target/ascendc/PLATFORM_BUGS.md",
    "ERROR_CORRECTIONS.md": "target/ascendc/ERROR_CORRECTIONS.md",
    "ASCENDC_API_CATALOG.md": "target/ascendc/API_CATALOG.md",
    "ASCENDC_LANGUAGE_REFERENCE.md": "target/ascendc/LANGUAGE_REFERENCE.md",
    "ASCENDC_SIMT_PATTERNS.md": "target/ascendc/SIMT_PATTERNS.md",
    "ASCENDC_SIMD_DEVELOPMENT_REFERENCE.md": "target/ascendc/SIMD_DEVELOPMENT_REFERENCE.md",
    "ASCEND_OP_PRECISION_STANDARD_v2.1.md": "target/ascendc/PRECISION_STANDARD_v2.1.md",
    "ROOFLINE_MODEL.md": "target/ascendc/ROOFLINE_MODEL.md",
    "MSPROF_AGENT_GUIDE.md": "target/ascendc/MSPROF_AGENT_GUIDE.md",
    "SIMT_VS_SIMD_DECISION.md": "target/ascendc/SIMT_VS_SIMD_DECISION.md",
    "patterns/": "target/ascendc/patterns/",
    # fa_class/ tree (fused-attention KB + device-evidence) lives under
    # target/ascendc/. Normalize bare FA-class references the same way P-P88
    # handled bare pattern IDs.
    "fa_class/": "target/ascendc/fa_class/",
    "ALWAYS_LOADED_RULES.md": "shared/ALWAYS_LOADED_RULES.md",
    "ANTI_PRESSURE_PROTOCOLS.md": "shared/ANTI_PRESSURE_PROTOCOLS.md",
    "BENCHMARK_METHODOLOGY.md": "shared/BENCHMARK_METHODOLOGY.md",
    "GATE_CONTRACT.md": "shared/GATE_CONTRACT.md",
    "REGRESSION_METHODOLOGY.md": "shared/REGRESSION_METHODOLOGY.md",
    "OUTPUT_PROJECT_LAYOUT.md": "shared/OUTPUT_PROJECT_LAYOUT.md",
    "exploration/": "shared/exploration/",
    "retrospectives/": "shared/retrospectives/",
    "ops_nn_layout/": "plugin-scope/port_a3/ops_nn_layout/",
}


# Bare pattern-ID form (e.g. `P-P88`, `P-P113`) — resolves to PATTERN_INDEX.md.
_PATTERN_ID_RE = re.compile(r"^P-P\d+$")


def resolve_legacy_kb_path(path_part: str) -> str:
    """Map legacy bare KB paths to new P88 reorganized layout. Returns
    the new path if `path_part` matches a known legacy entry, else
    returns `path_part` unchanged (passthrough for already-correct paths
    or unknown entries).
    """
    # Exact filename match
    if path_part in _LEGACY_PATH_REWRITE:
        return _LEGACY_PATH_REWRITE[path_part]
    # Directory-prefix match (e.g., `patterns/PATTERN_INDEX.md` →
    # `target/ascendc/patterns/PATTERN_INDEX.md`)
    for legacy_prefix, new_prefix in _LEGACY_PATH_REWRITE.items():
        if legacy_prefix.endswith("/") and path_part.startswith(legacy_prefix):
            return new_prefix + path_part[len(legacy_prefix):]
    # Bare pattern IDs (e.g. `P-P88`) emitted as classifier kb_recommendations
    # resolve to the pattern index that carries their inline content — the same
    # way `OL-`/`PB-` IDs resolve via `FILE#anchor`. Without this a bare pattern
    # ID is treated as a literal filename and fails the manifest existence check
    # (root-caused 2026-07-15 on a transcendental op whose
    # classification recommended P-P88, whose text lives inline in PATTERN_INDEX).
    if _PATTERN_ID_RE.match(path_part):
        return "target/ascendc/patterns/PATTERN_INDEX.md"
    return path_part


def validate_manifest_paths(
    sections: list[str],
    *,
    references_root: Optional[Path] = None,
) -> None:
    """Raise KBManifestMissingError if any manifest entry is missing on disk.

    Used as a precondition in `kb_manifest_block` (brief-construction
    boundary). Per-target hw-spec dispatch (P0abj) means this gate
    also catches "user wants A3 op-gen but ascend910c.md was deleted"
    upfront rather than during agent runtime.

    Args:
        sections: list of relative paths under kb/
        references_root: override for tests; defaults to repo's
            kb/

    Raises:
        KBManifestMissingError: first missing path encountered, with
        a multi-line message naming the file + the resolution root +
        the full manifest (for context).
    """
    root = references_root if references_root is not None else _REFERENCES_ROOT
    missing: list[str] = []
    for s in sections:
        # Strip section anchor (`#anchor`) — those are intra-doc
        # references, not separate files.
        path_part = s.split("#", 1)[0]
        if not path_part:
            continue
        # P88 KB reorg: rewrite legacy bare names to new layout before check
        resolved = resolve_legacy_kb_path(path_part)
        full = root / resolved
        if not full.exists():
            missing.append(path_part)
    if missing:
        raise KBManifestMissingError(
            f"KB manifest references {len(missing)} file(s) that don't "
            f"exist under {root}:\n"
            + "\n".join(f"  - MISSING: {p}" for p in missing)
            + f"\n\nFull manifest ({len(sections)} entries):\n"
            + "\n".join(f"    {s}" for s in sections)
            + "\n\nResolution: either (a) restore the missing file(s), "
            "(b) fix the typo in op_taxonomy.TARGET_HW_SPEC_MAP / "
            "DEFAULT_KB_SECTIONS_BASE / classification kb_recommendations, "
            "or (c) if the file was intentionally removed, also remove "
            "the manifest entry that references it."
        )


# Back-compat: callers that import `DEFAULT_KB_SECTIONS` get the A5 list.
# New callers should use `default_kb_sections(target)` and pass TARGET.
DEFAULT_KB_SECTIONS: list[str] = default_kb_sections("a5")


# ---------------------------------------------------------------------------
# Op → tags mapping
# ---------------------------------------------------------------------------
# Initial coverage spans representative L1+L2+L3 operators. Untagged ops fall
# back to default safe set. Multi-tag per op encouraged.
OP_TAGS: dict[str, list[str]] = {
    # === L1 — Elementary ===
    "1_GELU": ["elementwise", "transcendental", "reference-ub"],
    "1_RotaryMul": ["elementwise", "stateful-cache"],
    "2_GroupNormSwish": ["normalization", "fused", "transcendental"],
    "2_SwiGLU": ["elementwise", "fused", "transcendental"],
    "3_Add": ["elementwise"],
    "3_AdvanceStepFlashattn": ["fused", "stateful-cache"],
    "3_FusionAttention": ["fused", "softmax", "transcendental"],
    "4_Abs": ["elementwise"],
    "5_Cumsum": ["reduction"],
    "5_MoeInitRouting": ["scatter-gather"],
    "8_Sort": ["sort-select"],
    "9_TopK": ["sort-select"],
    "9_TopKTopP": ["sort-select", "fused", "transcendental", "reference-ub"],

    # === L2 — Composite ===
    "10_LayerNorm": ["normalization"],
    "10_SwigluQuant": ["fused", "scatter-gather"],
    "11_DequantSwigluQuant": ["fused", "scatter-gather"],
    "11_GroupNorm": ["normalization"],
    "12_KvRmsnormRopeCache": ["fused", "stateful-cache", "normalization"],
    "12_Permute": ["broadcast"],
    "13_Cat": ["broadcast"],
    "14_AdaptiveInstanceNormalization2DBackward": ["normalization", "loss-bwd"],
    "14_Split": ["broadcast"],
    "15_AttentionSoftmaxWithSoftcappingAndDropout": ["softmax", "fused"],
    "15_Pad": ["broadcast"],
    "16_Batched2DRopePositionEncodingBackward": ["loss-bwd", "stateful-cache"],
    "16_Repeat": ["broadcast"],
    "17_AdamW": ["optimizer-update", "path-a-cpu-truth"],
    "17_EmbeddingWithInitialLayernormBackward": ["loss-bwd", "normalization"],
    "18_FusedAddRmsnorm": ["normalization", "fused"],
    "18_Index": ["scatter-gather"],
    "19_FusedResidualRmsNormBackward": ["normalization", "loss-bwd"],
    "19_IndexPut": ["scatter-gather"],
    "20_FusedRopeWithQkNormAndKvCacheUpdate": ["fused", "stateful-cache", "normalization"],
    "20_Gather": ["scatter-gather"],
    "21_GaussianTopkSparseActivation": ["sort-select", "elementwise"],
    "21_Scatter": ["scatter-gather"],
    "22_HybridAttentionMaskPreparation": ["fused"],
    "22_Nonzero": ["scatter-gather", "reduction"],
    "23_HyenaFftSizePaddingRfft": ["fft", "fused"],
    "23_RepeatInterleave": ["broadcast"],
    "24_EmbeddingDenseBackward": ["loss-bwd", "scatter-gather"],
    "24_KvCacheUpdateWithRopeBackward": ["loss-bwd", "stateful-cache"],
    "25_MaskedSoftmaxWithAttentionDropoutBackward": ["softmax", "loss-bwd"],
    "25_NLLLoss": ["loss-bwd", "reduction"],
    "26_AvgPool3d": ["reduction", "broadcast"],
    "26_MoeGroupScoreAggregationAndMasking": ["scatter-gather", "reduction"],
    "27_MaxPool3d": ["reduction", "broadcast"],
    "27_MultiMaskAttentionAggregation": ["reduction", "fused"],
    "28_Interpolate": ["broadcast"],
    "28_MultimodalRopePositionComputationWithGridBasedIndexing": ["fused", "stateful-cache"],
    "29_DynamicQuant": ["scatter-gather", "elementwise"],
    "29_TanhGatedResidualAddBackward": ["loss-bwd", "fused"],
    "30_NMS": ["sort-select"],
    "30_TimeDecayExponentialStabilization": ["elementwise", "reduction"],
    "31_IOU": ["reduction"],

    # === Path-A (OL-68 case A) — torch_npu deprecated, use CPU truth ===
    "8_QuantScatter": ["path-a-cpu-truth", "scatter-gather"],
    "7_MoeGatingTopKSoftmax": ["path-a-cpu-truth", "softmax", "scatter-gather"],
    "6_Histc": ["path-a-cpu-truth", "reduction"],
    "7_Sum": ["reduction"],
    "6_MoeFinalizeRouting": ["scatter-gather", "fused"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class OpTaxonomy:
    op: str
    tags: list[str]
    kb_sections: list[str]      # all sections to load (default + per-tag, deduped)
    is_untagged_fallback: bool


# P0aaj regex source-scan signatures and `_infer_tags_from_source` were
# REMOVED in P0aak (v3, 2026-05-07). Replaced by LLM-driven classification
# via `/aog-op-classify` skill — see `phase_o17_classify.classify()`.

def lookup(
    op: str,
    workspace: Optional[Path] = None,
    target: str = "a5",
) -> OpTaxonomy:
    """Get KB sections for `op`'s brief — v3 reads `op_classification.json`.

    P0aak (2026-05-07): retired bench-name-keyed `OP_TAGS` lookup + regex
    `_SOURCE_SCAN_SIGNATURES` source-scan. Both were Python-side heuristics
    that didn't generalize. v3 reads the LLM-produced classification artifact
    written by `phase_o17_classify.classify()`.

    P0abj (2026-05-08): target-aware hardware-spec dispatch — `target` arg
    selects `hardware/target/<chip>.md` from TARGET_HW_SPEC_MAP. Pre-fix the
    default was hardcoded to `ascend950pr.md`, so A3/A2 op-gen loaded A5
    hw specs (wrong UB size, AIV count, atomics info). Defaults to "a5" for
    callers that haven't been updated yet.

    Resolution order:
      1. If workspace + workspace/op_classification.json exist → use it
      2. Else: default_kb_sections(target) only, mark is_untagged_fallback=True
         (signal: Phase O1.7 hasn't run, OR running standalone without
         classification — caller should run classify() first or warn)

    OP_TAGS / TAG_KB_SECTIONS / _SOURCE_SCAN_SIGNATURES are NOT consulted.
    They remain in this file for emergency rollback only.

    Args:
        op: op name / workspace dir name
        workspace: path to op's workspace dir; if None, falls through to
            untagged-fallback (no classification possible)
        target: "a5" | "a3" | "a2" (case-insensitive; -ds suffix stripped).
            Selects hardware/target/<chip>.md.

    Returns:
        OpTaxonomy. `tags` are descriptive labels from classification JSON.
        `kb_sections` are merged target-aware DEFAULT + classification's
        `kb_recommendations`.
    """
    is_fallback = False
    tags: list[str] = []
    classification_kb_paths: list[str] = []

    if workspace is not None:
        cls_json = workspace / "op_classification.json"
        if cls_json.exists():
            try:
                data = json.loads(cls_json.read_text())
                tags = list(data.get("op_class_tags", []))
                for rec in data.get("kb_recommendations", []):
                    if isinstance(rec, dict) and "path" in rec:
                        classification_kb_paths.append(rec["path"])
            except (json.JSONDecodeError, OSError):
                # Corrupt classification — treat as missing
                tags = []
                classification_kb_paths = []

    if not tags and not classification_kb_paths:
        is_fallback = True

    sections: list[str] = default_kb_sections(target) + classification_kb_paths
    # Dedup while preserving order
    seen = set()
    deduped = []
    for s in sections:
        if s not in seen:
            seen.add(s)
            deduped.append(s)

    return OpTaxonomy(
        op=op,
        tags=tags,
        kb_sections=deduped,
        is_untagged_fallback=is_fallback,
    )


@dataclass
class LayerSpec:
    """One layer in a fused-op layered implementation plan."""
    layer: int
    name: str
    sub_op: str
    inputs: list[str]
    outputs_added: list[str]
    outputs_placeholder: list[str]  # Layer 1 establishes; subsequent layers fill
    outputs_filled: list[str]  # tensors this layer fills (was placeholder before)
    reference_decomposition: str  # one-line Python expression
    verify_against: str  # "isolated_layer_ref" | "full_fixture"
    optional: bool


@dataclass
class LayeredPlan:
    """Layered implementation plan (Tier 3, P0aau-c35.e)."""
    applicable: bool
    rationale_when_inapplicable: Optional[str]
    layers: list[LayerSpec]


def read_layered_plan(workspace: Path) -> Optional[LayeredPlan]:
    """Read `algorithm_classification` + `layered_implementation_plan` from
    `workspace/op_classification.json`.

    Returns None if:
    - workspace doesn't exist
    - op_classification.json absent or unparseable
    - schema_version is missing/older AND new fields aren't present (backward compat)
    - `algorithm_classification != "fused"` OR
      `layered_implementation_plan.applicable != true`

    When None, callers route op through standard `await_worker` path. When
    non-None LayeredPlan with applicable=True, Stage 2 state machine routes
    through `await_layer_worker` and follows layer-by-layer build sequence.

    P0aau-c35.e (2026-05-09): backward-compatible add. Pre-v3 classifications
    return None gracefully — orchestrator falls back to standard routing.
    """
    if workspace is None or not workspace.exists():
        return None
    cls_json = workspace / "op_classification.json"
    if not cls_json.exists():
        return None
    try:
        data = json.loads(cls_json.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    algo_class = data.get("algorithm_classification")
    plan_dict = data.get("layered_implementation_plan", {})

    if algo_class != "fused" or not plan_dict.get("applicable"):
        return None

    raw_layers = plan_dict.get("layers", [])
    if not isinstance(raw_layers, list) or not raw_layers:
        return None

    parsed_layers: list[LayerSpec] = []
    for raw in raw_layers:
        if not isinstance(raw, dict):
            return None  # malformed — fail-closed (don't silently route)
        try:
            parsed_layers.append(LayerSpec(
                layer=int(raw["layer"]),
                name=str(raw["name"]),
                sub_op=str(raw["sub_op"]),
                inputs=list(raw.get("inputs", [])),
                outputs_added=list(raw.get("outputs_added", [])),
                outputs_placeholder=list(raw.get("outputs_placeholder", [])),
                outputs_filled=list(raw.get("outputs_filled", [])),
                reference_decomposition=str(raw["reference_decomposition"]),
                verify_against=str(raw.get("verify_against", "isolated_layer_ref")),
                optional=bool(raw.get("optional", False)),
            ))
        except (KeyError, TypeError, ValueError):
            return None  # malformed entry — fail-closed

    # Sanity: layers numbered 1..N strictly increasing
    for i, ls in enumerate(parsed_layers, start=1):
        if ls.layer != i:
            return None  # non-canonical layer numbering — fail-closed

    return LayeredPlan(
        applicable=True,
        rationale_when_inapplicable=None,
        layers=parsed_layers,
    )


def all_tags() -> list[str]:
    return sorted(TAG_KB_SECTIONS.keys())


def coverage_report() -> dict:
    """Diagnostics: how many ops covered, untagged tags, etc."""
    n_ops = len(OP_TAGS)
    tag_usage = {t: 0 for t in TAG_KB_SECTIONS}
    for tags in OP_TAGS.values():
        for t in tags:
            if t in tag_usage:
                tag_usage[t] += 1
    unused_tags = [t for t, n in tag_usage.items() if n == 0]
    untagged_dict_keys = [op for op, tags in OP_TAGS.items() if not tags]
    return {
        "n_ops_in_taxonomy": n_ops,
        "n_tags_defined": len(TAG_KB_SECTIONS),
        "tag_usage": tag_usage,
        "unused_tags": unused_tags,
        "ops_with_zero_tags": untagged_dict_keys,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="op_taxonomy — op-class tag lookup")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("lookup", help="show tags + KB sections for an op")
    sp.add_argument("op")

    sp = sub.add_parser("tags", help="list all defined tags")
    sp = sub.add_parser("coverage", help="diagnostics: tag usage + untagged ops")

    args = ap.parse_args()

    if args.cmd == "lookup":
        t = lookup(args.op)
        print(json.dumps({
            "op": t.op,
            "tags": t.tags,
            "kb_sections": t.kb_sections,
            "is_untagged_fallback": t.is_untagged_fallback,
        }, indent=2))
    elif args.cmd == "tags":
        for t in all_tags():
            n = sum(1 for tags in OP_TAGS.values() if t in tags)
            print(f"  {t:30s} ({n} ops)")
    elif args.cmd == "coverage":
        print(json.dumps(coverage_report(), indent=2))
