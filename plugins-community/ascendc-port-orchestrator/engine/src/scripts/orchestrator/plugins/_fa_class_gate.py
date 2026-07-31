# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Shared FA-class structural pre-build gate (Test 5-bis).

Op-class-scoped capability available to both supported mode plugins. Invoke when op_class is FA via
`is_fa_class(op_class)`. Empirically verified 2026-05-26 by cross-agent diff
to PASS an independently authored FlashAttention fixture + FAIL the F10.E.1 #3 monolithic
inline-flag Antipattern A emit.

Architectural placement: FA is an op-class, not a mode. The FA
template-assembly path applies to independently authored FA implementations. This module
sits at the source-mode-agnostic layer so each mode plugin opts in via its own
`is_fa_class(op_class)` gate + brief-injection hooks.

Origin: extracted from the short-lived `port_fa_cv_agent` plugin (deleted
2026-05-26 in same commit) which incorrectly modeled FA as a new mode rather
than an op-class enhancement to existing modes.
"""
from __future__ import annotations
import logging

import re
from pathlib import Path
from typing import Optional


def strip_cpp_comments(src: str) -> str:
    """Strip C/C++ comments so structural-token counts never count mentions
    inside comments.

    A clean e2e emit exposed that a doc comment
    "// hand-off goes through ... CrossCoreSetFlag" inflated check4's flag
    count from the real 4 to 6, almost mis-flagging a correct kernel as
    Antipattern A. A gate counting comment text is the syntactic-decoy trap
    we set out to avoid — counts must reflect emitted code, not prose.

    Block comments → single space (preserves line count is NOT needed here
    since line-count checks read raw files separately). Line comments →
    removed to EOL. Known limit: a `//` inside a string literal is also
    stripped; acceptable — kernel code does not carry the structural tokens
    inside string literals.
    """
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    return src


def pre_build_check_test_5bis(workspace: Path) -> Optional[str]:
    """FA-class structural pre-build gate. Returns None on pass, or a
    `fa_class_structure_violation: <check>: <msg>` string on first failure.

    Ports SKILL Step 1.4 Check 1/3/4/5 (KEPT — the real Antipattern-A
    discriminators) and REPLACES Check 2's syntactic `grep Matmul<` with a
    semantic dispatch-reachability test, because:

    - cv-agent's own proven-good stock FA kernel uses hand-written `Mmad`
      (0 `Matmul<`, 9 `Mmad` direct calls — disk-verified 2026-05-26 by
      an independent cross-agent diff). The old Check 2 `grep Matmul<`
      would FALSELY REJECT the very reference we mimic. The Antipattern
      discriminator is NOT "matmul library vs hand Mmad" — it is cube/vec
      split + WorkspaceQueue encapsulation of cross-flags vs monolithic +
      inline flag-spam.
    - A *syntactic* "flag inside `class WorkspaceQueue`" check is itself
      decoy-able: an empty `class WorkspaceQueue {}` wrapping inline flags
      games it exactly like the POC's empty `MatmulKernel<>` template gamed
      Check 2. So WorkspaceQueue is validated by SEMANTICS (real
      Producer/Consumer Acquire/Release method pairing + ring-slot constant),
      not by class-name presence.

    Verified to PASS the independent fixture (FlashAttentionCube/Vec + real
    WorkspaceQueue, 4 in-WQ flags) and FAIL the F10.E.1 #3 monolithic +
    inline-flag Antipattern A emit.
    """
    kdir = workspace / "kernel"
    if not kdir.is_dir():
        return "fa_class_structure_violation: precond: workspace/kernel/ absent"
    # Recursive: the independently generated whole-port layout keeps its cube/vec
    # engines under `kernel/wholeport/` (wp_block_cube.h, wp_block_vec_base.h,
    # regbase_*). A non-recursive glob silently missed that whole subtree → the
    # gate read cube_classes=0 and rejected valid generated kernels. rglob fixes
    # the detection bug inside the workspace only.
    headers = sorted(kdir.rglob("*.h"))
    srcs = headers + sorted(kdir.rglob("*.cpp"))
    if not srcs:
        return "fa_class_structure_violation: precond: no kernel sources in kernel/"
    per_file = {p: strip_cpp_comments(p.read_text(errors="ignore")) for p in srcs}
    text = "\n".join(per_file.values())

    # ── Independently generated whole-port FA branch ────────────────────────
    # Two valid FA architectures exist in the canonical template:
    #   (1) cv-agent hand-roll: FlashAttention{Cube,Vec} classes + WorkspaceQueue
    #       ring buffer + few cross-flags + small headers.
    #   (2) generated regbase/wholeport layout: large cube/vec engine files
    #       under wholeport/, raw Set/WaitCrossCore (or CrossCoreSetFlag) AIC↔AIV
    #       sync, 2000+ LOC engines (FA is inherently large).
    # The original Checks 1/3/4/5 are calibrated to (1) and wholesale-reject (2).
    # When the kernel uses this generated layout, validate the SEMANTIC intent
    # (real cube matmul present + genuine cube/vec engine separation + a real
    # cross-core sync mechanism) and skip the (1)-specific naming/LOC/flag knobs.
    is_wholeport_layout = (kdir / "wholeport").is_dir() or any(
        p.name.startswith("regbase_") or p.name.startswith("wp_block_")
        for p in headers
    )
    if is_wholeport_layout:
        # KEEP the load-bearing anti-pattern guards (not the cv-agent knobs):
        # (a) real cube matmul present — guards Antipattern B cube-bypass.
        cube_mm = (len(re.findall(r"\bMmad\b", text))
                   + len(re.findall(r"Matmul<|matmul::Matmul|\.IterateAll\b", text)))
        if cube_mm < 1:
            return ("fa_class_structure_violation: wholeport_check2: 0 cube "
                    "matmul primitive (neither Mmad nor Matmul) — cube-bypass")
        # (b) cube path not self-disabled.
        if re.search(r"cube_eligible\s*=\s*false", text) or \
           re.search(r"VEC-only.*NO cube|cube path.*(?:disabled|currently disabled)", text):
            return ("fa_class_structure_violation: wholeport_check2: cube "
                    "path hard-gated off (A-MAJ-7 / OL-188)")
        # (c) genuine cube/vec engine SEPARATION — a file carrying the cube
        #     matmul must be DISTINCT from a file carrying the vec/softmax work.
        cube_files = set()
        vec_files = set()
        for file_path, content in per_file.items():
            if re.search(r"\bMmad\b|Matmul<|matmul::Matmul", content):
                cube_files.add(file_path)
            if re.search(
                r"\bSoftmax\b|SoftMax|\bExp\b|ReduceMax|RowMax|\bvec\b",
                content,
                re.I,
            ):
                vec_files.add(file_path)
        if not cube_files:
            return ("fa_class_structure_violation: wholeport_check1: no cube "
                    "engine file (Mmad/Matmul) — monolithic risk")
        if not (vec_files - cube_files):
            return ("fa_class_structure_violation: wholeport_check1: cube and "
                    "vec work not in separate files — Antipattern A (monolithic)")
        # (d) a real AIC↔AIV cross-core sync mechanism exists (any of the valid
        #     forms — WorkspaceQueue ring OR raw upstream Set/Wait flags).
        has_sync = bool(re.search(r"WorkspaceQueue|SetCrossCore|WaitCrossCore|"
                                  r"CrossCoreSetFlag|CrossCoreWaitFlag|FftsCrossCore",
                                  text))
        if not has_sync:
            return ("fa_class_structure_violation: wholeport_check3: no AIC<->"
                    "AIV cross-core sync primitive found")
        return None
    # ── end generated whole-port branch; original checks below ───────────────

    # Check 1 (KEEP): cube/vec class separation in separate files.
    cube_classes = len(re.findall(r"^\s*class\s+\w*[Cc]ube\b", text, re.M))
    vec_classes = len(re.findall(r"^\s*class\s+\w*[Vv]ec\b", text, re.M))
    cv_files = len([p for p in headers if re.search(r"(?:_cube|_vec)\.h$", p.name)])
    if not (cube_classes >= 1 and vec_classes >= 1 and cv_files >= 2):
        return (f"fa_class_structure_violation: check1: cube/vec split absent "
                f"(cube_classes={cube_classes} vec_classes={vec_classes} "
                f"cv_files={cv_files})")

    # Check 2 REPLACED: semantic cube dispatch-reachability (Test 5-bis).
    if re.search(r"cube_eligible\s*=\s*false", text) or re.search(r"=\s*false\s*&&", text):
        return ("fa_class_structure_violation: check2_test5bis: cube_eligible "
                "hard-gated off (A-MAJ-7 / OL-188)")
    if re.search(r"VEC-only.*NO cube|cube path.*(?:disabled|currently disabled)", text):
        return ("fa_class_structure_violation: check2_test5bis: cube path "
                "self-labeled disabled (A-MAJ-7)")
    cube_mm = (len(re.findall(r"\bMmad\b", text))
               + len(re.findall(r"Matmul<|matmul::Matmul|\.IterateAll\b", text)))
    if cube_mm < 1:
        return ("fa_class_structure_violation: check2_test5bis: 0 cube matmul "
                "primitive (neither Mmad nor Matmul library) — Antipattern B "
                "cube-bypass / OL-188 PURE_VEC_FOR_CUBE_REQUIRED")

    # Check 3 (KEEP + SEMANTIC): WorkspaceQueue real ring buffer, not decoy.
    if "WorkspaceQueue" not in text:
        return ("fa_class_structure_violation: check3: 0 WorkspaceQueue — "
                "required for AIC<->AIV pipeline (Antipattern A risk)")
    has_producer = bool(re.search(r"ProducerAcquire", text)) and \
        bool(re.search(r"ProducerRelease", text))
    has_consumer = bool(re.search(r"ConsumerAcquire", text)) and \
        bool(re.search(r"ConsumerRelease", text))
    ring_evidence = bool(re.search(r"RING_SLOTS|ring_slots|prelaunch", text))
    if not (has_producer and has_consumer and ring_evidence):
        return ("fa_class_structure_violation: check3_semantic: WorkspaceQueue "
                "present but not a real ring buffer "
                f"(producer={has_producer} consumer={has_consumer} "
                f"ring={ring_evidence}) — empty-shell decoy wrapping inline "
                "flags is Antipattern A (same decoy class as MatmulKernel<>)")

    # Check 4 (KEEP): cross-flags inside the (now-confirmed-real) WQ.
    ccsf = len(re.findall(r"CrossCoreSetFlag", text))
    if ccsf > 5:
        return (f"fa_class_structure_violation: check4: {ccsf} CrossCoreSetFlag "
                "calls — inline flag-spam Antipattern A (>5 even with WQ)")

    # Check 5 (KEEP): no monolithic mega-header.
    max_loc = max((len(p.read_text(errors="ignore").splitlines())
                   for p in headers), default=0)
    if max_loc >= 700:
        return (f"fa_class_structure_violation: check5: largest .h is {max_loc} "
                "LOC >= 700 — likely monolithic Antipattern A")

    return None


def finalize_check_tile_size_consistency(workspace: Path, v: dict) -> Optional[str]:
    """FA-class finalize gate: assert the emitted kernel's tile constant equals
    the designer's `tile_level` `block_N`.

    Whitebox root-cause (2026-05-29): the designer emits an authoritative
    `block_N` in `design/tile_level/*.py`, but the LLM translator hand-writes
    the kernel's `FA_BLOCK_N` constant with NO mechanical enforcement — so it
    silently emits the slow verbatim-ported sibling tile (e.g. 64) instead of
    the design's fast 128. This gate is the missing deterministic check.

    Defensive contract: any parse failure / missing input → return None (do NOT
    crash or block). Only returns a violation string when BOTH a design block_N
    and at least one emitted tile constant are parseable AND they disagree. The
    harness routes the violation string to rollback + re-emit, forcing the
    translator to honor the design tile.
    """
    # 1. Locate design tile_level file(s). Absent → gate N/A.
    tile_dir = workspace / "design" / "tile_level"
    if not tile_dir.is_dir():
        return None
    tile_files = sorted(tile_dir.glob("*.py"))
    if not tile_files:
        return None

    # 2. Parse the DESIGN block_N (LAST authoritative assignment across files).
    #    Handle tuple form `block_M, block_N = 64, 128` and single `block_N = 128`.
    design_block_n: Optional[int] = None
    # tuple form: `block_M, block_N = 64, 128` → block_N is the 2nd RHS value.
    # single form: `block_N = 128` (NOT preceded by a `,` so it does not match
    # the tuple form's LHS). A single combined scan in source order lets the
    # LAST operative assignment win regardless of form.
    combined_re = re.compile(
        r"block_M\s*,\s*block_N\s*=\s*\d+\s*,\s*(?P<tup>\d+)"
        r"|(?<![,\w.])block_N\s*=\s*(?P<sgl>\d+)")
    for tf in tile_files:
        skip_current_item = False
        try:
            text = tf.read_text(errors="ignore")
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
        for m in combined_re.finditer(text):
            val = m.group("tup") or m.group("sgl")
            if val is None:
                continue
            try:
                design_block_n = int(val)
            except ValueError:
                pass
    if design_block_n is None:
        return None  # can't enforce what we can't read

    # 3. Parse the EMITTED kernel tile constant(s) from kernel/*.h.
    kdir = workspace / "kernel"
    if not kdir.is_dir():
        return None
    emitted_values: list = []
    const_re = re.compile(
        r"constexpr\s+[\w:]+\s+(?:FA_BLOCK_N|BLOCK_N)\s*=\s*(\d+)",
        re.IGNORECASE)
    for kf in sorted(kdir.glob("*.h")):
        skip_current_item = False
        try:
            ktext = strip_cpp_comments(kf.read_text(errors="ignore"))
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
        for m in const_re.finditer(ktext):
            try:
                emitted_values.append(int(m.group(1)))
            except (ValueError, IndexError):
                pass
    if not emitted_values:
        return None  # no kernel tile constant yet / other gates handle it

    # 4. COMPARE. Any emitted tile constant != design block_N → violation.
    if any(val != design_block_n for val in emitted_values):
        return (
            f"FA-class tile-size mismatch: emitted kernel tile constant(s) "
            f"{emitted_values} != designer tile_level block_N={design_block_n}. "
            f"The designer's tile_level is AUTHORITATIVE (CAND-FA-TILESIZE-1, "
            f"perf-design). Set the kernel tile constant = {design_block_n} and "
            f"re-derive ALL tile-dependent values (inner-loop bounds, tail/mask "
            f"column ranges, ReduceMax/ReduceSum lengths, workspace slot sizes, "
            f"UB buffer sizing) from it (translator SKILL §1.4.6 "
            f"self-consistency). Do NOT keep a verbatim-ported sibling kernel's "
            f"fixed tile size."
        )
    return None


# FA-class shared asset paths (under <plugin_root>/kb/target/ascendc/fa_class/).
# Used by mode plugins' kw_brief_phase_a / pp_brief_phase_block when op_class is FA.
# 2026-07-05: KB relocated to <plugin_root>/kb/. _fa_class_gate.py lives at
# <plugin_root>/engine/src/scripts/orchestrator/plugins/, so parents[5] == plugin_root.
_FA_CLASS_ASSETS = Path(__file__).resolve().parents[5] / "kb" / "target" / "ascendc" / "fa_class"


def fa_class_assets_root() -> Path:
    """Path to shared FA-class brief assets and templates."""
    return _FA_CLASS_ASSETS


def fa_class_brief_ascendc() -> str:
    """Standard FA-class kw_brief content directive — points kw at the shared
    ascendc_agent.md prompt. Mode plugins return this from kw_brief_phase_a
    when is_fa_class(op_class).
    """
    return (
        "FA-CLASS ASCENDC-AGENT DIRECTIVE (shared op-class brief, per "
        "src/scripts/orchestrator/plugins/_fa_class_gate.py): kw acts as "
        "cv-agent's ascendc-agent for FA-class ops. READ FIRST: "
        "kb/target/ascendc/fa_class/ascendc_agent.md + "
        "cv_lowering.md + cross_core_sync.md + api_mapping.md. Hard constraints "
        "(verified 16/16 on an independently authored V220 fixture): (1) cube.h/vec.h file split "
        "MANDATORY, no monolithic kernel; (2) KERNEL_TYPE_MIX_AIC_1_2 (Attention "
        "requirement); (3) in-loop T.set_cross_flag → WorkspaceQueue ring "
        "buffer with Producer/Consumer Acquire/Release pairs; file-scope inline "
        "CrossCoreSetFlag in tile loop = Antipattern A (FORBIDDEN, F10.E.1 #3 / "
        "507015); (4) use AscendC::Mmad directly where the schedule requires cube matmul (matmul library NOT "
        "required for FA-class — overrides global SKILL Step 1.4.2); (5) no "
        "host-side compute/pad/align in model_new_ascendc.py; tail blocks "
        "kernel-internal. FA-class ops are built by TEMPLATE-ASSEMBLY (owner "
        "2026-06-07): assemble a self-contained arch35 kernel from the arch22 "
        "algorithm spec + the codified KB templates (P-P103) — there is no "
        "separate design provider, so derive the tile schedule directly and do "
        "NOT wait on / emit a design-absent handoff. Pre-build gate: "
        "plugins._fa_class_gate.pre_build_check_test_5bis fires before compile."
    )


def fa_class_brief_debugger() -> str:
    """Standard FA-class pp_brief content directive — pp acts as
    ascendc-debugger. Returns from mode plugin's pp_brief_phase_block when
    is_fa_class(op_class).
    """
    return (
        "FA-CLASS ASCENDC-DEBUGGER DIRECTIVE (shared op-class brief): for this "
        "FA-class precision failure, use cv-agent's debugger principles: "
        "(1) evidence-driven repair — every fix needs a concrete reproducible "
        "signature + at least one failed hypothesis before declaring root "
        "cause (P5 anti-pressure); (2) tensor-dump bisection — instrument "
        "intermediate buffers (workspace_s after Cube Q·K^T, workspace_p after "
        "Vec softmax, workspace_o after Cube P·V) to localize which stage "
        "diverges; (3) do NOT edit model.py / "
        "3_FusionAttention.json / manifest.json — reference + "
        "fixture are IMMUTABLE; (4) do NOT blame hardware / API "
        "without tensor-dump evidence + failed-fix attempt; (5) exit when "
        "PASS_T1 achieved OR no new failure information emerges (prevents "
        "busywork). Reference: kb/target/ascendc/fa_class/"
        "ascendc_agent.md (cube/vec split + WorkspaceQueue still apply during "
        "repair — never revert to monolithic + inline flag)."
    )
