# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Phase 1 of /aog-prior-art-verify — detect arch35 prior-art candidates.

Detection sources (in trust order, HIGH → LOW):
  1. upstream_arch35        : <port_source>/op_kernel/arch35/*.{h,cpp}
                              (OL-141 Mode A — target-arch source, opt-in only)
  2. upstream_shared_common : <port_source>/../<family>_common/op_kernel/arch35/<op>_*
                              (OL-141 Mode B — target-arch source, opt-in only)
  3. upstream_v220_entry    : <port_source>/op_kernel/<op>.cpp                              (V220-pure entry, preferred under default-OFF)
  4. upstream_apt           : <port_source>/op_kernel/<op>_apt.cpp                          (cross-arch dispatcher — Mode A connector OL-132; falls back when no plain <op>.cpp)
  5. workspace_staged       : workspace/<op>/kernel/  (pre-staged by a human)
  6. workspace_external     : workspace/<op>/.prior_art_external/

Bug A (2026-05-23): Hybrid B+A fix for apt.cpp prestage XOR.
  - Hybrid B: when both <op>.cpp AND <op>_apt.cpp exist under default-OFF
    (OPGEN_PRESTAGE_ARCH35 unset/0), prefer <op>.cpp — V220-only entry has no
    arch35/ #includes so worker can stage it directly without modification.
    apt detector is suppressed in this case (avoids feeding the cross-arch
    dispatcher to default-OFF prestage which would trip ARCH35_WRAP_CHEAT).
  - Hybrid A: when ONLY <op>_apt.cpp exists (no plain <op>.cpp) under
    default-OFF, apt detector returns None (no prestage); worker hand-authors
    dispatcher from V220 algorithm source headers in op_kernel/*.h top level.
    Avoids the group_norm_silu-class await_user_decision lockup where apt.cpp
    references arch35/* the harness refuses to prestage.
  - Under OPGEN_PRESTAGE_ARCH35=1 (opt-in): legacy behavior — apt preferred
    over plain (cross-arch dispatcher gets the prestage slot).

OL-141 Mode A vs Mode B (2026-05-14, adaptive_avg_pool3d witness):
  - Mode A: op's own `op_kernel/arch35/` is populated (ada_layer_norm, fused_quant_mat_mul)
  - Mode B: op's own arch35/ is EMPTY (or absent); siblings `<family>_common/op_kernel/arch35/`
    contain the shared kernel files prefixed `<op_name>_*`. Op-specific `<op>_apt.cpp` then
    `#include "../<family>_common/arch35/..."`. First witness: adaptive_avg_pool3d (pooling
    family shares pooling/adaptive_pool3d_common/op_kernel/arch35/).

Usage:
  python3 scan_prior_art.py --op <name> --port-source <path> --workspace <dir>
  python3 scan_prior_art.py --batch <ops_yaml> --port-source-root <root>

The `--batch` form fans out across the configured arch22→arch35 migration ops and prints a status
table; the per-op `--op` form is what phase_o25_a3_ref invokes.

This module is pure-function + side-effect-isolated: scan() returns a dict,
write_scan_result() is the only file write.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional


_TRUST_ORDER = {
    "upstream_arch35": "HIGH",
    "upstream_shared_common": "HIGH",
    "upstream_v220_entry": "HIGH",  # Bug A (2026-05-23): V220-pure entry, default-OFF preferred
    "upstream_apt": "HIGH",
    "upstream_ascend950_config": "HIGH",
    "upstream_op_def": "HIGH",
    "upstream_apt_suppressed_apt_only_default_off": "LOW",  # informational, not a prestage source
    "workspace_staged": "MEDIUM",
    "workspace_external": "LOW",
}


def _hash_file(p: Path) -> str:
    """Return the full SHA-256 used to bind scan and staging objects."""
    h = hashlib.sha256()
    try:
        h.update(p.read_bytes())
    except OSError:
        return ""
    return h.hexdigest()


def _file_records(files: list[Path]) -> list[dict]:
    """Describe the exact files a later stage is authorized to read.

    ``files``/``hashes`` are retained for report compatibility.  This richer
    list removes path ambiguity (notably for multiple ``*_common`` siblings)
    and gives staging a closed, digest-bound input set instead of asking it to
    rescan a directory that may have changed since Phase 1.
    """
    return [
        {
            "path": str(path),
            "sha256": _hash_file(path),
        }
        for path in files
    ]


def _count_lines(files: list[Path]) -> int:
    n = 0
    for f in files:
        try:
            with f.open("rb") as fh:
                n += sum(1 for _ in fh)
        except OSError:
            pass
    return n


# ── DEBT-165 (2026-06-21): source-arch completeness (architecture-based) ─────
# A port is a SOURCE-arch → TARGET-arch migration (V220→V351 today; arch22→V351
# tomorrow). LEGIT = extract the real algorithm from the SOURCE-arch and re-derive
# the target kernel. CHEAT = the candidate has NO source-arch algorithm — its kernel
# entry is a pure `#include "<target-arch>/"` dispatch shell whose only implementation
# is the target-arch vendor answer the customer lacks (deformable_offsets: a 39-line
# `<op>_apt.cpp` that #includes only arch35/). Naming is architecture-based, never
# product-named (arch35 = V3xx target family; never "A5"/"950PR").
#
# Framework headers are not algorithm source; target-arch impl dirs are the answer,
# not extractable source. ANY non-framework, non-target-arch include (atvoss/* the
# arch-invariant elementwise lib, op_kernel/<op>_impl.hpp, sibling V220 headers) is a
# real source-arch algorithm → COMPLETE. gelu (apt-only but includes atvoss/*) = legit;
# deformable_offsets (apt-only, includes ONLY arch35/) = pure shell.
_FRAMEWORK_INCLUDE_MARKERS = ("kernel_operator.h", "kernel_tiling/")
# Target-arch (V3xx) vendor-impl subdirs. Extend when a new target arch is added.
_TARGET_ARCH_DIR_MARKERS = ("arch35/", "arch34/", "arch33/", "arch31/", "arch30/")


def _assess_source_arch_complete(port_source: Optional[Path], op: str) -> tuple[bool, str]:
    """Architecture-based source-completeness verdict (DEBT-165). Returns
    (complete, reason). PERMISSIVE — only a candidate with NO extractable source-arch
    algorithm (a pure target-arch dispatch shell) is flagged incomplete."""
    if port_source is None:
        return True, "no port_source — completeness check N/A (out of scope)"
    kdir = port_source / "op_kernel"
    if not kdir.is_dir():
        return True, "no op_kernel/ dir — not a target-arch dispatch shell (out of scope)"
    # 1. plain <op>.cpp = source-arch entry with a real algorithm → complete.
    if (kdir / f"{op}.cpp").is_file():
        return True, f"source-arch entry op_kernel/{op}.cpp present (real algorithm to extract)"
    # 2. no target-arch impl dir at all → source-arch-pure (nothing to copy) → complete.
    if not any((kdir / d.rstrip("/")).is_dir() for d in _TARGET_ARCH_DIR_MARKERS):
        return True, "no target-arch impl dir under op_kernel/ — source-arch-pure"
    # 3. target-arch present, no plain <op>.cpp: inspect kernel entry/headers for a
    #    NON-framework, NON-target-arch algorithm include = a real source-arch algorithm.
    inc_re = re.compile(r'#include\s+"([^"]+)"')
    for f in list(kdir.glob("*.cpp")) + list(kdir.glob("*.h")) + list(kdir.glob("*.hpp")):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for inc in inc_re.findall(text):
            low = inc.lower()
            if any(fw in low for fw in _FRAMEWORK_INCLUDE_MARKERS):
                continue
            if any(low.startswith(t) or ("/" + t) in low for t in _TARGET_ARCH_DIR_MARKERS):
                continue  # target-arch include = the vendor answer, not extractable source
            return True, (f"source-arch algorithm available: {f.name} references non-target-arch "
                          f"source '{inc}' (re-derive target from it)")
    # 4. a non-dispatcher source-arch .cpp (real compute, not just <op>_apt.cpp) → complete.
    real_cpp = [p.name for p in kdir.glob("*.cpp") if not p.name.endswith("_apt.cpp")]
    if real_cpp:
        return True, f"non-dispatcher source-arch .cpp present: {real_cpp[0]}"
    return False, (
        "PURE target-arch dispatch shell: op_kernel/ has a target-arch impl dir but no plain "
        "<op>.cpp and no non-target-arch algorithm include — the only implementation is the "
        "target-arch vendor answer (nothing to extract/port). DEBT-165 exemplar: "
        "deformable_offsets (39-line <op>_apt.cpp #include arch35 only)."
    )


def _detect_upstream_arch35(port_source_op: Path) -> Optional[dict]:
    """Look for op_kernel/arch35/ with at least one .h or .cpp file."""
    arch35 = port_source_op / "op_kernel" / "arch35"
    if not arch35.is_dir():
        return None
    files = sorted(
        list(arch35.glob("*.h")) + list(arch35.glob("*.cpp")) + list(arch35.glob("*.hpp"))
    )
    if not files:
        return None
    return {
        "type": "upstream_arch35",
        "path": str(arch35),
        "file_count": len(files),
        "total_lines": _count_lines(files),
        "files": [f.name for f in files],
        "hashes": {f.name: _hash_file(f) for f in files},
        "file_records": _file_records(files),
        "trust": _TRUST_ORDER["upstream_arch35"],
    }


def _detect_upstream_shared_common(port_source_op: Path, op: str) -> Optional[dict]:
    """OL-141 Mode B: arch35 files live in a sibling `<family>_common/op_kernel/arch35/`
    directory rather than in this op's own op_kernel/arch35/. Detection rule:
    walk port_source_op.parent for sibling dirs ending in `_common`, look inside each
    for `op_kernel/arch35/<op>_*.{h,cpp,hpp}` files. First witness 2026-05-14:
    adaptive_avg_pool3d (pooling/adaptive_pool3d_common/op_kernel/arch35/ contains
    adaptive_avg_pool3d_big_kernel.h + adaptive_avg_pool3d_parall_pool.h +
    adaptive_avg_pool3d_simt.h)."""
    parent = port_source_op.parent
    if not parent.is_dir():
        return None
    matched_files: list[Path] = []
    matched_dirs: list[Path] = []
    for sibling in sorted(parent.glob("*_common")):
        if not sibling.is_dir():
            continue
        shared_arch35 = sibling / "op_kernel" / "arch35"
        if not shared_arch35.is_dir():
            continue
        # Look for files whose basename starts with the op name
        op_files = sorted(
            [p for p in shared_arch35.iterdir()
             if p.is_file()
             and p.suffix in (".h", ".cpp", ".hpp")
             and p.name.startswith(f"{op}_")]
        )
        if op_files:
            matched_files.extend(op_files)
            matched_dirs.append(shared_arch35)
    if not matched_files:
        return None
    return {
        "type": "upstream_shared_common",
        "path": str(matched_dirs[0]) if len(matched_dirs) == 1 else [str(d) for d in matched_dirs],
        "shared_common_dirs": [str(d) for d in matched_dirs],
        "file_count": len(matched_files),
        "total_lines": _count_lines(matched_files),
        "files": [f.name for f in matched_files],
        "hashes": {f.name: _hash_file(f) for f in matched_files},
        "file_records": _file_records(matched_files),
        "trust": _TRUST_ORDER["upstream_shared_common"],
    }


def _detect_upstream_v220_entry(port_source_op: Path, op: str) -> Optional[dict]:
    """Bug A (2026-05-23): Look for plain <op>.cpp at op_kernel root — V220-pure entry.

    Contrast with `_detect_upstream_apt`:
      - <op>.cpp     : V220-only entry. #includes top-level headers like
                        op_kernel/<op>_impl.hpp, NOT arch35/<op>_impl.hpp.
                        Safe to prestage verbatim under default-OFF (no arch35 leak).
      - <op>_apt.cpp : cross-arch dispatcher. #includes arch35/<op>_impl.hpp + V220
                        siblings. Under default-OFF, prestaging this triggers
                        ARCH35_WRAP_CHEAT (group_norm_silu-class await_user_decision lockup).

    Surveyed 14 priority ops 2026-05-23 — split observed:
      - plain+apt (Hybrid B applies):  flat_quant, group_norm_silu, swi_glu,
                                        apply_adam_w_v2, FA, GMSQ_v2,
                                        adaptive_avg_pool3d, add_rms_norm_quant
      - plain-only (already V220-pure): LIG, fatrelu_mul, clipped_swiglu,
                                         fused_quant_mat_mul
      - apt-only (Hybrid A required):   erfinv, fast_gelu, gelu, elu
    """
    plain = port_source_op / "op_kernel" / f"{op}.cpp"
    if not plain.is_file():
        return None
    return {
        "type": "upstream_v220_entry",
        "path": str(plain),
        "file_count": 1,
        "total_lines": _count_lines([plain]),
        "files": [plain.name],
        "hashes": {plain.name: _hash_file(plain)},
        "file_records": _file_records([plain]),
        "trust": _TRUST_ORDER["upstream_v220_entry"],
    }


def _detect_upstream_apt(port_source_op: Path, op: str) -> Optional[dict]:
    """Look for <op>_apt.cpp at op_kernel root — the target-aware tiling-key dispatcher.

    Bug A (2026-05-23): suppressed under default-OFF (OPGEN_PRESTAGE_ARCH35 unset/0)
    when an `upstream_v220_entry` (plain <op>.cpp) is also present. See the
    scan() docstring for Hybrid B+A logic; the suppression itself happens at
    scan-orchestration level (not inside this detector — kept pure for tests).
    """
    apt = port_source_op / "op_kernel" / f"{op}_apt.cpp"
    if not apt.is_file():
        return None
    return {
        "type": "upstream_apt",
        "path": str(apt),
        "file_count": 1,
        "total_lines": _count_lines([apt]),
        "files": [apt.name],
        "hashes": {apt.name: _hash_file(apt)},
        "file_records": _file_records([apt]),
        "trust": _TRUST_ORDER["upstream_apt"],
    }


def _detect_workspace_staged(workspace_op: Path) -> Optional[dict]:
    """Look for hand-staged kernel/ directory in the workspace itself."""
    kernel = workspace_op / "kernel"
    if not kernel.is_dir():
        return None
    files = sorted(
        list(kernel.glob("**/*.h"))
        + list(kernel.glob("**/*.cpp"))
        + list(kernel.glob("**/*.hpp"))
    )
    if not files:
        return None
    return {
        "type": "workspace_staged",
        "path": str(kernel),
        "file_count": len(files),
        "total_lines": _count_lines(files),
        "files": [str(f.relative_to(kernel)) for f in files],
        "hashes": {str(f.relative_to(kernel)): _hash_file(f) for f in files},
        "file_records": _file_records(files),
        "trust": _TRUST_ORDER["workspace_staged"],
    }


def _detect_workspace_external(workspace_op: Path) -> Optional[dict]:
    """Look for .prior_art_external/ cherry-pick stash."""
    external = workspace_op / ".prior_art_external"
    if not external.is_dir():
        return None
    files = sorted(list(external.glob("**/*")))
    files = [f for f in files if f.is_file()]
    if not files:
        return None
    return {
        "type": "workspace_external",
        "path": str(external),
        "file_count": len(files),
        "total_lines": _count_lines([f for f in files if f.suffix in (".h", ".cpp", ".hpp")]),
        "files": [str(f.relative_to(external)) for f in files],
        "hashes": {str(f.relative_to(external)): _hash_file(f) for f in files},
        "file_records": _file_records(files),
        "trust": _TRUST_ORDER["workspace_external"],
    }


def _detect_upstream_ascend950_config(port_source_op: Path) -> Optional[dict]:
    """Describe target registration files that participate in the build."""
    config_dir = port_source_op / "op_host" / "config" / "ascend950"
    if not config_dir.is_dir():
        return None
    files = sorted(path for path in config_dir.iterdir() if path.is_file())
    if not files:
        return None
    return {
        "type": "upstream_ascend950_config",
        "path": str(config_dir),
        "file_count": len(files),
        "total_lines": _count_lines(files),
        "files": [path.name for path in files],
        "hashes": {path.name: _hash_file(path) for path in files},
        "file_records": _file_records(files),
        "trust": _TRUST_ORDER["upstream_ascend950_config"],
    }


def _detect_upstream_op_def(port_source_op: Path, op: str) -> Optional[dict]:
    """Describe the target-aware host definition when it is part of the candidate."""
    path = port_source_op / "op_host" / f"{op}_def.cpp"
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if "ascend950" not in content:
        return None
    return {
        "type": "upstream_op_def",
        "path": str(path),
        "file_count": 1,
        "total_lines": _count_lines([path]),
        "files": [path.name],
        "hashes": {path.name: _hash_file(path)},
        "file_records": _file_records([path]),
        "trust": _TRUST_ORDER["upstream_op_def"],
    }


def scan(op: str, port_source: Path, workspace: Path) -> dict:
    """Run all detectors. Returns dict suitable for json.dump.

    P0dd extension (2026-05-23, owner direction "默认不查找目标架构实现"):
    upstream_arch35 + upstream_shared_common detectors are target-source discovery
    paths. By default (env OPGEN_PRESTAGE_ARCH35 unset or 0), they are
    NOT EVEN INVOKED — migration mode generates from the arch22 algorithm source,
    not from an existing arch35 candidate. The worker brief stays unaware of the candidate's
    existence (no `.prior_art_scan.json` entry of those source types).

    When OPGEN_PRESTAGE_ARCH35=1 (explicit opt-in): all 5 detectors run
    as before. Use case: the owner wants to build and measure an arch35 candidate.
    """
    import os as _os
    consult_a5 = _os.environ.get("OPGEN_PRESTAGE_ARCH35", "0") not in ("0", "", "false", "False")
    sources = []
    detectors: list = []
    if consult_a5:
        # Opt-in: include target-source discovery detectors.
        detectors.extend([
            lambda: _detect_upstream_arch35(port_source),
            lambda: _detect_upstream_shared_common(port_source, op),
            lambda: _detect_upstream_ascend950_config(port_source),
            lambda: _detect_upstream_op_def(port_source, op),
        ])
    # Always-on detectors (V220 algorithm source + workspace state, not A5-specific)
    # Bug A (2026-05-23): _detect_upstream_v220_entry checks plain <op>.cpp first.
    # If present + default-OFF, upstream_apt is suppressed below (Hybrid B).
    # If absent + default-OFF + apt present, upstream_apt is suppressed (Hybrid A —
    # worker hand-authors dispatcher from V220 source files in op_kernel/*.h).
    # If opt-in (consult_a5=True): legacy behavior — apt preferred, plain skipped
    # (the cross-arch dispatcher gets the prestage slot for arch35 testing).
    detectors.extend([
        lambda: _detect_upstream_v220_entry(port_source, op),
        lambda: _detect_upstream_apt(port_source, op),
        lambda: _detect_workspace_staged(workspace),
        lambda: _detect_workspace_external(workspace),
    ])
    for detector in detectors:
        result = detector()
        if result is not None:
            sources.append(result)
    if consult_a5:
        target_implementation_present = any(
            source["type"] in {
                "upstream_arch35", "upstream_shared_common", "upstream_apt"
            }
            for source in sources
        )
        if not target_implementation_present:
            sources = [
                source for source in sources
                if source["type"] not in {
                    "upstream_ascend950_config", "upstream_op_def"
                }
            ]
    # Bug A Hybrid B+A — suppress upstream_apt under default-OFF
    if not consult_a5:
        v220_present = any(s["type"] == "upstream_v220_entry" for s in sources)
        apt_present = any(s["type"] == "upstream_apt" for s in sources)
        if apt_present:
            # Default-OFF: never feed cross-arch apt dispatcher to prestage.
            # If plain <op>.cpp present (Hybrid B) → use that; apt suppressed.
            # If apt-only (Hybrid A) → no upstream prestage; worker hand-authors.
            # Either way the apt entry is dropped from sources.
            sources = [s for s in sources if s["type"] != "upstream_apt"]
            if not v220_present:
                # Record the Hybrid A skip for transparency (downstream readers
                # can see "apt was available but we chose to hand-author").
                sources.append({
                    "type": "upstream_apt_suppressed_apt_only_default_off",
                    "reason": (
                        "apt-only op under default-OFF: upstream <op>_apt.cpp "
                        "exists but #includes arch35/* which is forbidden under "
                        "default-OFF. Worker hand-authors dispatcher from V220 "
                        "source headers in op_kernel/*.h top level."
                    ),
                    "trust": "LOW",  # informational record, NOT a prestage source
                    "files": [f"{op}_apt.cpp"],
                })
    # DEBT-165: architecture-based source-completeness — independent of the prestage
    # detectors above (runs regardless of consult_a5). Distinguishes a legit apt-only op
    # (gelu: apt references atvoss/* arch-invariant algorithm) from a pure target-arch
    # dispatch shell (deformable_offsets: apt references only arch35/) so the port-entry
    # gate can reject the latter BEFORE generation.
    src_complete, src_complete_reason = _assess_source_arch_complete(port_source, op)
    return {
        "schema_version": 1,
        "op": op,
        "port_source": str(port_source) if port_source else None,
        "workspace": str(workspace) if workspace else None,
        "has_prior_art": bool(sources),
        "consulted_a5_sources": consult_a5,  # transparency for downstream readers
        "source_arch_complete": src_complete,
        "source_arch_complete_reason": src_complete_reason,
        "sources": sources,
        "highest_trust": (
            max(sources, key=lambda s: ("HIGH", "MEDIUM", "LOW").index(s["trust"]) * -1)["trust"]
            if sources
            else None
        ),
    }


def write_scan_result(workspace: Path, result: dict) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    out = workspace / ".prior_art_scan.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return out


# ---------------------------------------------------------------------------
# Batch mode — quick status sweep across configured arch22→arch35 ops
# ---------------------------------------------------------------------------

# Known op → CANN repo-relative path mapping. Multi-repo because migration ops
# span ops-nn, ops-math, ops-transformer, ops-cv. Paths are relative to
# `<port_source_root>` (default /home/npu_user/workspace/cann/).
#
# Verified 2026-05-13 against local CANN tree.
# Cohort 1: original 10 ops from tasks/a3-toi-a5.xlsx (May 12).
# Cohort 2: 20 ops from tasks/a3-to-a5-task02.png (May 13).
_CANN_OP_PATHS = {
    # ---------- Cohort 1: original 10 (tasks/a3-toi-a5.xlsx, 2026-05-12) ----------
    "ada_layer_norm":          "ops-nn/norm/ada_layer_norm",
    "apply_adam_w_quant":      "ops-nn/optim/apply_adam_w_quant",
    "ctc_loss_v3":             "ops-nn/loss/ctc_loss_v3",
    "dynamic_rnnv2":           "ops-nn/rnn/dynamic_rnnv2",
    "fused_quant_mat_mul":     "ops-nn/matmul/fused_quant_mat_mul",
    "gather_elements_v2":      "ops-nn/index/gather_elements_v2",
    "group_norm_silu_quant":   "ops-nn/norm/group_norm_silu_quant",
    "index_put_with_sort":     "ops-nn/index/index_put_with_sort",
    "rms_norm_quant":          "ops-nn/norm/rms_norm_quant",
    "top_k_top_p_sample_v2":   "ops-nn/index/top_k_top_p_sample_v2",
    # ---------- Cohort 2: 20 ops (tasks/a3-to-a5-task02.png, 2026-05-13) ----------
    # Two entries overlap cohort 1 (CTCLossV3, FusedQuantMatmul) and are NOT
    # re-listed — the mapping is keyed by canonical snake-case op name so
    # duplicates are de-duped naturally.
    "adaptive_avg_pool3d":         "ops-nn/pooling/adaptive_avg_pool3d",
    "apply_adam_w_v2":             "ops-nn/optim/apply_adam_w_v2",
    "foreach_erf":                 "ops-nn/foreach/foreach_erf",
    "grid_sample":                 "ops-cv/image/grid_sample",
    "group_norm_silu":             "ops-nn/norm/group_norm_silu",
    "histogram_v2":                "ops-math/math/histogram_v2",
    "masked_select_v3":            "ops-math/conversion/masked_select_v3",
    "slice_v2":                    "ops-math/conversion/strided_slice_v2",
    "repeat_interleave_v2":        "ops-nn/index/repeat_interleave",
    "swi_glu":                     "ops-nn/activation/swi_glu",
    "add_rms_norm_quant":          "ops-nn/norm/add_rms_norm_quant",
    "rope_with_sin_cos_cache":     "ops-transformer/posembedding/rope_with_sin_cos_cache",
    "moe_init_routing_v3":         "ops-transformer/moe/moe_init_routing_v3",
    "top_p_sample":                "ops-nn/index/top_k_top_p_sample",
    "flat_quant":                  "ops-nn/quant/flat_quant",
    "grouped_matmul_swiglu_quant": "ops-transformer/gmm/grouped_matmul_swiglu_quant",
    "flash_attention_score":       "ops-transformer/attention/flash_attention_score",
    "lightning_indexer_grad":      "ops-transformer/attention/lightning_indexer_grad",
}

# Tag each op with its cohort + display name for the report
_COHORT_META = {
    # cohort 1 — original 10
    "ada_layer_norm":          ("c1", "AdaLayerNorm"),
    "apply_adam_w_quant":      ("c1", "ApplyAdamWQuant"),
    "ctc_loss_v3":             ("c1", "CTCLossV3"),
    "dynamic_rnnv2":           ("c1", "DynamicRNNv2"),
    "fused_quant_mat_mul":     ("c1", "FusedQuantMatMul"),
    "gather_elements_v2":      ("c1", "GatherElementsV2"),
    "group_norm_silu_quant":   ("c1", "GroupNormSiluQuant"),
    "index_put_with_sort":     ("c1", "IndexPutWithSort"),
    "rms_norm_quant":          ("c1", "RmsNormQuant"),
    "top_k_top_p_sample_v2":   ("c1", "TopKTopPSampleV2"),
    # cohort 2 — 20 ops from task02.png
    "adaptive_avg_pool3d":         ("c2", "AdaptiveAvgPool3d"),
    "apply_adam_w_v2":             ("c2", "ApplyAdamWV2"),
    "foreach_erf":                 ("c2", "ForeachErf"),
    "grid_sample":                 ("c2", "GridSample"),
    "group_norm_silu":             ("c2", "GroupNormSilu"),
    "histogram_v2":                ("c2", "HistogramV2"),
    "masked_select_v3":            ("c2", "MaskedSelectV3"),
    "slice_v2":                    ("c2", "SliceV2"),
    "repeat_interleave_v2":        ("c2", "RepeatInterleaveV2"),
    "swi_glu":                     ("c2", "SwiGlu"),
    "add_rms_norm_quant":          ("c2", "AddRmsNormQuant"),
    "rope_with_sin_cos_cache":     ("c2", "RopeWithSinCosCache"),
    "moe_init_routing_v3":         ("c2", "MoeInitRoutingV3"),
    "top_p_sample":                ("c2", "TopPSample"),
    "flat_quant":                  ("c2", "FlatQuant"),
    "grouped_matmul_swiglu_quant": ("c2", "GroupedMatmulSwigluQuant"),
    "flash_attention_score":       ("c2", "FlashAttentionScore"),
    "lightning_indexer_grad":      ("c2", "LightningIndexerGrad"),
}

# Backwards-compat alias (older callers may import _OPS_NN_PATHS).
_OPS_NN_PATHS = _CANN_OP_PATHS


def batch_scan(port_source_root: Path, workspace_root: Path) -> dict:
    """Sweep all configured arch22→arch35 ops; return summary dict.

    `port_source_root` should now point to the parent CANN dir
    (`/home/npu_user/workspace/cann/`), not a specific repo subdir — the
    per-op path string is `<repo>/<subdir>` to support multi-repo.
    """
    rows = []
    for op, repo_subdir in _CANN_OP_PATHS.items():
        port_source = port_source_root / repo_subdir
        workspace = workspace_root / op
        cohort, display = _COHORT_META.get(op, ("?", op))
        base = {"op": op, "cohort": cohort, "display": display}
        if not port_source.is_dir():
            rows.append({**base, "verdict": "MISSING_UPSTREAM", "path": str(port_source)})
            continue
        result = scan(op, port_source, workspace)
        rows.append(
            {
                **base,
                "has_prior_art": result["has_prior_art"],
                "highest_trust": result["highest_trust"],
                "sources": [
                    {"type": s["type"], "files": s["file_count"], "lines": s["total_lines"]}
                    for s in result["sources"]
                ],
            }
        )
    return {
        "schema_version": 2,
        "port_source_root": str(port_source_root),
        "workspace_root": str(workspace_root),
        "n_ops": len(rows),
        "n_with_prior_art": sum(1 for r in rows if r.get("has_prior_art")),
        "rows": rows,
    }


def _has_upstream_arch35(row: dict) -> bool:
    return any(s.get("type") == "upstream_arch35" for s in row.get("sources", []))


def format_batch_table(summary: dict) -> str:
    """Markdown table for human / Discord consumption.

    Sections (by cohort × verdict):
      - Available (HIGH-trust upstream arch35) — advisory candidate may be staged
      - No upstream arch35 — standard generation continues without that candidate
      - Missing upstream tree — op not present in CANN checkout
    Each section is split by cohort (c1 = original 10 ops, c2 = 20 ops from task02.png).
    """
    actionable = [
        r for r in summary["rows"]
        if r.get("verdict") != "MISSING_UPSTREAM" and _has_upstream_arch35(r)
    ]
    no_upstream = [
        r for r in summary["rows"]
        if r.get("verdict") != "MISSING_UPSTREAM" and not _has_upstream_arch35(r)
    ]
    missing = [r for r in summary["rows"] if r.get("verdict") == "MISSING_UPSTREAM"]

    def _cohort_label(c: str) -> str:
        return {"c1": "Cohort 1 (10 ops)",
                "c2": "Cohort 2 (20 ops)"}.get(c, c)

    lines = [
        "# Prior-art scan summary",
        "",
        f"- port_source_root: `{summary['port_source_root']}`",
        f"- ops scanned: {summary['n_ops']}",
        f"- advisory arch35 candidates present: {len(actionable)}",
        f"- no upstream arch35 candidate: {len(no_upstream)}",
        f"- MISSING_UPSTREAM (not in CANN checkout): {len(missing)}",
        "",
        "## Advisory arch35 candidate detected",
        "",
        "| cohort | op | display | upstream files | lines | also workspace-staged? | verify status |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for r in sorted(actionable, key=lambda x: (x.get("cohort", "?"), x["op"])):
        up = next(s for s in r["sources"] if s["type"] == "upstream_arch35")
        ws = any(s["type"] == "workspace_staged" for s in r["sources"])
        lines.append(
            f"| {r.get('cohort','?')} | `{r['op']}` | {r.get('display','—')} | "
            f"{up['files']} | {up['lines']} | "
            f"{'yes' if ws else 'no'} | unverified |"
        )

    lines += [
        "",
        "## No upstream arch35 candidate — continue standard generation",
        "",
        "| cohort | op | display | workspace sources |",
        "|---|---|---|---|",
    ]
    for r in sorted(no_upstream, key=lambda x: (x.get("cohort", "?"), x["op"])):
        sources = (
            "; ".join(f"{s['type']} / {s['files']}f / {s['lines']}L" for s in r["sources"])
            if r["sources"]
            else "(none)"
        )
        lines.append(
            f"| {r.get('cohort','?')} | `{r['op']}` | {r.get('display','—')} | {sources} |"
        )

    if missing:
        lines += [
            "",
            "## MISSING_UPSTREAM — op directory not found in local CANN checkout",
            "",
            "| cohort | op | display | tried path |",
            "|---|---|---|---|",
        ]
        for r in sorted(missing, key=lambda x: (x.get("cohort", "?"), x["op"])):
            lines.append(
                f"| {r.get('cohort','?')} | `{r['op']}` | {r.get('display','—')} | `{r['path']}` |"
            )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    single = sub.add_parser("scan", help="single-op prior-art scan")
    single.add_argument("--op", required=True)
    single.add_argument("--port-source", required=True, type=Path)
    single.add_argument("--workspace", required=True, type=Path)
    single.add_argument("--json", action="store_true", help="print JSON to stdout")

    batch = sub.add_parser("batch", help="quick sweep across configured arch22→arch35 ops")
    batch.add_argument(
        "--port-source-root",
        type=Path,
        default=Path(os.path.expanduser("~/workspace/cann")),
        help=(
            "Parent dir containing the CANN repos (ops-nn, ops-math, "
            "ops-transformer, ops-cv). Per-op paths are `<repo>/<subdir>`."
        ),
    )
    batch.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(os.path.expanduser("~/workspace/a5_ops/workspace")),
    )
    batch.add_argument("--json", action="store_true", help="emit JSON not markdown")
    batch.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional output path for the markdown table",
    )

    args = p.parse_args(argv)

    if args.cmd == "scan":
        result = scan(args.op, args.port_source, args.workspace)
        out_path = write_scan_result(args.workspace, result)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"wrote {out_path}")
            print(f"has_prior_art={result['has_prior_art']} "
                  f"highest_trust={result['highest_trust']} "
                  f"n_sources={len(result['sources'])}")
        return 0

    if args.cmd == "batch":
        summary = batch_scan(args.port_source_root, args.workspace_root)
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            md = format_batch_table(summary)
            print(md)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(md)
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
