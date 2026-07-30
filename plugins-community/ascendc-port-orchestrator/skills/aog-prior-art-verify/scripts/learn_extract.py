# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Phase 6 of /aog-prior-art-verify — extract KB-promotable patterns.

Phase 6 ALWAYS runs (per user directive 2026-05-13). The Phase 5 verdict
just decides WHICH KB section the lessons land in:

  VERDICT=CANDIDATE_PASS          → positive candidate patterns
  VERDICT=CANDIDATE_PRECISION_GAP → precision counterexamples
  VERDICT=CANDIDATE_PERF_GAP      → performance counterexamples
  VERDICT=CANDIDATE_DET_GAP       → determinism counterexamples
  VERDICT=CANDIDATE_BUILD_GAP     → build-failure counterexamples

Mechanical extraction (no LLM): diff signal counts only (lines added/removed,
key API surface mentions, structural-feature flags). Promotion to canonical
PATTERN_INDEX.md / OL goes through `aog-knowledge-maintain` Mode 1 — this
module only WRITES candidates to `patterns/unverified/candidates.md`.

Outputs:
- `workspace/<op>/prior_art_learn.md` — human-readable extracted lessons
- Append to `patterns/unverified/candidates.md` — auto-tagged candidates
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_PLUGIN_ROOT = Path(__file__).resolve().parents[3]
_CANDIDATES_PATH = (_PLUGIN_ROOT / "kb" / "target" / "ascendc"
                    / "patterns" / "unverified" / "candidates.md")


@dataclass
class LearnReport:
    op: str
    verdict: str
    candidates_added: int = 0
    deltas_extracted: list[dict] = field(default_factory=list)
    learn_md_path: Optional[Path] = None
    candidates_appended: bool = False
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mechanical delta detection (no LLM)
# ---------------------------------------------------------------------------
_API_SURFACE_PATTERNS = {
    "MicroAPI_RegTensor":       r"\bRegTensor\b",
    "MicroAPI_VEC_SCOPE":       r"__VEC_SCOPE__",
    "MicroAPI_MaskReg":         r"\bMaskReg\b",
    "MicroAPI_CastTrait":       r"\bCastTrait\b",
    "SIMT_simt_vf":             r"__simt_vf__",
    "SIMT_LAUNCH_BOUND":        r"\bLAUNCH_BOUND\b",
    "SIMT_local_mem":           r"__local_mem__",
    "SIMT_AtomicAdd":           r"Simt::AtomicAdd",
    "MicroScaling_fp8_e8m0":    r"\bfp8_e8m0_t\b",
    "MicroScaling_l12l0a_mx":   r"asc_copy_l12l0[ab]_mx",
    "Cube_MatmulImpl":          r"\bMatmulImpl\b",
    "Cube_typed_config":        r"\bMatmulApiStaticTiling\b",
    "Norm_subnormal_LnConfig":  r"LnConfig.*subnormal",
    "SPR_overflow":             r"\bspr_overflow\b",
    "Multi_variant_split":      r"#include\s+[\"'].*_(big_kernel|parall|simt|welford|full_load)\.h",
    "Single_to_multi_algo":     r"TEMPLATE_MODE|DYTPE_MODE|tilingkey.*dispatch",
}

_GUARD_REMOVAL_PATTERNS = [
    r"__CCE_AICORE__\s*==\s*220",
    r"__NPU_ARCH__\s*==\s*(3003|3113)",
]


def _scan_file_for_signals(path: Path) -> dict:
    """Return dict of signal_name → count for one file."""
    try:
        text = path.read_text()
    except OSError:
        return {}
    signals = {}
    for name, pat in _API_SURFACE_PATTERNS.items():
        matches = re.findall(pat, text)
        if matches:
            signals[name] = len(matches)
    # Anti-pattern: V220-specific guards (we DON'T want these in arch35/)
    guard_hits = 0
    for pat in _GUARD_REMOVAL_PATTERNS:
        guard_hits += len(re.findall(pat, text))
    if guard_hits:
        signals["V220_guards_present"] = guard_hits
    return signals


def _scan_candidate_dir(candidate_dir: Path) -> dict:
    """Aggregate signal counts across all .h/.cpp files in candidate."""
    agg = {}
    for f in candidate_dir.rglob("*"):
        if not f.is_file() or f.suffix not in (".h", ".cpp", ".hpp"):
            continue
        sigs = _scan_file_for_signals(f)
        for name, count in sigs.items():
            agg[name] = agg.get(name, 0) + count
    return agg


def _line_counts(candidate_dir: Path) -> dict:
    """Total lines in arch35 / shared-common to give a port-size proxy."""
    totals = {"arch35_h": 0, "arch35_cpp": 0, "apt_cpp": 0, "shared_common": 0}
    for f in candidate_dir.rglob("*"):
        if not f.is_file():
            continue
        try:
            n = sum(1 for _ in f.open("rb"))
        except OSError:
            continue
        parts = f.parts
        if "arch35" in parts and f.suffix == ".h":
            totals["arch35_h"] += n
        elif "arch35" in parts and f.suffix == ".cpp":
            totals["arch35_cpp"] += n
        elif f.name.endswith("_apt.cpp"):
            totals["apt_cpp"] += n
        elif any(p.endswith("_common") for p in parts):
            totals["shared_common"] += n
    return totals


def extract(op: str, workspace: Path) -> LearnReport:
    """Phase 6 entry. Always runs; verdict from prior_art_verdict.json
    determines the tag/section in the output."""
    verdict_path = workspace / "prior_art_verdict.json"
    candidate_dir = workspace / ".prior_art_candidate"
    rep = LearnReport(op=op, verdict="UNKNOWN")

    if verdict_path.is_file():
        try:
            data = json.loads(verdict_path.read_text())
            rep.verdict = str(data.get("verdict", "UNKNOWN"))
        except Exception as e:
            rep.errors.append(f"verdict parse error: {e}")

    if not candidate_dir.is_dir():
        rep.errors.append("candidate dir missing — nothing to extract")
        return rep

    signals = _scan_candidate_dir(candidate_dir)
    sizes = _line_counts(candidate_dir)

    # Build a delta entry per non-zero signal
    is_pass = rep.verdict == "CANDIDATE_PASS"
    source_tag = "upstream_pass" if is_pass else "upstream_fail"
    counter_example = not is_pass

    for sig_name, count in sorted(signals.items()):
        # V220 guard hits become anti-pattern under any verdict
        is_anti = sig_name == "V220_guards_present"
        rep.deltas_extracted.append({
            "signal": sig_name,
            "count": count,
            "source": "upstream_fail" if is_anti else source_tag,
            "counter_example": counter_example or is_anti,
        })

    rep.candidates_added = len(rep.deltas_extracted)

    # Write prior_art_learn.md
    md_path = workspace / "prior_art_learn.md"
    lines = [
        f"# Prior-art extraction — `{op}`",
        "",
        f"**Verdict**: `{rep.verdict}`  | **Source tag**: `{source_tag}`",
        "",
        "## Port size (line counts)",
        "",
        f"- arch35 headers: {sizes['arch35_h']} lines",
        f"- arch35 cpp:     {sizes['arch35_cpp']} lines",
        f"- _apt.cpp:       {sizes['apt_cpp']} lines",
        f"- shared_common:  {sizes['shared_common']} lines",
        "",
        "## API-surface + structural signals",
        "",
    ]
    if rep.deltas_extracted:
        lines.append("| Signal | Count | Tag | Counter-example |")
        lines.append("|---|---|---|---|")
        for d in rep.deltas_extracted:
            lines.append(
                f"| `{d['signal']}` | {d['count']} | `{d['source']}` | "
                f"{'yes' if d['counter_example'] else 'no'} |"
            )
    else:
        lines.append("_no notable signals detected_")
    lines.append("")
    lines.append("## Action")
    lines.append("")
    if is_pass:
        lines.append(
            "This candidate is eligible as a provenance-bound implementation seed. "
            "It does not replace fresh arch22 truth capture or the normal O4/O5 "
            "customer verification. Promote reusable principles via "
            "`aog-knowledge-maintain` Mode 1."
        )
    else:
        lines.append(
            f"This prior-art candidate has an advisory gap ({rep.verdict}). "
            "Signals above flag what it tried; the worker brief should avoid any "
            "`counter_example=yes` row while continuing normal generation."
        )
    md_path.write_text("\n".join(lines) + "\n")
    rep.learn_md_path = md_path

    return rep


def append_candidates(rep: LearnReport,
                      candidates_path: Optional[Path] = None) -> bool:
    """Append extracted deltas to patterns/unverified/candidates.md as a
    single CAND-PRIOR-ART block (one block per op, listing all signals)."""
    if not rep.deltas_extracted:
        return False
    target = candidates_path or _CANDIDATES_PATH
    if not target.parent.is_dir():
        # Test envs may not have the canonical path; bail silently
        return False

    block = [
        "",
        f"### CAND-PRIOR-ART-{rep.op}",
        "",
        f"`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class={rep.op}`",
        f"`verified_on: workspace prior_art_verify run; verdict={rep.verdict}`",
        "",
        f"**Source**: aog-prior-art-verify Phase 6 extraction "
        f"(verdict={rep.verdict}).",
        "",
        "**Mechanical signals from upstream `.prior_art_candidate/`**:",
        "",
        "| Signal | Count | Source tag | Counter-example |",
        "|---|---|---|---|",
    ]
    for d in rep.deltas_extracted:
        block.append(
            f"| `{d['signal']}` | {d['count']} | `{d['source']}` | "
            f"{'yes' if d['counter_example'] else 'no'} |"
        )
    block.append("")
    block.append(
        "**Status**: 1-op evidence. Promote via `aog-knowledge-maintain` "
        "Mode 1 cross-op validation."
    )
    block.append("")

    with target.open("a") as f:
        f.write("\n".join(block) + "\n")
    rep.candidates_appended = True
    return True


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--op", required=True)
    p.add_argument("--workspace", required=True, type=Path)
    p.add_argument("--no-append", action="store_true",
                   help="don't append to patterns/unverified/candidates.md")
    args = p.parse_args(argv)
    rep = extract(args.op, args.workspace)
    if not args.no_append:
        append_candidates(rep)
    print(f"verdict={rep.verdict} signals={rep.candidates_added} "
          f"appended={rep.candidates_appended} → {rep.learn_md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
