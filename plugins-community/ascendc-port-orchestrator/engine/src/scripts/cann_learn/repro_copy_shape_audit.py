# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""repro_copy_shape_audit.py — layer-(b) backstop for the hermetic reproduction probe.

Design: docs/design/HERMETIC_REPRODUCTION_PROBE_DESIGN.md §(b) (main, 2026-05-30).
Owner concern 2026-05-30 03:26Z: "白盒的成果会被黑盒拿过去直接用" — a reproduction
probe is only admissible if the orch-generated kernel was NOT copied from our own
finished white-box archive (via working-tree, git history, or a memorized handover).

This is the *load-bearing* control: hermetic worktree (a) just makes copying harder;
this audit catches a copy via ANY channel. Reuses C34c `copy_shape.check()`
(token n-gram contiguous overlap, identifier-normalized → catches renamed-identifier
copy), comparing the probe-generated `op_kernel/*` against the archived white-box
`op_kernel/*`.

Verdict: per-file max overlap >= threshold ⇒ COPY_DETECTED ⇒ probe verdict VOID
(hard-gate). Below ⇒ INDEPENDENT (residual overlap is generic shared-API noise).

CALIBRATION (measured on an FA arch35 op_kernel) — why the defaults
are n=20 / threshold=0.40 / shared-infra headers excluded, NOT the C34c default
n=5 / 5% (which false-positives at 64-72% on independent kernels):

  op-specific files only (exclude kernel_common/matmul_tile/workspace_queue/pybind11):
    n    self(exact copy)   FA-vs-Attn(same-class indep)   FA-vs-elu(diff-class)
    5        1.00                  0.69                          0.64      ← all VOID (useless)
    16       1.00                  0.31                          0.09
    20       1.00                  0.27                          0.03      ← chosen
    25       1.00                  0.17                          0.002
    40       1.00                  0.059                         0.0

  Same-class-independent (FA vs another attention kernel) is the hardest false-positive.
  n=20 + threshold 0.40 separates exact/renamed copy (→1.0 / ~0.9, identifier-normalized)
  from same-class-independent (0.27). Identifier-normalization (C34c) makes generic
  AscendC boilerplate look identical, so whole-file n=5 is dominated by shared structure;
  longer n-grams + infra-exclusion isolate op-specific logic.

  Re-calibrate per kernel CLASS before trusting as a hard gate (run --calibrate).

  ⚠️ ADVISORY ONLY for cv_agent-lineage ops (FA / port_fa_cv_agent): a LEGIT port of
  cv_agent's sanctioned design shares ~0.60 lineage overlap with our own archive
  (which is itself a cv_agent port), so a flat probe-vs-our-archive threshold would
  FALSE-VOID a legitimate port. Do NOT use as a hard gate for such ops. The
  principled discriminator (a5-specific delta = our-archive n-grams NOT in cv_agent)
  is DEFERRED per owner's over-design caution 2026-05-30 — not built. Rely on
  hermetic blinding (0 reads of our archive) as the primary non-copy evidence.

This mirrors the params baked into orchestrator/hermetic_probe.py (main, 4beed656).

Usage:
  python3 -m cann_learn.repro_copy_shape_audit \\
      --generated workspace/<op>/op_kernel \\
      --reference output/a3_to_a5_port/src/kernels/<op>/op_kernel \\
      [--threshold 0.40] [--n 20] [--glob '*.h,*.cpp']
  # calibration sweep (reproduce the table above for a new op + independent controls):
  python3 -m cann_learn.repro_copy_shape_audit --calibrate \\
      --reference <op>/op_kernel --controls <indep1>/op_kernel,<indep2>/op_kernel
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from cann_learn.copy_shape import check as _c34c_check
except Exception:  # allow running as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cann_learn.copy_shape import check as _c34c_check


# Shared-infra headers: generic utilities any kernel of the class reuses verbatim.
# Excluded from copy-shape comparison so the score reflects op-specific logic, not
# boilerplate (see CALIBRATION in module docstring). Mirrors hermetic_probe.py.
_INFRA_EXCLUDE = {"kernel_common.h", "matmul_tile.h", "workspace_queue.h", "pybind11.cpp"}
_AUDIT_N = 20
_AUDIT_THRESHOLD = 0.40


def _collect(d: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out.extend(sorted(d.glob(pat)))
    # dedup preserving order; drop shared-infra boilerplate headers
    seen = set()
    uniq = []
    for p in out:
        if p.is_file() and p not in seen and p.name not in _INFRA_EXCLUDE:
            seen.add(p)
            uniq.append(p)
    return uniq


def audit(generated_dir: Path, reference_dir: Path, *, threshold: float = _AUDIT_THRESHOLD,
          n: int = _AUDIT_N, patterns: list[str] | None = None) -> dict:
    patterns = patterns or ["*.h", "*.cpp", "*.cc"]
    gen_files = _collect(generated_dir, patterns)
    ref_files = _collect(reference_dir, patterns)

    per_file = []
    max_score = 0.0
    for gf in gen_files:
        text = gf.read_text(errors="replace")
        res = _c34c_check(gf.name, text, ref_files, n=n, threshold=threshold)
        max_score = max(max_score, res.score)
        per_file.append({
            "generated_file": gf.name,
            "overlap_score": round(res.score, 4),
            "match_count": res.match_count,
            "total_windows": res.total_windows,
            "copied": res.score >= threshold,
            "sample_matches": res.sample_matches[:3],
        })

    verdict = "COPY_DETECTED" if max_score >= threshold else "INDEPENDENT"
    return {
        "verdict": verdict,
        "probe_admissible": verdict == "INDEPENDENT",
        "max_overlap_score": round(max_score, 4),
        "threshold": threshold,
        "n_gram": n,
        "generated_dir": str(generated_dir),
        "reference_dir": str(reference_dir),
        "n_generated_files": len(gen_files),
        "n_reference_files": len(ref_files),
        "per_file": per_file,
        "note": ("COPY_DETECTED ⇒ probe verdict VOID (not autonomous generation). "
                 "INDEPENDENT ⇒ residual overlap is generic shared-API noise; "
                 "calibrate threshold on a known-independent kernel pair before trusting."),
    }


def calibrate(reference_dir: Path, controls: list[Path], *,
              ns=(5, 12, 16, 20, 25, 30, 40), patterns: list[str] | None = None) -> dict:
    """Sweep n over (reference-vs-itself, reference-vs-each-independent-control) to
    re-derive the separation table for a new op class. self≈1.0 (exact-copy upper
    bound); a usable (n, threshold) is one where independent controls sit well below
    threshold while self/renamed-copy stay high. Pick the smallest n whose worst
    control < ~0.5*threshold.
    """
    patterns = patterns or ["*.h", "*.cpp", "*.cc"]
    ref_files = _collect(reference_dir, patterns)
    rows = []
    for n in ns:
        self_s = max((_c34c_check(f.name, f.read_text(errors="replace"), ref_files, n=n).score
                      for f in ref_files), default=0.0)
        ctrl_scores = {}
        for ctrl in controls:
            cf = _collect(ctrl, patterns)
            ctrl_scores[ctrl.parent.name or ctrl.name] = round(max(
                (_c34c_check(f.name, f.read_text(errors="replace"), cf, n=n).score
                 for f in ref_files), default=0.0), 4)
        rows.append({"n": n, "self": round(self_s, 4), "controls": ctrl_scores})
    return {"reference": str(reference_dir), "ns": list(ns), "table": rows,
            "note": "pick smallest n where worst control << threshold; self stays ~1.0"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generated", type=Path, help="probe-generated kernel dir (op_kernel/)")
    ap.add_argument("--reference", required=True, type=Path,
                    help="archived white-box kernel dir (the answer)")
    ap.add_argument("--threshold", type=float, default=_AUDIT_THRESHOLD)
    ap.add_argument("--n", type=int, default=_AUDIT_N)
    ap.add_argument("--glob", default="*.h,*.cpp,*.cc",
                    help="comma-separated file globs")
    ap.add_argument("--calibrate", action="store_true",
                    help="run n-sweep separation table instead of auditing")
    ap.add_argument("--controls", default="",
                    help="comma-separated independent kernel dirs (for --calibrate)")
    args = ap.parse_args(argv)
    pats = [p.strip() for p in args.glob.split(",") if p.strip()]

    if not args.reference.is_dir():
        print(f"ERROR: --reference not a dir: {args.reference}", file=sys.stderr)
        return 2

    if args.calibrate:
        controls = [Path(c.strip()) for c in args.controls.split(",") if c.strip()]
        print(json.dumps(calibrate(args.reference, controls, patterns=pats), indent=2))
        return 0

    if not args.generated or not args.generated.is_dir():
        print(f"ERROR: --generated not a dir: {args.generated}", file=sys.stderr)
        return 2
    out = audit(args.generated, args.reference,
                threshold=args.threshold, n=args.n, patterns=pats)
    print(json.dumps(out, indent=2))
    # exit 3 on COPY_DETECTED so a CI/gate can branch on it (0=admissible)
    return 3 if out["verdict"] == "COPY_DETECTED" else 0


if __name__ == "__main__":
    sys.exit(main())
