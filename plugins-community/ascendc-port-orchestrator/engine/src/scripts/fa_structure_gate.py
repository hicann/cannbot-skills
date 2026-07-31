# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""FA white-box structural verify-gate.

Catches the black-box divergence the owner flagged: a generated FA kernel that
drops to vec-only / single-matmul / full-S2-one-shot instead of the source-derived
performant structure (cube QK^T + cube P@V + online-softmax S2 row-tiling).
See docs/design/FA_CLASS_DESIGN_NOTES.md#fa-a3-whitebox-generation-directive.

Heuristic for AscendC matmul::Matmul/Mmad structure. NOT a
substitute for reading the kernel — a fast deterministic divergence flag.

Usage: python3 src/scripts/fa_structure_gate.py <kernel_dir_or_file> [...]
Exit 0 = aligned (PASS), 1 = divergent (FAIL), 2 = no FA kernel found.

HEURISTIC LIMITATIONS (this is a CLI DIAGNOSTIC, NOT an auto-reject finalize-gate).
It runs textual regex over concatenated source, so it can mis-judge:
- FALSE-POSITIVE (over-count): `cube_total` counts pattern HITS, not real matmul
  instantiations — a single matmul appearing in both a declaration AND a comment, or
  a `MatmulImpl<...>` mentioned inside a comment/dead-code block, inflates the count.
  (The `<`-suffix requirement filters bare `MatmulImpl,` mentions, but not `MatmulImpl<`
  in a comment.)
- FALSE-POSITIVE (tiling): `has_s2_tiling` passes on a SINGLE token match (e.g. a
  `sInnerSize` in a comment or unreachable branch).
- No comment / dead-code scoping (no AST parse).
Therefore: use the output as a human-read divergence HINT (run it during verify, read
the cube/tiling breakdown), NOT as an automatic kernel-reject. Hardening to an auto-gate
(count real matmul-instantiations via AST, require BOTH tiling signals, scope out
comments) is tracked separately; #263's tile-size-consistency gate is the deterministic
auto-gate companion on a different axis.
"""
import logging
import re
import sys
from pathlib import Path

_SRC_EXT = {".h", ".hpp", ".cpp", ".cc", ".cce", ".py"}
# Cube-matmul construction signals (each ~ one cube matmul operand pipeline).
_CUBE_PATS = [
    (re.compile(r"matmul::Matmul\s*<"), "matmul::Matmul<>"),
    (re.compile(r"\bMatmulImpl\s*<"), "MatmulImpl<>"),          # AscendC cube abstraction (kw path)
    (re.compile(r"REGIST_MATMUL_OBJ|RegistMatmul"), "REGIST_MATMUL"),
    (re.compile(r"\bMmad\s*\("), "Mmad()"),                     # raw cube intrinsic
]
# Online-softmax S2 row-tiling signals (FlashAttention inner kv-loop + running stat).
_TILING_PATS = [
    re.compile(r"sInnerLoop|singleProcessSInner|s2(Base|Inner)|kvInner|n_kv_tile|sInnerSize"),
    re.compile(r"running[_A-Za-z]*max|rowmax|gMax|lMax|onlineSoftmax|update.*max.*sum", re.I),
]


def _gather_text(target: Path) -> str:
    if target.is_file():
        files = [target]
    else:
        files = []
        for candidate in target.rglob("*"):
            if (
                candidate.is_file()
                and candidate.suffix in _SRC_EXT
                and "pybind11" not in candidate.name
                and "test" not in candidate.name.lower()
            ):
                files.append(candidate)
    out = []
    for f in files:
        try:
            out.append(f.read_text(errors="ignore"))
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
    return "\n".join(out)


def gate(target: Path) -> dict:
    text = _gather_text(target)
    if not text.strip():
        return {"target": str(target), "verdict": "NO_KERNEL", "reason": "no source files"}
    cube_hits = {}
    for pat, name in _CUBE_PATS:
        n = len(pat.findall(text))
        if n:
            cube_hits[name] = n
    cube_total = sum(cube_hits.values())
    has_tiling = all(p.search(text) for p in _TILING_PATS) or bool(_TILING_PATS[0].search(text))
    # Performant FA needs cube for BOTH matmuls (QK^T + P@V) → ≥2 cube constructs.
    cube_ok = cube_total >= 2
    verdict = "ALIGNED" if (cube_ok and has_tiling) else "DIVERGENT"
    reasons = []
    if not cube_ok:
        reasons.append(f"cube-matmul constructs={cube_total} (<2 → vec-only/single-matmul divergence; "
                       f"performant FA needs cube QK^T + cube P@V — OL-186)")
    if not has_tiling:
        reasons.append("no online-softmax S2 row-tiling signal (full-S2-one-shot → UB-overflow class)")
    return {"target": str(target), "verdict": verdict, "cube_hits": cube_hits,
            "cube_total": cube_total, "has_s2_tiling": has_tiling,
            "reason": "; ".join(reasons) or "cube-double-matmul + S2-tiling present"}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    worst = 0
    for arg in sys.argv[1:]:
        r = gate(Path(arg))
        print(f"[{r['verdict']}] {r['target']}")
        print(f"    cube={r.get('cube_hits', {})} total={r.get('cube_total','?')} "
              f"s2_tiling={r.get('has_s2_tiling','?')}")
        print(f"    {r['reason']}")
        if r["verdict"] == "DIVERGENT":
            worst = max(worst, 1)
        elif r["verdict"] == "NO_KERNEL":
            worst = max(worst, 2)
    return worst


if __name__ == "__main__":
    sys.exit(main())
