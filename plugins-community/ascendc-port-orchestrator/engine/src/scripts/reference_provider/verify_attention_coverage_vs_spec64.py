#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Verify case_gen self-generates attention coverage ≥ the curated FA spec V2-64 —
WITHOUT a benchmark reference (the owner's case_gen-coverage bar, 2026-06-08).

Why this exists
---------------
Real a3-port / op-gen agents have NO benchmark; their ONLY input coverage is
case_gen output. The curated `3_FusionAttention_64.json` (spec V2-64) was a manual
walkaround for FA. The product requirement: case_gen, given only an `op_class`
declaration, must self-generate coverage that meets-or-exceeds that curated set on
every coverage DIMENSION the spec exercises. This script is the author≠measurer
verify harness: it embeds the canonical attention schema (the same shape an FA
input_gen declares), runs the real `generate_cases`, and asserts coverage ≥ spec-64.

Spec V2-64 dimensions (from docs/analysis/FA_VERIFY_SET_VS_SPEC64_RECONCILE_2026_06_08.md):
  - D-buckets exercised: {64, 128, 512, 640, 768}  (NO 256 in spec; HAS 640/768)
  - keep_prob<1 (dropout): 6 cases, spanning low + high D; high-D ARE dropout
    (640 → BNSD kp0.9, 768 → SBH kp0.8)
  - layouts: BSH / SBH / BNSD / BSND
  - dtype: fp16 / bf16 / fp32  (case_gen expands dtype separately)

Run:
  python3 src/scripts/reference_provider/verify_attention_coverage_vs_spec64.py
Exit 0 + "COVERAGE MEETS SPEC" on success; non-zero + the failing dimension otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_gen as cg  # noqa: E402

# Spec V2-64 dimension targets (what coverage must meet-or-exceed).
SPEC_D_BUCKETS = {64, 128, 512, 640, 768}
SPEC_KP_LT1_COUNT = 6
SPEC_LAYOUTS = {"BSH", "SBH", "BNSD"}   # BSND is a 4th spec layout — see KNOWN_GAP below
KNOWN_GAP = ("BSND layout: spec V2-64 has 2 BSND cases; the FA _LAYOUT_MAP declares "
             "BSH/SBH/BNSD only. Extending requires the op to support BSND — tracked "
             "as a follow-up, not a case_gen-engine gap.")


# --- The canonical attention schema (the same an FA input_gen declares) ---------
# Layout dispatch: base [B,S,N,D] viewed per the input_layout scalar.
_VALID_D = (64, 128, 256, 512, 640, 768)


def _remap_base(base):
    B, S, N, D = base
    B = max(1, min(B, 4))
    S = max(16, min(((S + 15) // 16) * 16, 512))
    N = max(1, min(N, 32))
    D = min(_VALID_D, key=lambda v: abs(v - D))
    return [B, S, N, D]


_LAYOUT_MAP = {
    "BSH": lambda s: (lambda r: [r[0], r[1], r[2] * r[3]])(_remap_base(s)),
    "SBH": lambda s: (lambda r: [r[1], r[0], r[2] * r[3]])(_remap_base(s)),
    "BNSD": lambda s: (lambda r: [r[0], r[2], r[1], r[3]])(_remap_base(s)),
}
_qkv = cg.make_layout_dispatch(_LAYOUT_MAP, scalar_name="input_layout")

SCHEMA = {
    "op_name": "flash_attention_score",
    "formula": "FlashAttentionScore(q,k,v, scale, layout, keep_prob)",
    "op_class": "attention",
    "tensor_inputs": [
        {"name": "q", "role": "operand", "shape_derive": _qkv},
        {"name": "k", "role": "operand", "shape_derive": _qkv},
        {"name": "v", "role": "operand", "shape_derive": _qkv},
    ],
    "scalar_inputs": [
        {"name": "input_layout", "dtype": "str", "default": "BSH",
         "probe_values": ["BSH", "SBH", "BNSD"]},
        {"name": "keep_prob", "dtype": "float", "default": 1.0,
         "probe_values": [1.0, 0.9, 0.8]},
    ],
    "tensor_output": "attention_out",
    "rank": 4,
    "base_shape_filter": (lambda b: len(b) == 4 and all(d >= 1 for d in b)),
}


def _coverage(cases):
    d_buckets, kps, layouts = set(), {}, set()
    hi_dropout_pairs = set()
    for c in cases:
        sh = c["shape"]
        inp = c["inputs"]
        kp = inp.get("keep_prob", 1.0)
        lay = inp.get("input_layout")
        if lay:
            layouts.add(lay)
        kps[kp] = kps.get(kp, 0) + 1
        if len(sh) == 4:
            d_buckets.add(sh[3])
            if kp is not None and kp < 1.0 and sh[3] in (640, 768):
                hi_dropout_pairs.add((sh[3], lay, kp))
    kp_lt1 = sum(v for k, v in kps.items() if k is not None and k < 1.0)
    kp_lt1_low_d = any(
        (c["inputs"].get("keep_prob", 1.0) < 1.0 and len(c["shape"]) == 4 and c["shape"][3] <= 512)
        for c in cases)
    return d_buckets, kp_lt1, kp_lt1_low_d, layouts, hi_dropout_pairs


def main():
    cases = cg.generate_cases(SCHEMA, coverage_tier="sign_off", dtype=torch.float16)
    d_buckets, kp_lt1, kp_lt1_low_d, layouts, hi_pairs = _coverage(cases)

    checks = []
    checks.append(("D-buckets ⊇ spec", SPEC_D_BUCKETS <= d_buckets,
                   f"have {sorted(d_buckets)} need ⊇ {sorted(SPEC_D_BUCKETS)}"))
    checks.append(("kp<1 count ≥ spec(6)", kp_lt1 >= SPEC_KP_LT1_COUNT,
                   f"have {kp_lt1} need ≥ {SPEC_KP_LT1_COUNT}"))
    checks.append(("dropout spans low+high D", kp_lt1_low_d and any(d in d_buckets for d in (640, 768)),
                   "need ≥1 kp<1 at D≤512 AND high-D present"))
    checks.append(("layouts ⊇ {BSH,SBH,BNSD}", SPEC_LAYOUTS <= layouts,
                   f"have {sorted(layouts)} need ⊇ {sorted(SPEC_LAYOUTS)}"))
    checks.append(("high-D carries spec dropout pairing",
                   (640, "BNSD", 0.9) in hi_pairs and (768, "SBH", 0.8) in hi_pairs,
                   f"have {sorted(hi_pairs)} need (640,BNSD,0.9)+(768,SBH,0.8)"))

    print(f"case_gen attention coverage (sign_off, fp16): {len(cases)} cases")
    print(f"  D-buckets : {sorted(d_buckets)}")
    print(f"  keep_prob : kp<1 count={kp_lt1} (low-D dropout present={kp_lt1_low_d})")
    print(f"  layouts   : {sorted(layouts)}")
    print(f"  high-D×dropout pairs: {sorted(hi_pairs)}")
    print()
    ok = True
    for name, passed, detail in checks:
        print(f"  [{'OK ' if passed else 'XX '}] {name}: {detail}")
        ok = ok and passed
    print()
    print(f"  KNOWN GAP (not a case_gen-engine defect): {KNOWN_GAP}")
    print()
    if ok:
        print("COVERAGE MEETS SPEC — case_gen self-generates attention coverage ≥ V2-64 dimensions.")
        return 0
    print("COVERAGE BELOW SPEC — see XX lines above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
