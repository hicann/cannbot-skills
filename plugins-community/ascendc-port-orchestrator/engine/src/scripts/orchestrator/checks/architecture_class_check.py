# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Architecture-class safety-net check per OL-188 (2026-05-25, owner directive).

Detects "pure-VEC kernel generated for cube-required op-class" hack pattern —
algorithmically same anti-cheat tier as CPU fallback. Owner directive 02:16Z:
"如果 cann 的参考代码是 vec/cube 复合的融合代码，那么如果生成纯使用 vec 的代码要
看做一种 hack（与使用 cpu 一样）。"

Workflow:
  1. Detect the arch22 source family from `<port_source>` and durable tags
  2. Classify reference architecture per OL-188 markers
  3. Classify generated kernel architecture per same markers
  4. REJECT if reference=cube-required AND generated=pure-VEC

There is no waiver or target-source fallback.
"""
from __future__ import annotations
import logging

import json
import re
from pathlib import Path
from typing import Optional

from source_arch import detect_source_arch


# Cube-required marker regexes (any one present → cube architecture)
CUBE_MARKER_PATTERNS = [
    r"matmul::Matmul\s*<",          # matmul library cube instantiation
    r"MatmulImpl\s*<",              # lower-level cube primitive
    r"REGIST_MATMUL_OBJ\b",         # KFC client cube registration
    r"KERNEL_TASK_TYPE_DEFAULT\s*\(\s*KERNEL_TYPE_MIX_AIC_[12]_[12]",  # MIX_AIC task type
    r"\bcube_block\b",              # V351 cube tile-block abstraction
]
CUBE_MARKER_RE = re.compile("|".join(CUBE_MARKER_PATTERNS))

# Explicit vec-only task-type declaration. CANN declares a kernel's execution
# unit via KERNEL_TASK_TYPE_DEFAULT; an op that runs only on the vector core
# declares KERNEL_TYPE_AIV_ONLY, while a cube/MIX op declares
# KERNEL_TYPE_MIX_AIC_* or AIC. An explicit AIV_ONLY declaration is therefore
# an authoritative, op-specific vec-only signal — strictly stronger than the
# coarse "under a cube-family directory" prefix heuristic below. It overrides
# the strict family prefix (see _classify_reference_arch) so that a genuinely
# vec-only op (e.g. recurrent_gated_delta_rule, the decode-step variant) is not
# mis-classified cube-required merely because it lives under a cube-heavy family
# (ops-transformer/attention) alongside cube ops like chunk_gated_delta_rule.
# Safe: a real cube/MIX op never declares AIV_ONLY, and any cube marker present
# still wins (the override only fires when zero cube markers are found).
VEC_ONLY_DECL_RE = re.compile(r"KERNEL_TASK_TYPE_DEFAULT\s*\(\s*KERNEL_TYPE_AIV_ONLY")

# Cube-required op-family directory prefixes (from CANN 2026-05-25 grep,
# canonical list in src/skills/references/target/ascendc/cann_classification/
# cube_required_ops.txt). When port_source is under one of these prefixes,
# the op is cube-required by classification regardless of grep result.
CUBE_REQUIRED_FAMILIES_STRICT = {
    "ops-transformer/attention",
    "ops-nn/matmul",
    "ops-transformer/experimental",
    "ops-nn/conv",
    "ops-transformer/mhc",
    "ops-transformer/gmm",
    "ops-transformer/ffn",
    "ops-nn/rnn",
}

# Boundary families: cube-required for SOME variants, vec-OK for others.
# Detection should be per-variant via upstream tiling dispatch (OL-187).
BOUNDARY_FAMILIES = {
    "ops-nn/quant",                 # flat_quant has cube + vec branches
    "ops-transformer/posembedding",  # rotary_position_embedding partial
    "ops-nn/experimental",          # mixed
}


def _has_cube_markers(text: str) -> bool:
    """True if text contains any cube-architecture marker."""
    return bool(CUBE_MARKER_RE.search(text))


def _classify_reference_arch(port_source: Path) -> str:
    """Classify only source-detector-approved arch22 files.

    Two signals combined:
    1. Op-family directory prefix (strict / boundary / vec-only-default)
    2. Content scan on the detector's allowed arch22 paths for cube markers
    """
    if not port_source.is_dir():
        return "unknown"

    # Signal 1: family prefix.
    # CANN-root-agnostic (2026-06-01, design §3.3 / §8): find a CANN family-root
    # segment (`ops-nn` / `ops-transformer`) ANYWHERE in the resolved path and
    # take family = that segment + the next component (e.g. "ops-nn/matmul").
    # The previous impl hardcoded `~/workspace/cann/` .replace() → any other
    # checkout root (`/data/cann/...`, an A3-host mount, a staging copy)
    # degraded the family signal to "vec-default" and the gate could miss a
    # cube reference. The family-root scan resolves identically for every root.
    _FAMILY_ROOTS = ("ops-nn", "ops-transformer")
    family = ""
    try:
        parts = port_source.resolve().parts
        for i, p in enumerate(parts):
            if p in _FAMILY_ROOTS and i + 1 < len(parts):
                family = f"{p}/{parts[i + 1]}"
                break
    except Exception:
        family = ""

    family_signal = None
    if family in CUBE_REQUIRED_FAMILIES_STRICT:
        family_signal = "cube-required"
    elif family in BOUNDARY_FAMILIES:
        family_signal = "boundary"
    else:
        family_signal = "vec-default"

    detection = detect_source_arch(port_source)
    if not detection.supported or detection.arch != "arch22":
        return "unknown"

    # Signal 2: inspect only paths already admitted as source-side evidence.
    grep_signal = "vec-only"
    aiv_only_declared = False
    for relative in detection.analyzed_paths:
        rel_path = Path(relative)
        if "arch35" in {part.lower() for part in rel_path.parts}:
            raise RuntimeError("source detector exposed a target path")
        f = port_source / rel_path
        try:
            text = f.read_text(errors="replace")
        except OSError:
            return "unknown"
        if _has_cube_markers(text):
            grep_signal = "cube-required"
        if VEC_ONLY_DECL_RE.search(text):
            aiv_only_declared = True

    # Combine: boundary first; then an EXPLICIT vec-only declaration (with no
    # cube markers anywhere) overrides even a strict cube-family prefix — the
    # op self-declares it runs on the vector core only (see VEC_ONLY_DECL_RE);
    # then cube wins if either remaining signal says cube.
    if family_signal == "boundary":
        return "boundary"
    if aiv_only_declared and grep_signal != "cube-required":
        return "vec-only"
    if family_signal == "cube-required" or grep_signal == "cube-required":
        return "cube-required"
    return "vec-only"


def _classify_generated_arch(workspace: Path, op_name: str) -> str:
    """Classify worker-generated kernel files as 'cube-required' / 'vec-only'.

    Searches workspace/kernel/*.{cpp,h} + workspace/op_kernel/*.{cpp,h} for
    cube markers. If ANY marker found → cube-required (worker correctly used
    cube primitives). If NONE → vec-only (worker produced pure-VEC kernel).
    """
    for subdir in ("kernel", "op_kernel"):
        kdir = workspace / subdir
        if not kdir.is_dir():
            continue
        for f in kdir.rglob("*"):
            if not f.is_file() or f.suffix not in (".h", ".cpp"):
                continue
            skip_current_item = False
            try:
                text = f.read_text(errors="replace")
            except Exception as error:
                logging.getLogger(__name__).debug(
                    "Recoverable operation failed.", exc_info=error
                )
                skip_current_item = True
            if skip_current_item:
                continue
            if _has_cube_markers(text):
                return "cube-required"
    return "vec-only"


def check_architecture_class(
    workspace: Path,
    op_name: str,
    port_source: Optional[Path] = None,
) -> dict:
    """Run architecture-class check.

    Returns dict with:
      - verdict: "PASS" | "ARCHITECTURAL_HACK" | "SOURCE_ARCH_UNVERIFIED"
      - reference_arch: classification of CANN ref
      - generated_arch: classification of generated kernel
      - reason: human-readable explanation
    """
    if port_source is None:
        # Try to read port_source from .opgen_state.json
        state = workspace / ".opgen_state.json"
        if state.is_file():
            try:
                d = json.loads(state.read_text())
                ps = d.get("port_a3_source") or d.get("port_source")
                if ps:
                    port_source = Path(ps)
            except Exception as error:
                logging.getLogger(__name__).debug(
                    "Recoverable operation failed.", exc_info=error
                )

    if port_source is None or not port_source.is_dir():
        return {
            "verdict": "SOURCE_ARCH_UNVERIFIED",
            "reference_arch": "unknown",
            "generated_arch": _classify_generated_arch(workspace, op_name),
            "reason": "arch22 source evidence is unavailable — fail closed",
        }

    ref_arch = _classify_reference_arch(port_source)
    gen_arch = _classify_generated_arch(workspace, op_name)

    if ref_arch == "unknown":
        return {
            "verdict": "SOURCE_ARCH_UNVERIFIED",
            "reference_arch": ref_arch,
            "generated_arch": gen_arch,
            "reason": (
                "the supplied source directory does not contain detector-approved "
                "arch22 evidence — fail closed"
            ),
        }

    if ref_arch == "cube-required" and gen_arch == "vec-only":
        return {
            "verdict": "ARCHITECTURAL_HACK",
            "reference_arch": ref_arch,
            "generated_arch": gen_arch,
            "reason": (
                f"CANN reference at {port_source} is cube-required (op-family or "
                f"op_kernel grep matched cube markers) but generated kernel in "
                f"{workspace}/{kernel_or_op_kernel(workspace)}/ is pure-VEC (no cube "
                f"markers). Per OL-188 + owner 2026-05-25T02:16Z: pure-VEC for cube-"
                f"required op = HACK class (same tier as CPU fallback). REJECT pre-ship."
            ),
        }

    return {
        "verdict": "PASS",
        "reference_arch": ref_arch,
        "generated_arch": gen_arch,
        "reason": (
            f"Architecture class match: reference={ref_arch}, generated={gen_arch}. "
            f"No hack pattern detected."
        ),
    }


def kernel_or_op_kernel(workspace: Path) -> str:
    """Return 'kernel' or 'op_kernel' subdir name (whichever exists in workspace)."""
    if (workspace / "kernel").is_dir():
        return "kernel"
    if (workspace / "op_kernel").is_dir():
        return "op_kernel"
    return "kernel"  # default for error message


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="OL-188 architecture-class hack detector")
    ap.add_argument("--workspace", required=True, type=Path)
    ap.add_argument("--op-name", required=True)
    ap.add_argument(
        "--port-source",
        type=Path,
        default=None,
        help=(
            "CANN ops-nn / ops-transformer op dir "
            "(e.g. ~/workspace/cann/ops-transformer/attention/flash_attention_score)"
        ),
    )
    args = ap.parse_args()

    result = check_architecture_class(
        workspace=args.workspace,
        op_name=args.op_name,
        port_source=args.port_source,
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["verdict"] == "PASS" else 1)
