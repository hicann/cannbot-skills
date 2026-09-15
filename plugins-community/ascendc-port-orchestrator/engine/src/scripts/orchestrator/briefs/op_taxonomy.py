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
- `default_kb_sections(target)` (still used) provides bookshelf-level baseline
- See `src/skills/aog-op-classify/SKILL.md` for procedure

Transition shim: `lookup()` still exists; it now reads
`op_classification.json` if present, else falls back to `default_kb_sections()`
only (no more bench-name-keyed dict, no regex source-scan).

OKF-only 迁移（2026-08）：legacy KB（旧索引 + 旧 manifest 渲染、legacy
路径重写、manifest 落盘校验）已随 brief_kb 的 legacy manifest 一起退役；
b-tier 唯一路径是 OKF 检索（见 `briefs/brief_kb.py`）。本模块保留
`lookup()` 的 tags 产出（OKF 查询词来源）与 target-aware 硬件规格路由。

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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Default safe set always loaded for every op (regardless of tags).
# Target-INDEPENDENT base — same on every backend.
# OKF-only 迁移（2026-08）：旧导航索引与旧 target 目录条目已移除
# （知识指针由 OKF 检索产出，见 brief_kb._okf_reference_block）；这里只剩
# 与知识来源无关的纪律文档，硬件规格经 TARGET_HW_SPEC_MAP 按 target 追加。
_DEFAULT_KB_SECTIONS_BASE: list[str] = [
    "shared/ALWAYS_LOADED_RULES.md",                       # OL-* always-loaded rules
]

# P0abj (2026-05-08): target-aware hardware-spec dispatch. Pre-fix, the
# default manifest hardcoded the a5 hardware spec regardless of
# TARGET — A3/A2 op-gen on DS env was loading A5 specs (UB size, AIV count,
# atomics, register file all wrong for V220). DS-flagged when reviewing
# whether ascend950pr.md could be removed.
TARGET_HW_SPEC_MAP: dict[str, str] = {
    "a5": "okf/runbooks/hardware/target-ascend950pr.md",   # Ascend950PR — V351 / arch35
    "a3": "okf/runbooks/hardware/target-ascend910c.md",    # Ascend910 V220 single-die (910C)
    "a2": "okf/runbooks/hardware/target-ascend910b.md",    # Ascend910 V220 single-die (910B)
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


# OKF-only 迁移（2026-08）：`_LEGACY_PATH_REWRITE` / `resolve_legacy_kb_path` /
# `validate_manifest_paths` / `KBManifestMissingError` / `DEFAULT_KB_SECTIONS`
# 别名均已删除——它们是 legacy manifest 渲染链路的组成部分，随
# brief_kb.kb_manifest_block 的 legacy 分支一起退役（b-tier 唯一路径 = OKF）。


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
    selects `okf/runbooks/hardware/target-<chip>.md` from TARGET_HW_SPEC_MAP. Pre-fix the
    default was hardcoded to `ascend950pr.md`, so A3/A2 op-gen loaded A5
    hw specs (wrong UB size, AIV count, atomics info). Defaults to "a5" for
    callers that haven't been updated yet.

    Resolution order:
      1. If workspace + workspace/op_classification.json exist → use it
      2. Else: default_kb_sections(target) only, mark is_untagged_fallback=True
         (signal: Phase O1.7 hasn't run, OR running standalone without
         classification — caller should run classify() first or warn)

    `_SOURCE_SCAN_SIGNATURES` is NOT consulted. (`OP_TAGS` / `TAG_KB_SECTIONS`
    used to sit here too, kept "for emergency rollback"; they were removed on
    2026-09-05 — 9 of the 10 KB paths they named had been deleted with the legacy
    `kb/target/` tree, so the rollback they promised was no longer possible.)

    Args:
        op: op name / workspace dir name
        workspace: path to op's workspace dir; if None, falls through to
            untagged-fallback (no classification possible)
        target: "a5" | "a3" | "a2" (case-insensitive; -ds suffix stripped).
            Selects okf/runbooks/hardware/target-<chip>.md.

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


    args = ap.parse_args()

    if args.cmd == "lookup":
        t = lookup(args.op)
        print(json.dumps({
            "op": t.op,
            "tags": t.tags,
            "kb_sections": t.kb_sections,
            "is_untagged_fallback": t.is_untagged_fallback,
        }, indent=2))
