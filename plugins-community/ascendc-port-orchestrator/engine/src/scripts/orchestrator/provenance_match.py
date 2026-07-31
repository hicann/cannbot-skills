#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""DEBT-203 S2 — find_similar_proven_op matcher (READ-ONLY).

Design: docs/design/DEBT_203_PROVENANCE_TREE_DESIGN.md §5 (main-gated).

Given a new op's signature (op_class_tags + algorithm_classification, from its
op_classification.json), scan the archived-op corpus and return the most-similar
PROVEN op — correctness-eligible (precision PASS + determinism), fitness >= min,
non-buggy — above a similarity threshold, else None.

S2 is READ-ONLY: it does NOT seed cold-start (that is S3, gated on main's anti-cheat
sign-off) and is NOT wired on by default. Below-threshold / no-eligible-candidate
always returns None (caller then cold-starts exactly as today — strictly additive).

Reuse (not reinvent): op-class alias canonicalization from cann_learn/kb_overlap so
semantically-equivalent tags intersect; eligibility/fitness from provenance_node.
"""
from __future__ import annotations
import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import provenance_node as _pn

# Reuse kb_overlap's op-class vocabulary + alias normalization (do not reinvent a matcher).
# Import is best-effort: if the cann_learn path isn't importable, fall back to identity.
try:  # pragma: no cover - import wiring
    import sys as _sys
    import os as _os
    _CL = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "cann_learn"))
    if _CL not in _sys.path:
        _sys.path.insert(0, _CL)
    from kb_overlap import _extract_op_classes as _kb_extract_op_classes  # type: ignore
except Exception:  # pragma: no cover
    _kb_extract_op_classes = None


def _normalize_tags(tags: list) -> set:
    """Canonicalize a tag list to a comparable set. Uses kb_overlap's alias engine
    when available (so 'attention' / 'fused-attention' etc. intersect); always also
    keeps the raw lowercased tags so classifier-native vocabulary matches directly."""
    raw = {str(t).strip().lower() for t in (tags or []) if str(t).strip()}
    if _kb_extract_op_classes is not None and raw:
        try:
            canon = _kb_extract_op_classes(" ".join(sorted(raw)))
            return raw | {c.lower() for c in canon}
        except Exception:
            return raw
    return raw


def op_signature_similarity(sig_a: dict, sig_b: dict) -> float:
    """Jaccard similarity of two op signatures' (normalized) op_class_tags, with a
    small bonus when algorithm_classification agrees. 0.0 if either has no tags.
    """
    a = _normalize_tags((sig_a or {}).get("op_class_tags", []))
    b = _normalize_tags((sig_b or {}).get("op_class_tags", []))
    if not a or not b:
        return 0.0
    jac = len(a & b) / len(a | b)
    ac_a = (sig_a or {}).get("algorithm_classification")
    ac_b = (sig_b or {}).get("algorithm_classification")
    if ac_a and ac_a == ac_b:
        jac = min(1.0, jac + 0.1 * (1.0 - jac))  # nudge up without exceeding 1.0
    return round(jac, 6)


@dataclass
class ProvenOpRef:
    node_id: str
    op: str
    archive_dir: str
    similarity: float
    fitness: float
    signature: dict


def _iter_archived_ops(archive_root: Path):
    """Yield (op_dir, verification_dict) for each archived op that has a verification.json."""
    if not archive_root.is_dir():
        return
    for op_dir in sorted(archive_root.iterdir()):
        vp = op_dir / "verification.json"
        if not (op_dir.is_dir() and vp.is_file()):
            continue
        skip_current_item = False
        try:
            v = json.loads(vp.read_text())
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
        if isinstance(v, dict):
            yield op_dir, v


def find_similar_proven_op(
    signature: dict,
    *,
    archive_root: Path,
    threshold: float,
    min_fitness: float = 0.8,
) -> Optional[ProvenOpRef]:
    """Return the best PROVEN, eligible, fitness>=min_fitness archived op whose signature
    similarity to `signature` is >= threshold, else None. Read-only. Tie-break: higher
    similarity, then higher fitness.
    """
    archive_root = Path(archive_root)
    best: Optional[ProvenOpRef] = None
    best_key = (-1.0, -1.0)  # (similarity, fitness)
    for op_dir, v in _iter_archived_ops(archive_root):
        # eligibility = CORRECTNESS-ONLY (main Q1/Q3) + non-buggy + fitness floor
        if not _pn.is_branch_eligible(v) or _pn.is_op_buggy(v):
            continue
        node = v.get("provenance_node") or {}
        fitness = node.get("fitness")
        if fitness is None:
            fitness = _pn.compute_fitness(v)
        if fitness < min_fitness:
            continue
        cand_sig = node.get("signature") or {}
        sim = op_signature_similarity(signature, cand_sig)
        if sim < threshold:
            continue
        key = (sim, float(fitness))
        if key > best_key:
            best_key = key
            best = ProvenOpRef(
                node_id=node.get("node_id", f"{op_dir.name}@unknown"),
                op=op_dir.name,
                archive_dir=str(op_dir),
                similarity=sim,
                fitness=float(fitness),
                signature=cand_sig,
            )
    return best
