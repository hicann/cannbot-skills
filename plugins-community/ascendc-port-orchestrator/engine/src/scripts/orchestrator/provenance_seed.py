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
"""DEBT-203 S3 — migration cold-start seeding from proven prior art (default-OFF).

Design: docs/design/DEBT_203_S3_ANTICHEAT_INTERACTION.md (main-signed 2026-07-11).

At a new op's cold-start (AFTER the wipe, before the first `await_worker` spawn), if
the DEBT-203 seed flag is ON and `find_similar_proven_op` (S2) returns a proven,
correctness-eligible, non-buggy, fitness>=floor archived op, copy that op's `kernel/`
into `branched_from_kernel/` (a distinct NON-output path that dodges every cold-start
wipe pattern) as a labeled BRANCH BASE — and record `branched`/`parent_id` provenance.

Safety: `branched_from_kernel/` is advisory context, not truth or a submission.
The worker must adapt it into `kernel/`, record the reference path, and re-verify
against THIS op's fresh arch22 source-NPU capture on the target NPU. Lineage is
threaded into final provenance, while ordinary acceptance gates remain unchanged.

Default-OFF: seeding only runs when env `AOG_DEBT203_SEED_ENABLED` is truthy.
"""
from __future__ import annotations
import os
import json
import shutil
from pathlib import Path
from typing import Optional

import provenance_match as _pm

SEED_DIR_NAME = "branched_from_kernel"   # distinct non-output path (dodges all wipe patterns)
SEED_FLAG_ENV = "AOG_DEBT203_SEED_ENABLED"
SEED_MARKER = ".branched_from.json"      # provenance/telemetry marker at workspace root

# conservative defaults (main-gated): high similarity bar + full-correctness fitness floor
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_FITNESS = 0.8


def seed_enabled() -> bool:
    """True iff the DEBT-203 cold-start seeding flag is ON (default OFF)."""
    return str(os.environ.get(SEED_FLAG_ENV, "")).strip().lower() in ("1", "true", "yes", "on")


def _read_op_signature(workspace: Path) -> Optional[dict]:
    cls = workspace / "op_classification.json"
    if not cls.is_file():
        return None
    try:
        d = json.loads(cls.read_text())
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    return {"op_class_tags": d.get("op_class_tags", []),
            "algorithm_classification": d.get("algorithm_classification")}


def maybe_seed_from_similar_op(
    op: str,
    workspace: Path,
    *,
    archive_root: Path,
    threshold: float = DEFAULT_THRESHOLD,
    min_fitness: float = DEFAULT_MIN_FITNESS,
    force_enabled: Optional[bool] = None,
) -> Optional[_pm.ProvenOpRef]:
    """Post-wipe cold-start hook. Returns the ProvenOpRef seeded from, or None.

    None (no-op) when: flag OFF, no signature, no eligible similar op, or the source
    has no kernel/ to copy. On a hit: copy source kernel/ → workspace/branched_from_kernel/
    and drop a .branched_from.json provenance marker. NEVER writes workspace/kernel/
    (the worker authors that). Fail-open: any error → None (cold-start proceeds as today).
    """
    enabled = seed_enabled() if force_enabled is None else force_enabled
    if not enabled:
        return None
    try:
        workspace = Path(workspace)
        sig = _read_op_signature(workspace)
        if not sig:
            return None
        ref = _pm.find_similar_proven_op(
            sig, archive_root=Path(archive_root), threshold=threshold, min_fitness=min_fitness)
        if ref is None:
            return None
        src_kernel = Path(ref.archive_dir) / "kernel"
        if not src_kernel.is_dir():
            return None  # nothing to seed from
        seed_dir = workspace / SEED_DIR_NAME
        if seed_dir.exists():
            shutil.rmtree(seed_dir, ignore_errors=True)
        shutil.copytree(src_kernel, seed_dir)
        (workspace / SEED_MARKER).write_text(json.dumps({
            "branched": True,
            "parent_id": ref.node_id,
            "parent_op": ref.op,
            "similarity": ref.similarity,
            "parent_fitness": ref.fitness,
            "seed_dir": SEED_DIR_NAME,
        }, indent=2))
        return ref
    except Exception:
        return None
