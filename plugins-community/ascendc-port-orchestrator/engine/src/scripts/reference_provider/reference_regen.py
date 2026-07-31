# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Re-provision the O2.5 CPU-truth reference (edge_dataset.pt) for `--optimize`
re-entry from a committed `reference_regen` recipe.

harness-gap #1 — cpu_pytorch variant (scan 2026-07-24; main's unified
`.opgen_state.json` `reference_regen` schema; OL-282 / DEBT-158 re-entry-reference gap).

Problem it closes: a verified op's O2.5 CPU-truth reference (`edge_dataset.pt`) is
transient/gitignored and is NOT hydrated on `--optimize` re-entry
(`_optimize_reentry_workspace` requires a pre-existing workspace and does not copy
from the op archive), so re-entry hard-blocks at O2.5 (surfaced 2026-07-24 on
selective_scan_fwd_simd re-optimize). This regenerates the reference deterministically
from a committed recipe.

Schema (`.opgen_state.json` `reference_regen`):
    truth_source:      "cpu_pytorch"                  (= reference_baseline)
    input_gen:         {path, sha256}                 (regenerate the inputs)
    input_data_sha256: str                            (verify input determinism)
    generator:         {path, sha256}                 (CPU-truth generator)
    env:               {...}                          (minimal execution environment)
    combine_cmd:       [argv, ...] | str              (combine -> edge_dataset.pt)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def regen_reference(workspace: Path) -> bool:
    """Regenerate the O2.5 reference from a committed `reference_regen` recipe.

    Returns True iff a reference was regenerated. Returns False (no-op) when there is
    no `reference_regen` block — callers keep existing behavior unchanged (the hook is
    additive). Raises RuntimeError on recipe-integrity / input-determinism / step failure — a wrong
    reference must NEVER be silently produced (fail-loud).
    """
    state_path = workspace / ".opgen_state.json"
    if not state_path.is_file():
        return False
    try:
        rr = json.loads(state_path.read_text()).get("reference_regen")
    except Exception:
        return False
    if not isinstance(rr, dict) or not rr:
        return False

    truth = rr.get("truth_source")
    if truth != "cpu_pytorch":
        return False  # unknown truth_source → no-op (fail-open: leave existing behavior)

    # ---- cpu_pytorch: deterministic CPU regeneration ----
    def _resolve(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (workspace / p)

    # (1) recipe integrity — committed input_gen / generator must be untampered.
    for key in ("input_gen", "generator"):
        spec = rr.get(key)
        if not isinstance(spec, dict) or "path" not in spec or "sha256" not in spec:
            raise RuntimeError(f"reference_regen.{key} missing path/sha256")
        script = _resolve(spec["path"])
        if not script.is_file():
            raise RuntimeError(f"reference_regen.{key} script absent: {script}")
        got = _sha256_file(script)
        if got != spec["sha256"]:
            raise RuntimeError(
                f"reference_regen.{key} sha256 mismatch ({script}): committed "
                f"{spec['sha256'][:12]} != on-disk {got[:12]} — recipe tampered/stale"
            )

    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (rr.get("env") or {}).items()})
    py = env.get("REFERENCE_REGEN_PYTHON", sys.executable)

    def _run(argv, step):
        r = subprocess.run(argv, cwd=str(workspace), env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"reference_regen cpu_pytorch step '{step}' failed (rc={r.returncode}): "
                f"{(r.stderr or r.stdout)[-400:]}"
            )

    # (2) regen inputs → edge_inputs.pt (+ manifest.json)
    _run([py, str(_resolve(rr["input_gen"]["path"]))], "input_gen")
    edge_inputs = workspace / "edge_inputs.pt"
    if not edge_inputs.is_file():
        raise RuntimeError("reference_regen: input_gen did not produce edge_inputs.pt")

    # (3) verify input DETERMINISM — regen'd inputs must match the committed hash, else
    #     the CPU-truth would be computed on drifted inputs (refuse to build it).
    idsha = rr.get("input_data_sha256")
    if idsha:
        got = _sha256_file(edge_inputs)
        if got != idsha:
            raise RuntimeError(
                f"reference_regen: regen'd edge_inputs.pt sha256 {got[:12]} != committed "
                f"input_data_sha256 {idsha[:12]} — non-deterministic input regen; refusing "
                f"to build a reference on drifted inputs"
            )

    # (4) generator (Path-A CPU-truth) → outputs
    _run([py, str(_resolve(rr["generator"]["path"]))], "generator")

    # (5) combine → edge_dataset.pt
    combine = rr.get("combine_cmd")
    if not combine:
        raise RuntimeError("reference_regen.combine_cmd missing")
    if isinstance(combine, str):
        combine = combine.split()
    _run([str(c) for c in combine], "combine")

    if not (workspace / "edge_dataset.pt").is_file():
        raise RuntimeError("reference_regen: combine did not produce edge_dataset.pt")
    return True
