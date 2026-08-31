# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-NEW (2026-05-14, user 2026-05-15T00:00Z): output project metadata.

User directive: "你需要在output的项目目录里记录项目信息，用于识别使用的模式。
以及其他选哟记录的信息" — output project dirs need a metadata file that
records WHICH op-gen mode they use, so the safety net (scan_delegation_cheating
+ finalize gates) can apply mode-appropriate checks without guessing from
file shape.

Schema: output/<project>/PROJECT.json
{
  "schema_version": 1,
  "project": "<dir name>",
  "opgen_mode": "port_a3_to_a5" | "backward",
  "source": {
    "type": "<cann_ops_nn | forward_spec>",
    "path": "<absolute or repo-relative path to source>",
    "repo": "<source repo url / branch if applicable>",
    "commit": "<source commit sha if known>"
  },
  "reference_baseline": "<a3_cann | npubench | cpu_fp64_autograd>",
  "target_chip": "<Ascend950PR | Ascend910_V220 | Ascend910b>",
  "cohort": {
    "size": <int>,
    "task_source": "<file or doc that listed the ops>"
  },
  "created": "<UTC iso8601>",
  "owner_agent": "<who created this project, e.g. 'ascendc-op-gen-skill'>"
}

Purpose:
1. **Mode discovery**: scan_delegation_cheating reads opgen_mode and applies
   appropriate file-name patterns + scan rules (no more guessing from
   presence of model_new_ascendc.py vs op_kernel/arch35/).
2. **Reference baseline check**: finalize_pipeline can verify
   verification.json.truth_source matches the project's declared
   reference_baseline (arch22→arch35 migration may reference its live A3
   capture or a frozen NPUKernelBench task; backward generation
   must reference CPU/fp64 autograd).
3. **Cross-project audits**: gen_e2e_cost_report can scan all output
   projects, read their declared modes, and emit mode-broken-down stats.

Helpers in this module:
- read_project_metadata(project_dir) → dict or None
- write_project_metadata(project_dir, **fields) — idempotent, fills
  in created/schema_version defaults
- detect_mode_for_workspace(workspace) → str: lookup the workspace's
  parent project metadata to disambiguate mode in finalize gate contexts
"""
from __future__ import annotations
import logging

import json
from pathlib import Path
from typing import Optional
import datetime as _dt


SCHEMA_VERSION = 1
PROJECT_METADATA_FILENAME = "PROJECT.json"

VALID_OPGEN_MODES = frozenset({
    "port_a3_to_a5",
    "backward",
})

VALID_REFERENCE_BASELINES = frozenset({
    "a3_cann",      # port_a3_to_a5 default
    "npubench",     # port_a3_to_a5 with frozen old-format NPUKernelBench task
    "cpu_fp64_autograd",  # backward-generation reference
})


def read_project_metadata(project_dir: Path) -> Optional[dict]:
    """Read output/<project>/PROJECT.json. Returns None if missing or malformed."""
    fp = project_dir / PROJECT_METADATA_FILENAME
    if not fp.is_file():
        return None
    try:
        return json.loads(fp.read_text())
    except Exception:
        return None


def write_project_metadata(
    project_dir: Path,
    *,
    opgen_mode: str,
    source_type: str,
    source_path: str,
    reference_baseline: str,
    target_chip: str,
    project_name: Optional[str] = None,
    cohort_size: Optional[int] = None,
    cohort_task_source: Optional[str] = None,
    source_repo: Optional[str] = None,
    source_commit: Optional[str] = None,
    owner_agent: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Write output/<project>/PROJECT.json. Validates fields, fills defaults.

    Raises ValueError if opgen_mode / reference_baseline not in valid set,
    or if file already exists and overwrite=False.
    """
    if opgen_mode not in VALID_OPGEN_MODES:
        raise ValueError(
            f"opgen_mode={opgen_mode!r} not in {sorted(VALID_OPGEN_MODES)}"
        )
    if reference_baseline not in VALID_REFERENCE_BASELINES:
        raise ValueError(
            f"reference_baseline={reference_baseline!r} not in "
            f"{sorted(VALID_REFERENCE_BASELINES)}"
        )
    fp = project_dir / PROJECT_METADATA_FILENAME
    if fp.exists() and not overwrite:
        raise ValueError(
            f"{fp} already exists; pass overwrite=True to replace"
        )
    project_dir.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "schema_version": SCHEMA_VERSION,
        "project": project_name or project_dir.name,
        "opgen_mode": opgen_mode,
        "source": {
            "type": source_type,
            "path": source_path,
        },
        "reference_baseline": reference_baseline,
        "target_chip": target_chip,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if source_repo:
        payload["source"]["repo"] = source_repo
    if source_commit:
        payload["source"]["commit"] = source_commit
    if cohort_size is not None or cohort_task_source is not None:
        payload["cohort"] = {}
        if cohort_size is not None:
            payload["cohort"]["size"] = cohort_size
        if cohort_task_source is not None:
            payload["cohort"]["task_source"] = cohort_task_source
    if owner_agent:
        payload["owner_agent"] = owner_agent

    fp.write_text(json.dumps(payload, indent=2) + "\n")
    return fp


def detect_mode_for_workspace(workspace: Path) -> Optional[str]:
    """Locate the parent output project for `workspace` and return its
    opgen_mode. Workspace paths follow `workspace/<op>/` (working) or
    `output/<project>/src/kernels/<op>/` (archived). For working
    workspaces, fall back to .opgen_state.json opgen_mode field.

    Returns None if metadata unavailable (caller may fail-open or use
    legacy heuristics).
    """
    # Try archive path: output/<project>/src/kernels/<op>/
    if workspace.parent.name == "kernels" and workspace.parent.parent.name == "src":
        project_dir = workspace.parent.parent.parent
        meta = read_project_metadata(project_dir)
        if meta:
            return meta.get("opgen_mode")
    # Fall back to .opgen_state.json
    state_fp = workspace / ".opgen_state.json"
    if state_fp.is_file():
        try:
            return json.loads(state_fp.read_text()).get("opgen_mode")
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
    return None
