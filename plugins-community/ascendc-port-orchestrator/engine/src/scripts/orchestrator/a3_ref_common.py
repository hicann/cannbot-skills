# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Phase O2.5 A3-CANN reference — shared common base (decomposed 2026-07-06).

Foundational LEAF for the `phase_o25_a3_ref` facade and its sibling leaves
(a3_ref_derive / a3_ref_npu / a3_ref_validate / a3_ref_provision). Holds the
container-path translation, `.ascendc_env` locator, the two report dataclasses,
and the tiny case/kwargs shared helpers. Imports ONLY stdlib + logging_config
(+ a call-time `briefs._common` inside _ascendc_env_path, matching the original
so the DEFAULT_ASCENDC_ENV monkeypatch stays effective); NEVER imports from any
sibling a3_ref module (unidirectional edge, no cycle).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from logging_config import get_logger  # cv-agent style logger

log = get_logger(__name__)

# Type alias for the dependency-injection point used by tests.
# Signature: (shell_cmd) -> (returncode, stdout, stderr).
RunRemote = Callable[[str], tuple[int, str, str]]

_A3_CONTAINER_HOME_DEFAULT = "/home/npu_user"
# A5 sliced-container fallback; override via A3_CONTAINER_HOME in .ascendc_env.


def _ascendc_env_path() -> Path:
    """Resolve the canonical `.ascendc_env` path — single source of truth.

    DEBT-101: the three A3 config readers below previously each composed
    `workspace.parent / ".ascendc_env"` inline. For a real port_a3 run that path
    equals `briefs._common.DEFAULT_ASCENDC_ENV` (`_PROJECT_ROOT/workspace/.ascendc_env`),
    but the inline form **bypassed the canonical constant** — so a test that
    monkeypatched `briefs._common.DEFAULT_ASCENDC_ENV` (the documented patch point,
    used the same way at orchestrator.py:3401 `_bc.load_env(_bc.DEFAULT_ASCENDC_ENV)`)
    still hit whatever real `.ascendc_env` sat next to the temp workspace
    (e.g. A3_HOST=198.51.100.70 on one agent's checkout vs 198.51.100.92 on another),
    making readers + their tests fs-state-dependent.

    Accessing `_bc.DEFAULT_ASCENDC_ENV` as a module attribute at call time (NOT a
    module-load `from ... import`) keeps the monkeypatch effective — same rule the
    orchestrator dispatch path follows (orchestrator.py:3394 note).
    """
    from briefs import _common as _bc
    return _bc.DEFAULT_ASCENDC_ENV


def _a3_container_home() -> str:
    """A3 container canonical home — CONFIG-DRIVEN (`A3_CONTAINER_HOME` in `.ascendc_env`),
    falling back to the a5 default. This path is a FUNCTIONAL runtime constant (the mount
    the A3 container actually exposes), so it must be genericized via config, NOT string-
    scrubbed: a scrubbed / non-npu_user deployment sets its own `A3_CONTAINER_HOME` and the
    mount still matches (scrubbing the source literal alone would break the runtime path).
    Mirrors `A3_HOST_HOME` (_a3_host_workspace_root_from_env). Forward-refs `_ascendc_env_path`
    (defined later; resolved at call-time).

    Resolution precedence (DEBT-185): (1) `A3_CONTAINER_HOME` env var — deployment-layout-robust
    (an `export` always wins regardless of where `_ascendc_env_path()`/`_PROJECT_ROOT` resolves;
    fixes the vendored-cannbot-engine case where the engine-relative `workspace/.ascendc_env`
    is NOT the file the operator edited, so a correct file value was silently ignored and the
    scrub-broken DEFAULT leaked into the runtime path); (2) the resolved `.ascendc_env` file;
    (3) the a5 default."""
    env_override = os.environ.get("A3_CONTAINER_HOME", "").strip().strip("'\"")
    if env_override:
        return env_override
    env_path = _ascendc_env_path()
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("A3_CONTAINER_HOME="):
                val = line.split("=", 1)[1].strip().strip("'\"")
                if val:
                    return val
    return _A3_CONTAINER_HOME_DEFAULT


def _translate_to_a3_container_path(local_path: Path) -> Path:
    """Translate a local-host path to its A3-container counterpart.

    Local `/home/<local_user>/workspace/...` → A3 container
    `/home/npu_user/workspace/...`. Anything not under local user home
    (e.g. `/tmp/foo`, `/opt/bar`) is returned unchanged — the caller is
    expected to have already passed a container-namespace path in that
    case, and we don't want to silently mangle it.

    Detection: strip `/home/<one-segment>/` prefix if present; substitute
    `/home/npu_user/`. Robust to any local username.
    """
    s = str(local_path)
    if not s.startswith("/home/"):
        return local_path
    parts = s.split("/", 3)  # ['', 'home', '<user>', 'rest/of/path']
    if len(parts) < 4:
        return local_path  # /home/<user> with no tail — nothing meaningful to translate
    return Path(f"{_a3_container_home()}/{parts[3]}")


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------
@dataclass
class O25A3Report:
    """Outcome of Phase O2.5 A3-CANN reference provisioning.

    verdict:
      READY              — all artifacts emitted, a3_outputs ready for kw to consume
      MISSING_ENTRY      — no aclnn entry point found in op_dir
      MISSING_DEPS       — op_host/CMakeLists.txt references peer op(s) not in ops-nn checkout
      INPUT_GEN_FAILED   — neither UT gen_data.py nor SCHEMA case_gen could produce inputs
      BUILD_FAILED       — A3 kernel build inside container failed
      EXEC_FAILED        — A3 aclnn binary exec returned non-zero
      CAPTURE_INCOMPLETE — exec/cache "succeeded" but edge_dataset.pt has
                           n_captured < n_total a3_outputs OR a3_baseline_perf.json
                           median_ms_per_case empty (task#25). NOT fallback-eligible:
                           must fail-fast, NOT silently degrade to CPU-truth (a
                           misleading fp32 oracle for quant ops = fake-pass).
      A3_UNREACHABLE     — A3_HOST / A3_CONTAINER unreachable from local machine
      A3_BUSY            — A3 NPU AICore% above busy threshold; refuse to contaminate baseline
      SOURCE_ARCH_INCOMPLETE — DEBT-165 port-entry gate: candidate's source-arch kernel is a
                           pure `#include "<target-arch>/"` dispatch shell with no source-arch
                           algorithm to extract (deformable_offsets exemplar). Hard-reject
                           BEFORE generation; no source means there is no valid migration input.
    """
    verdict: str
    aclnn_entry: Optional[Path] = None
    gen_data_source: Optional[Path] = None
    peer_op_dependencies: list[str] = field(default_factory=list)
    a3_outputs_path: Optional[Path] = None
    a3_perf_path: Optional[Path] = None
    errors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    summary: str = ""
    # W14 marker: when verdict is "READY but no live exec attempted",
    # caller may treat as planning-only. When the live A3 exec ran,
    # this is True regardless of success/failure of the exec.
    a3_exec_attempted: bool = False
    capture_id: Optional[str] = None
    capture_manifest_path: Optional[Path] = None


@dataclass
class ModelContractResult:
    """Outcome of the Phase-O2.5 model.py↔input_gen contract validation.

    reason_code:
      OK                    — model.forward ran on a sample case, produced a tensor.
      NPU_DELEGATION        — model.py imports/calls torch_npu / npu_* / .npu(); it
                              cannot produce a CPU fp64 golden. (static check).
      SIGNATURE_MISMATCH    — forward's params don't accept the input_gen case keys
                              (extra/missing/renamed kwargs, e.g. input_layout vs layout).
      OUTPUT_SHAPE_MISMATCH — forward's output shape differs from the a3_outputs shape.
      FORWARD_RAISED        — forward raised for another reason (surfaced verbatim).
      NO_MODEL_CLASS        — model.py doesn't define a `Model` class.
      SKIPPED_NO_MODEL      — no model.py (nothing to validate — non-blocking here).
      SKIPPED_NO_INPUTS     — no edge_inputs.pt / get_input_groups() sample to test.
    """
    ok: bool
    reason_code: str
    message: str
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared case/kwargs helpers (used by derivation, capture, and validation)
# ---------------------------------------------------------------------------
def _case_model_kwargs(case):
    """Extract the model.forward args from an edge_inputs case (DEBT-199 v2, 2026-07-03).

    case_gen emits WRAPPED cases: `{idx, name, shape, inputs:{arg: tensor|scalar}, meta}` — the
    real model.forward args (tensors AND scalar args) live in the nested `case['inputs']` dict
    (per case_gen.py:8 + the FA pass_a_runner.py:105-107 unwrap pattern). Feeding the whole wrapper
    as kwargs raises `TypeError: unexpected keyword 'idx'` (the DEBT-199 v1 bug + the same latent bug
    in provision_native_capture, both disk-confirmed 43/43-fail on FA 2026-07-03). Older/flat cases
    (the kwargs dict directly, or a positional list/tuple) are returned unchanged.

    Returns the unwrapped forward-args (dict for kwargs, list/tuple for positional, else a scalar).
    """
    if (isinstance(case, dict) and isinstance(case.get("inputs"), dict)
            and any(k in case for k in ("idx", "name", "shape", "meta"))):
        return case["inputs"]
    return case


def _coerce_case_list(obj):
    """Coerce an INT-KEYED edge_dataset dict {0:case, 1:case, ...} → ordered list.

    aog-a3-author-authored run_a3_reference.py scripts sometimes emit the per-case
    dataset as an int-keyed dict (e.g. flash_attention_score: {0:{name,a3_outputs,
    meta}, ..., 42}) instead of the documented list-of-case-dict (edge_dataset.pt
    header contract: "list of {**case, a3_outputs}"). That shape has no top-level
    a3_outputs/inputs/cases key, so BOTH ingestion points mis-read it — the validator
    (_count_a3_outputs) as 0 cases (false CAPTURE_INCOMPLETE) and the grader
    (precision_eval._to_tensor_list) as 1 case (silent under-grade). Both are
    silent-schema-drift hollow/false verdicts (DEBT-195-adjacent).

    Coerces a dict whose keys are EXACTLY the contiguous ints 0..N-1 (or their str
    forms) into [obj[0], ..., obj[N-1]] so downstream sees the canonical list. Leaves
    lists + the two documented dict schemas ({inputs,a3_outputs} / 'cases'-keyed)
    untouched. Pure-stdlib (operates on the already-loaded object) so it is safe to
    mirror into the torch-light in-container grader. Robustness fix (fix-harness-for-
    next-customer), NOT a per-archive data patch — coerces any op's authored drift.

    NOTE: keep the coercion LOGIC in sync with precision_eval_port_a3_two_tier.
    _coerce_case_list (mirrored, not shared-imported, to avoid the in-container
    sibling-module import fragility that bit DEBT-184; docstrings differ per file).
    """
    if not isinstance(obj, dict) or not obj:
        return obj
    # A documented schema (aligned-lists / cases-keyed) — never touch.
    if any(k in obj for k in ("a3_outputs", "inputs", "outputs", "cases")):
        return obj
    try:
        idx = sorted(int(k) for k in obj.keys())
    except (TypeError, ValueError):
        return obj  # non-int keys → not an int-keyed dataset
    if idx != list(range(len(idx))):
        return obj  # not contiguous 0..N-1 → don't guess
    kmap = {int(k): k for k in obj.keys()}
    return [obj[kmap[i]] for i in idx]
