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
"""finalize_checks — finalize-eligibility gate CHECK functions (stable façade).

Originally extracted from finalize_pipeline.py (behavior-neutral god-file
decomposition, 2026-07-05). DEBT-201 sub-split (2026-07-06): the 25
`_check_*(workspace, ...) -> Optional[str]` gate functions are grouped by
CATEGORY into cohesive sibling modules and re-imported here (bottom import),
so every call site + import path (`from finalize_checks import _check_...` and
`finalize_checks._check_...` attribute access) stays byte-for-byte stable. The
gate-ID mapping + dispatch continue to live in finalize_pipeline.

Sibling modules (each imports its own shared helpers from finalize_pipeline):
  - finalize_checks_precision   : pass-count / coverage / performance methodology
  - finalize_checks_infra       : baseline-blame / retry-budget / paper-over
  - finalize_checks_provenance  : binary & verify-path provenance / KB / plugin dispatch
  - finalize_checks_structural  : code-shape / entrypoint / architecture / topology

Two checks intentionally stay in finalize_pipeline: _check_stale_orchestrator
(monkeypatched via finalize_pipeline module attrs) + _check_op_host_completeness
(source-grep contract test reads finalize_pipeline.py). finalize_pipeline
re-imports the names below (its own bottom import) so call sites are unaffected."""
from __future__ import annotations

# --- gate checks split by category into sibling modules (behavior-neutral,
#     DEBT-201, 2026-07-06). Re-export so `from finalize_checks import _check_*`
#     and `finalize_checks._check_*` attribute access both keep resolving. The
#     siblings themselves `from finalize_pipeline import ...` shared helpers,
#     which are already defined by the time this bottom-imported module loads. ---
from finalize_checks_precision import (
    _check_pass_a_coverage, _check_port_a3_pass_b_schema, _check_pass_b_coverage,
    _check_pass_count_concrete, _check_perf_methodology, _check_methodology_declaration,
)
from finalize_checks_infra import (
    _check_platform_blame_backed, _check_infra_paper_over, _check_infra_retry_budget,
)
from finalize_checks_provenance import (
    _check_binary_provenance, _check_a5_verify_path_provenance,
    _check_ge_ophost_raw_cann_copy, _check_kb_writeup,
    _check_verifier_uses_modelnew, _check_post_worker_audit,
    _check_delegation_scan_marker,
)
from finalize_checks_structural import (
    _check_model_py_shape, _check_universal_entrypoints, _check_arch35_wrap_cheat,
    _check_architecture_class, _check_project_json_metadata,
    _check_pp88_compliance, _check_pybind_host_logic,
)
