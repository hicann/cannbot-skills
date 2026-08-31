# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Direct O2.5 coverage for the native NPUKernelBench target route."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import _reorg_paths  # noqa: F401  (stable sys.path setup for reorganized tests)

import phase_o25_npubench as phase


def test_o25_delegates_native_preflight_to_target_transport(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reference = {"source": "npubench", "bundle_sha256": "a" * 64}
    observed: list[dict] = []
    emitted: list[tuple] = []
    target = types.ModuleType("npubench.npubench_target")

    def preflight_npubench_on_target(**kwargs):
        observed.append(kwargs)
        return {
            "status": "PASS",
            "binding_sha256": "b" * 64,
            "target_receipt_path": "npubench_evidence/preflight_target_receipt.json",
        }

    target.preflight_npubench_on_target = preflight_npubench_on_target
    monkeypatch.setitem(sys.modules, "npubench.npubench_target", target)
    monkeypatch.setattr(phase, "verify_npubench_stage", lambda *_args: (True, "ok", {"sidecar_encoding": "jsonl"}))
    monkeypatch.setattr(phase.events, "emit", lambda *args, **kwargs: emitted.append((args, kwargs)))

    assert phase.provision_npubench_reference(workspace=workspace, reference=reference, lane=3) is None
    assert observed == [{"workspace": workspace, "reference": reference, "lane": 3}]
    assert emitted[-1][0][1] == "orchestrator.phase_o25_npubench_ready"
    assert emitted[-1][1]["data"]["binding_sha256"] == "b" * 64


def test_o25_target_failure_never_falls_back_to_controller(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reference = {"source": "npubench", "bundle_sha256": "a" * 64}
    target = types.ModuleType("npubench.npubench_target")

    def preflight_npubench_on_target(**_kwargs):
        return {"status": "ERROR", "reason": "target dependency is unavailable"}

    target.preflight_npubench_on_target = preflight_npubench_on_target
    monkeypatch.setitem(sys.modules, "npubench.npubench_target", target)
    monkeypatch.setattr(phase, "verify_npubench_stage", lambda *_args: (True, "ok", {}))
    monkeypatch.setattr(phase.events, "emit", lambda *_args, **_kwargs: None)

    assert phase.provision_npubench_reference(workspace=workspace, reference=reference, lane=0) == 7
