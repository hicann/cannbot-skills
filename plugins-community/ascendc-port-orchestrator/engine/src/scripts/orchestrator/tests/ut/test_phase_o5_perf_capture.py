# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression tests for phase_o5_perf_capture (task #17, 2026-05-18).

Design: docs/design/PERF_METHODOLOGY_NOTES.md#phase-o5-perf-capture-design-2026-05-18

The module's job is to take perf measurement OUT of the worker context
and into orchestrator-driven territory. Local backward execution is supported;
target-specific remote paths return honest NOT_VERIFIED results.

These tests pin the dispatch surface, result schema, and stub
behavior. They do NOT exercise the actual torch_npu.profiler subprocess
— that requires real NPU hardware and lives in the foreach_sqrt pilot
integration (migration plan step 2).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors
import pytest

# Import the module under test by direct path to avoid PYTHONPATH wiring.
_MODULE_PATH = _reorg_paths.ORCH_DIR / "phase_o5_perf_capture.py"


def _load_module():
    import importlib.util
    # Python 3.13 dataclass requires cls.__module__ to be resolvable via
    # sys.modules during exec_module — register before loading.
    if "phase_o5_perf_capture" in sys.modules:
        return sys.modules["phase_o5_perf_capture"]
    spec = importlib.util.spec_from_file_location(
        "phase_o5_perf_capture", _MODULE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase_o5_perf_capture"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_module_imports_without_torch() -> None:
    """phase_o5_perf_capture must NOT import torch at module-load time.

    Critical: this module is loaded into the orchestrator main process.
    The orchestrator is pure-Python; torch_npu only lives in the
    subprocess harness. If module-load required torch_npu the
    orchestrator would crash on dev hosts without an NPU.
    """
    mod = _load_module()
    assert hasattr(mod, "measure_op_perf")
    assert hasattr(mod, "PerfResult")
    assert hasattr(mod, "is_enabled")


def test_is_enabled_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Feature flag is OFF by default — pilot-then-default-on per design."""
    mod = _load_module()
    monkeypatch.delenv("AOG_PERF_CAPTURE", raising=False)
    assert mod.is_enabled() is False


def test_is_enabled_on_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    for val in ("1", "true", "TRUE", "yes"):
        monkeypatch.setenv("AOG_PERF_CAPTURE", val)
        assert mod.is_enabled() is True, f"value {val!r} should enable"


def test_is_enabled_off_when_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    monkeypatch.setenv("AOG_PERF_CAPTURE", "0")
    assert mod.is_enabled() is False


def test_port_a3_returns_not_verified(tmp_path: Path) -> None:
    """port_a3 dispatch is stubbed in v1 — must NOT silently return None
    or a fake ratio. DEBT-140 follow-up: the honest "could not measure" outcome
    is canonical N/A + reason (was NOT_VERIFIED_SAME_METHOD, which the finalize
    gate rejected → LOOP-BREAK). The diagnostic stays in `error`/`reason`.
    """
    mod = _load_module()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "model.py").write_text("# stub")
    (workspace / "model_new_ascendc.py").write_text("# stub")

    plugin = SimpleNamespace(name="port_a3")
    result = mod.measure_op_perf("foreach_neg", workspace, plugin=plugin)

    assert result["status"] == "N/A"
    assert result.get("reason")
    assert "port_a3" in result.get("error", "").lower()
    assert "ratio" not in result or result.get("ratio") is None


def test_missing_candidate_returns_not_verified(tmp_path: Path) -> None:
    """No model_new_*.py → canonical N/A + reason naming the gap (DEBT-140)."""
    mod = _load_module()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "model.py").write_text("# stub")
    # No candidate file.

    plugin = SimpleNamespace(name="backward")
    result = mod.measure_op_perf("13_Cat", workspace, plugin=plugin)
    assert result["status"] == "N/A"
    assert result.get("reason")
    assert "candidate" in result.get("error", "").lower()


def test_candidate_filename_supported_default(tmp_path: Path) -> None:
    """A supported workflow resolves the canonical AscendC candidate."""
    mod = _load_module()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "model_new_ascendc.py").write_text("# stub")

    resolved = getattr(mod, '_candidate_filename_for')("backward", workspace)
    assert resolved == "model_new_ascendc.py"


def test_build_result_pass(tmp_path: Path) -> None:
    """ratio ≥ 1.0 → PASS verdict."""
    mod = _load_module()
    result = getattr(mod, '_build_result')(
        method="test",
        ref_samples=[200.0, 200.0, 200.0],
        cand_samples=[100.0, 100.0, 100.0],
        threshold=0.9,
    )
    assert result.status == "PASS"
    assert result.ratio == 2.0


def test_build_result_within_tolerance() -> None:
    mod = _load_module()
    result = getattr(mod, '_build_result')(
        method="test",
        ref_samples=[95.0, 95.0, 95.0],
        cand_samples=[100.0, 100.0, 100.0],
        threshold=0.9,
    )
    assert result.status == "PASS_WITHIN_TOLERANCE"
    assert result.ratio == 0.95


def test_build_result_below_threshold() -> None:
    mod = _load_module()
    result = getattr(mod, '_build_result')(
        method="test",
        ref_samples=[50.0, 50.0, 50.0],
        cand_samples=[100.0, 100.0, 100.0],
        threshold=0.9,
    )
    assert result.status == "BELOW_THRESHOLD"
    assert result.ratio == 0.5


def test_build_result_zero_candidate_not_verified() -> None:
    """Defensive: cand_us=0 doesn't divide-by-zero — returns canonical N/A +
    reason (DEBT-140 follow-up; was NOT_VERIFIED_SAME_METHOD).
    """
    mod = _load_module()
    result = getattr(mod, '_build_result')(
        method="test",
        ref_samples=[100.0],
        cand_samples=[0.0],
        threshold=0.9,
    )
    assert result.status == "N/A"
    assert result.ratio is None
    assert "non-positive" in result.error
    assert result.reason and "non-positive" in result.reason


def test_method_label_p141_compatible() -> None:
    """Method label format must include the keywords P141 gate expects
    when looking for the 'profiler-CSV' gold-standard method.
    """
    mod = _load_module()
    label = mod.METHOD_LABEL.format(warmup=5, repeats=5)
    assert "torch_npu.profiler.profile" in label
    assert "operator_details.csv" in label
    assert "Device Self Duration(us)" in label
    assert "symmetric" in label


def test_perf_result_to_dict_drops_error_when_none() -> None:
    """Clean PASS result should NOT carry an `error` key — keeps the
    verification.json schema tidy.
    """
    mod = _load_module()
    result = mod.PerfResult(
        method="m", status="PASS", ratio=1.5,
        reference_us=150.0, candidate_us=100.0,
    )
    d = result.to_dict()
    assert "error" not in d
    assert d["status"] == "PASS"


def test_perf_result_to_dict_preserves_error_when_set() -> None:
    mod = _load_module()
    result = mod.PerfResult(
        method="m", status="N/A", error="reason here", reason="reason here",
    )
    d = result.to_dict()
    assert d.get("error") == "reason here"
    assert d["status"] == "N/A"


def test_parse_harness_output_well_formed() -> None:
    mod = _load_module()
    stdout = """some logging noise
PHASE_O5_PERF_RESULT={"ref": [100.0, 105.0], "cand": [50.0, 52.0]}
trailing log"""
    ref, cand = getattr(mod, '_parse_harness_output')(stdout)
    assert ref == [100.0, 105.0]
    assert cand == [50.0, 52.0]


def test_parse_harness_output_missing_marker_raises() -> None:
    mod = _load_module()
    with pytest.raises(getattr(mod, '_PerfCaptureFailure'), match="did not emit"):
        getattr(mod, '_parse_harness_output')("no marker line at all\n")


def test_parse_harness_output_invalid_json_raises() -> None:
    mod = _load_module()
    with pytest.raises(getattr(mod, '_PerfCaptureFailure'), match="JSON parse failed"):
        getattr(mod, '_parse_harness_output')("PHASE_O5_PERF_RESULT={not valid json}")


def test_parse_harness_output_empty_samples_raises() -> None:
    mod = _load_module()
    with pytest.raises(getattr(mod, '_PerfCaptureFailure'), match="empty samples"):
        getattr(mod, '_parse_harness_output')('PHASE_O5_PERF_RESULT={"ref": [], "cand": []}')


def test_version_string_present() -> None:
    """Schema version tag must be on every returned dict for forward-compat."""
    mod = _load_module()
    result = getattr(mod, '_not_verified')("test_method", "test_reason")
    assert result.get("phase_o5_perf_capture_version") == "1.0"


def test_dispatch_unknown_plugin_falls_through_to_default(tmp_path: Path) -> None:
    """Unknown plugin name → falls through to
    default benchmark-style local execution. Caller should not see crash.

    Since we don't actually have torch_npu in test env, the subprocess
    will fail; we just verify the dispatch path doesn't raise.
    """
    mod = _load_module()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "model.py").write_text("# stub no torch")
    (workspace / "model_new_ascendc.py").write_text("# stub no torch")

    plugin = SimpleNamespace(name="weird_unknown_plugin")
    result = mod.measure_op_perf("test_op", workspace, plugin=plugin)
    # Subprocess will fail because stubs don't define Model / ModelNew;
    # honest "could not measure" is canonical N/A + reason (DEBT-140 follow-up).
    assert result["status"] == "N/A"
    assert result.get("reason")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))
