# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for verify_candidate — Phase 4 of aog-prior-art-verify."""
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from verify_candidate import (  # noqa: E402
    verify, write_verify_report,
    classify_precision_case, aggregate_precision_status,
    aggregate_perf_status, PrecisionCase, _default_invoker,
)
from stage_candidate import compute_candidate_digest  # noqa: E402


def _setup_workspace(tmp_path: Path, op: str = "op_a",
                      build_verdict: str = "SUCCESS",
                      so_exists: bool = True) -> Path:
    workspace = tmp_path / "ws" / op
    workspace.mkdir(parents=True)
    candidate_dir = workspace / ".prior_art_candidate"
    source = candidate_dir / "op_kernel" / "arch35" / f"{op}.h"
    source.parent.mkdir(parents=True)
    source.write_text("// candidate")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    files = [{
        "rel_path": f"op_kernel/arch35/{op}.h",
        "build_rel_path": f"norm/{op}/op_kernel/arch35/{op}.h",
        "sha": source_sha,
        "source_type": "upstream_arch35",
        "origin": f"/source/{op}.h",
    }]
    manifest = {
        "schema_version": 2,
        "op": op,
        "candidate_dir": str(candidate_dir),
        "repo_name": "ops-nn",
        "op_repo_rel_path": f"norm/{op}",
        "sources_staged": ["upstream_arch35"],
        "file_count": 1,
        "files": files,
        "candidate_digest": compute_candidate_digest(op, "ops-nn", files),
        "warnings": [],
        "errors": [],
    }
    manifest_path = candidate_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    build_out = candidate_dir / "build"
    build_out.mkdir(parents=True)
    so_path = build_out / f"{op}_kernels.so"
    if so_exists:
        so_path.write_text("ELF mock")
    so_sha = hashlib.sha256(so_path.read_bytes()).hexdigest() if so_exists else None
    (workspace / ".prior_art_build.json").write_text(json.dumps({
        "schema_version": 2,
        "op": op, "verdict": build_verdict,
        "so_path": str(so_path) if so_exists else None,
        "so_sha256": so_sha,
        "candidate_digest": manifest["candidate_digest"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }))
    (workspace / "edge_dataset.pt").write_text("mock edge dataset")
    (workspace / "a3_baseline_perf.json").write_text('{"cases": []}')
    (workspace / "a3_reference_runnable.json").write_text(json.dumps({
        "verdict": "READY",
        "a3_exec_attempted": True,
        "capture_id": "capture-current-run",
    }))
    return workspace


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_classify_t1_exact() -> None:
    t1, t2 = classify_precision_case(0.0, 0.0)
    assert t1 and t2


def test_classify_t2_only() -> None:
    t1, t2 = classify_precision_case(5e-4, 5e-4)
    assert not t1
    assert t2


def test_classify_fail() -> None:
    t1, t2 = classify_precision_case(1e-2, 1e-2)
    assert not t1 and not t2


def test_aggregate_precision_all_t1() -> None:
    cases = [PrecisionCase(f"c{i}", 0.0, 0.0, True, True) for i in range(8)]
    status, n_t1, n_t2, n = aggregate_precision_status(cases)
    assert status == "PASS_T1"
    assert n_t1 == 8 and n_t2 == 8 and n == 8


def test_aggregate_precision_t2_only() -> None:
    cases = [PrecisionCase(f"c{i}", 5e-4, 5e-4, False, True) for i in range(8)]
    status, n_t1, n_t2, n = aggregate_precision_status(cases)
    assert status == "PASS_T2"
    assert n_t1 == 0 and n_t2 == 8


def test_aggregate_precision_partial_fail() -> None:
    cases = [PrecisionCase("c0", 0.0, 0.0, True, True),
             PrecisionCase("c1", 1e-2, 1e-2, False, False)]
    status, _, _, _ = aggregate_precision_status(cases)
    assert status == "FAIL"


def test_aggregate_precision_empty() -> None:
    status, _, _, _ = aggregate_precision_status([])
    assert status == "SKIPPED"


def test_aggregate_perf_pass_at_threshold() -> None:
    status, med = aggregate_perf_status([0.6, 0.6, 0.7])
    assert status == "PASS"
    assert med >= 0.6


def test_aggregate_perf_fail_below_floor() -> None:
    status, med = aggregate_perf_status([0.3, 0.4, 0.55])
    assert status == "FAIL_FLOOR"
    assert med < 0.6


def test_aggregate_perf_empty() -> None:
    status, _ = aggregate_perf_status([])
    assert status == "SKIPPED"


# ---------------------------------------------------------------------------
# Top-level verify() with injected invoker
# ---------------------------------------------------------------------------
def test_verify_all_t1_pass(tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)

    def _inv(so, ds, perf, binding):
        return {
            "binding": binding,
            "cases": [
                {"case_id": "c0", "max_abs_err": 0.0, "max_rel_err": 0.0,
                 "a3_time_s": 0.5, "a5_time_s": 0.25},  # ratio 2.0
                {"case_id": "c1", "max_abs_err": 0.0, "max_rel_err": 0.0,
                 "a3_time_s": 0.6, "a5_time_s": 0.4},   # ratio 1.5
            ],
            "determinism_n_runs": 2, "determinism_n_identical": 2,
        }
    rep = verify("op_a", ws, invoker=_inv)
    assert rep.precision_status == "PASS_T1"
    assert rep.n_pass_t1 == 2 and rep.n_total == 2
    assert rep.perf_status == "PASS"
    assert rep.perf_ratio_median > 1.0
    assert rep.observed_deterministic is True
    assert rep.errors == []


def test_verify_t2_with_slow_perf(tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)

    def _inv(so, ds, perf, binding):
        return {
            "binding": binding,
            "cases": [
                {"case_id": "c0", "max_abs_err": 5e-4, "max_rel_err": 5e-4,
                 "a3_time_s": 1.0, "a5_time_s": 2.5},  # ratio 0.4 (below floor)
            ],
            "determinism_n_runs": 2,
            "determinism_n_identical": 2,
        }
    rep = verify("op_a", ws, invoker=_inv)
    assert rep.precision_status == "PASS_T2"
    assert rep.perf_status == "FAIL_FLOOR"


def test_verify_no_build(tmp_path: Path) -> None:
    workspace = tmp_path / "ws" / "op_a"
    workspace.mkdir(parents=True)
    # No .prior_art_build.json
    rep = verify("op_a", workspace, invoker=lambda *a: {})
    assert rep.precision_status == "SKIPPED"
    assert any("prior_art_build" in e for e in rep.errors)


def test_verify_build_failed_propagates(tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path, build_verdict="BUILD_FAILED")
    rep = verify("op_a", ws, invoker=lambda *a: {})
    assert rep.precision_status == "SKIPPED"
    assert any("BUILD_FAILED" in e for e in rep.errors)


def test_verify_missing_edge_dataset(tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    (ws / "edge_dataset.pt").unlink()
    rep = verify("op_a", ws, invoker=lambda *a: {})
    assert rep.precision_status == "SKIPPED"
    assert any("edge_dataset" in e for e in rep.errors)


def test_verify_determinism_signal_observed_false(tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)

    def _inv(so, ds, perf, binding):
        return {
            "binding": binding,
            "cases": [
                {"case_id": "c0", "max_abs_err": 0.0, "max_rel_err": 0.0,
                 "a3_time_s": 1.0, "a5_time_s": 1.0},
            ],
            "determinism_n_runs": 2, "determinism_n_identical": 1,
        }
    rep = verify("op_a", ws, invoker=_inv)
    assert rep.observed_deterministic is False


def test_write_verify_report_schema(tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)

    def _inv(so, ds, perf, binding):
        return {"binding": binding,
                "cases": [{"case_id": "c0", "max_abs_err": 0.0,
                            "max_rel_err": 0.0,
                            "a3_time_s": 1.0, "a5_time_s": 0.5}],
                "determinism_n_runs": 2,
                "determinism_n_identical": 2}
    rep = verify("op_a", ws, invoker=_inv)
    out = write_verify_report(rep, ws)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["op"] == "op_a"
    assert data["precision"]["status"] == "PASS_T1"
    assert "ratio_median" in data["performance"]
    assert data["schema_version"] == 2
    assert data["binding"]["candidate_digest"]


def test_verify_rejects_adapter_binding_mismatch(tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)

    def _inv(so, ds, perf, binding):
        wrong = dict(binding)
        wrong["candidate_sha256"] = "0" * 64
        return {
            "binding": wrong,
            "cases": [{
                "case_id": "c0", "max_abs_err": 0.0, "max_rel_err": 0.0,
                "a3_time_s": 1.0, "a5_time_s": 1.0,
            }],
            "determinism_n_runs": 2,
            "determinism_n_identical": 2,
        }

    rep = verify("op_a", ws, invoker=_inv)
    assert rep.precision_status == "SKIPPED"
    assert any("exact candidate/truth binding" in error for error in rep.errors)


def test_verify_rejects_so_changed_after_build(tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    so_path = ws / ".prior_art_candidate" / "build" / "op_a_kernels.so"
    so_path.write_text("different library")

    rep = verify("op_a", ws, invoker=lambda *args: pytest.fail("must not invoke"))

    assert rep.precision_status == "SKIPPED"
    assert any("digest does not match" in error for error in rep.errors)


def test_verify_requires_fresh_source_capture_metadata(tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    (ws / "a3_reference_runnable.json").write_text(json.dumps({
        "verdict": "READY", "a3_exec_attempted": False,
    }))

    rep = verify("op_a", ws, invoker=lambda *args: pytest.fail("must not invoke"))

    assert rep.precision_status == "SKIPPED"
    assert any("live execution" in error or "capture_id" in error
               for error in rep.errors)


def test_default_invoker_preserves_environment_and_passes_explicit_binding(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = tmp_path / "verify_prior_art_candidate.py"
    runner.write_text("# adapter stub")
    so_path = tmp_path / "candidate.so"
    edge_dataset = tmp_path / "edge_dataset.pt"
    source_perf = tmp_path / "a3_baseline_perf.json"
    for path in (so_path, edge_dataset, source_perf):
        path.write_text(path.name)
    binding = {
        "candidate_path": str(so_path),
        "candidate_sha256": "1" * 64,
        "candidate_digest": "2" * 64,
        "manifest_sha256": "3" * 64,
        "edge_dataset_path": str(edge_dataset),
        "edge_dataset_sha256": "4" * 64,
        "source_perf_path": str(source_perf),
        "source_perf_sha256": "5" * 64,
        "capture_id": "capture-id",
        "capture_metadata_sha256": "6" * 64,
    }
    captured = {}
    monkeypatch.setenv("VERIFY_KEEP_ME", "preserved")

    def _run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps({"binding": binding, "cases": []}) + "\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _run)
    result = _default_invoker(so_path, edge_dataset, source_perf, binding)

    assert result["binding"] == binding
    assert captured["env"]["VERIFY_KEEP_ME"] == "preserved"
    assert captured["env"]["PRIOR_ART_SO_SHA256"] == "1" * 64
    assert "--candidate-so" in captured["command"]
    assert "--edge-dataset-sha256" in captured["command"]
    assert captured["cwd"] == str(tmp_path)


def test_default_invoker_fails_closed_without_adapter(tmp_path: Path) -> None:
    so_path = tmp_path / "candidate.so"
    edge_dataset = tmp_path / "edge_dataset.pt"
    source_perf = tmp_path / "a3_baseline_perf.json"
    binding = {
        "candidate_sha256": "1" * 64,
        "candidate_digest": "2" * 64,
        "edge_dataset_sha256": "3" * 64,
        "source_perf_sha256": "4" * 64,
        "capture_id": "capture-id",
    }

    result = _default_invoker(so_path, edge_dataset, source_perf, binding)

    assert "adapter unavailable" in result["error"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
