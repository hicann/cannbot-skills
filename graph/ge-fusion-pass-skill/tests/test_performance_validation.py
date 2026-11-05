#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise offline performance validation through its public CLI."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_evidence.py"


def write_manifest(root, label, source, input_path):
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    context = {
        "source_model_sha256": source_hash,
        "input_sha256": input_hash,
        "seed": 0,
        "preprocess": {},
        "soc_version": "Host-fixture",
        "compile_parameters": {},
        "run_parameters": {},
        "environment": {"scope": "synthetic Host artifacts; no NPU execution"},
        "warmup": 0,
        "runs": 3,
    }
    manifest = {
        "context": context,
        "execution": {
            "source_model": str(source),
            "source_model_sha256": source_hash,
            "input_sha256": input_hash,
            "input_hash_scheme": "single-file-sha256-v1",
            "input_files": [
                {
                    "index": 0,
                    "path": str(input_path),
                    "sha256": input_hash,
                    "size": input_path.stat().st_size,
                }
            ],
        },
        "latencies_ms": [1.0, 2.0, 3.0],
    }
    path = root / f"{label}.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.fixture
def performance_case(tmp_path):
    source = tmp_path / "source-model.fixture"
    source.write_bytes(b"synthetic source artifact for Host provenance checks")
    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"synthetic input")
    manifests = {
        label: write_manifest(tmp_path, label, source, input_path)
        for label in ("baseline", "optimized")
    }
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "op_statistic.csv").write_text(
        "OP Type,Count,Total Time(us)\nFixtureOp,3,6000\n", encoding="utf-8"
    )
    groups = tmp_path / "groups.json"
    groups.write_text(
        json.dumps({"baseline": ["FixtureOp"], "optimized": ["FixtureOp"]}),
        encoding="utf-8",
    )
    return manifests, profile, groups, tmp_path / "comparison.json"


def run_performance(manifests, profile, groups, output):
    command = [
        sys.executable,
        str(SCRIPT),
        "performance",
        "--baseline",
        str(manifests["baseline"]),
        "--optimized",
        str(manifests["optimized"]),
        "--baseline-profile",
        str(profile),
        "--optimized-profile",
        str(profile),
        "--operator-groups",
        str(groups),
        "--out-json",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(output.read_text(encoding="utf-8"))


def reject_nonfinite_json(value):
    raise ValueError(f"nonfinite JSON result: {value}")


@pytest.mark.parametrize("side", ["baseline", "optimized"])
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("NaN", "FAILED"),
        ("Infinity", "FAILED"),
        ("-Infinity", "FAILED"),
        (-1.0, "FAILED"),
        (0.0, "PASSED"),
        ("2.5", "PASSED"),
    ],
)
def test_performance_latency_validation(performance_case, side, value, expected):
    manifests, profile, groups, output = performance_case
    path = manifests[side]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["latencies_ms"][1] = value
    path.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")

    result = run_performance(manifests, profile, groups, output)
    assert result["provenance"]["status"] == "PASSED"
    assert result["operator_group_comparison"]["status"] == "PASSED"
    assert result["status"] == expected
    json.loads(output.read_text(encoding="utf-8"), parse_constant=reject_nonfinite_json)
    if isinstance(value, str) and value in {"NaN", "Infinity", "-Infinity"}:
        assert result["baseline"] is None
        assert result["optimized"] is None
        assert result["delta"] is None
        assert any("finite" in issue for issue in result["issues"])

    report = output.with_name("report.json")
    command = [
        sys.executable,
        str(SCRIPT),
        "report",
        str(output),
        "--out-json",
        str(report),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == expected


@pytest.mark.parametrize(
    ("selected_pass", "expected"),
    [("TargetPass", "PASSED"), ("OtherPass", "FAILED")],
)
def test_event_pass_selection(tmp_path, selected_pass, expected):
    events = tmp_path / "events.jsonl"
    with events.open("w", encoding="utf-8") as stream:
        for kind in ("candidate", "matched", "guard_passed", "applied"):
            stream.write(json.dumps({"event": kind, "pass": "TargetPass"}) + "\n")
    output = tmp_path / "events-result.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "events",
        "--events",
        str(events),
        "--pass-name",
        selected_pass,
        "--require-applied",
        "--out-json",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == expected
