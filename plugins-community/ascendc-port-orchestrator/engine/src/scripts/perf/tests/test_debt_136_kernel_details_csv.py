# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-136 (2026-05-26): `run_npu_profile.py` accepts torch_npu 2.8.0
`kernel_details_*.csv` in addition to torch_npu 2.7 `op_summary_*.csv`.

Caught by independent review 2026-05-26 23:46Z: A3 FA e2e run took ~32min (vs cv-agent's
autonomous ~8min) because the profiler CSV parser only globbed
`op_summary_*.csv`. torch_npu 2.8.0 (current on A3+A5 fleet) emits
`kernel_details_*.csv` with renamed columns. `finalize_pipeline._PROFILER_CSV_TOKENS`
was already extended to accept both per DEBT-128; this is the driver-side
symmetric fix.
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from run_npu_profile import (  # noqa: E402
    _find_op_summary_csv,
    _parse_op_durations,
    _resolve_col,
    _DURATION_COL_CANDIDATES,
    _STEPID_COL_CANDIDATES,
    _NAME_COL_CANDIDATES,
)


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def test_find_csv_accepts_op_summary(tmp_path: Path):
    """Legacy op_summary_*.csv still discovered."""
    sub = tmp_path / "2026-05-26_pid123" / "mindstudio_profiler_output"
    sub.mkdir(parents=True)
    f = sub / "op_summary_0.csv"
    f.write_text("Type,Name,Step Id,Task Duration(us)\n")
    found = _find_op_summary_csv(tmp_path)
    assert len(found) == 1 and found[0] == f


def test_find_csv_accepts_kernel_details(tmp_path: Path):
    """DEBT-136: kernel_details_*.csv (torch_npu 2.8+) now discovered."""
    sub = tmp_path / "2026-05-26_pid123" / "mindstudio_profiler_output"
    sub.mkdir(parents=True)
    f = sub / "kernel_details_0.csv"
    f.write_text("Kernel Name,Step Id,Duration(us)\n")
    found = _find_op_summary_csv(tmp_path)
    assert len(found) == 1 and found[0] == f


def test_find_csv_accepts_both_formats_in_same_run(tmp_path: Path):
    """If a session somehow emits both, both are returned (caller aggregates)."""
    sub = tmp_path / "2026-05-26_pid123" / "mindstudio_profiler_output"
    sub.mkdir(parents=True)
    (sub / "op_summary_0.csv").write_text("Type,Name,Step Id,Task Duration(us)\n")
    (sub / "kernel_details_0.csv").write_text("Kernel Name,Step Id,Duration(us)\n")
    found = _find_op_summary_csv(tmp_path)
    assert len(found) == 2
    names = sorted(p.name for p in found)
    assert names == ["kernel_details_0.csv", "op_summary_0.csv"]


def test_resolve_col_case_insensitive():
    assert _resolve_col(["Task Duration(us)"], _DURATION_COL_CANDIDATES) == "Task Duration(us)"
    assert _resolve_col(["Duration(us)"], _DURATION_COL_CANDIDATES) == "Duration(us)"
    # case-insensitive match (kernel_details may be "duration(us)")
    assert _resolve_col(["duration(us)"], _DURATION_COL_CANDIDATES) == "duration(us)"
    # missing entirely
    assert _resolve_col(["Foo", "Bar"], _DURATION_COL_CANDIDATES) is None


def test_parse_op_summary_legacy_format(tmp_path: Path):
    """Legacy op_summary_*.csv parsing still works (regression guard)."""
    csv_path = tmp_path / "op_summary_0.csv"
    _write_csv(
        csv_path,
        ["Type", "Name", "Step Id", "Task Duration(us)"],
        [
            ["MatMul", "matmul_0", "0", "100.0"],
            ["MatMul", "matmul_0", "0", "200.0"],
            ["MatMul", "matmul_0", "1", "150.0"],
        ],
    )
    out = _parse_op_durations(csv_path)
    # Step 0: 100+200=300us → 0.3ms, Step 1: 150us → 0.15ms
    assert out == pytest.approx([0.300, 0.150])


def test_parse_kernel_details_new_format(tmp_path: Path):
    """DEBT-136 core: kernel_details_*.csv parses correctly via column-aliasing."""
    csv_path = tmp_path / "kernel_details_0.csv"
    _write_csv(
        csv_path,
        ["Kernel Name", "Step Id", "Duration(us)"],
        [
            ["fa_softmax_kernel", "0", "250.0"],
            ["fa_matmul_kernel", "0", "750.0"],
            ["fa_softmax_kernel", "1", "260.0"],
        ],
    )
    out = _parse_op_durations(csv_path)
    # Step 0: 250+750=1000us → 1.0ms, Step 1: 260us → 0.26ms
    assert out == pytest.approx([1.000, 0.260])


def test_parse_kernel_details_filter_by_name(tmp_path: Path):
    """kernel_op match works against 'Kernel Name' field in new format."""
    csv_path = tmp_path / "kernel_details_0.csv"
    _write_csv(
        csv_path,
        ["Kernel Name", "Step Id", "Duration(us)"],
        [
            ["fa_softmax_kernel", "0", "250.0"],
            ["fa_matmul_kernel", "0", "750.0"],
            ["fa_softmax_kernel", "1", "260.0"],
        ],
    )
    out = _parse_op_durations(csv_path, kernel_op="softmax", kernel_op_match="substring")
    # Only softmax: step 0 = 250us → 0.25ms, step 1 = 260us → 0.26ms
    assert out == pytest.approx([0.250, 0.260])


def test_parse_kernel_details_aiv_duration_variant(tmp_path: Path):
    """Some kernel_details_*.csv variants emit 'Aiv Duration(us)' (Vector
    core path) — column-alias must accept it.
    """
    csv_path = tmp_path / "kernel_details_aiv.csv"
    _write_csv(
        csv_path,
        ["Kernel Name", "Step Id", "Aiv Duration(us)"],
        [
            ["vec_kernel", "0", "500.0"],
            ["vec_kernel", "1", "600.0"],
        ],
    )
    out = _parse_op_durations(csv_path)
    assert out == pytest.approx([0.500, 0.600])


def test_parse_unrecognized_csv_returns_empty(tmp_path: Path):
    """If neither legacy nor new-format duration column found, return empty
    list — caller falls back to inline timer path. Must NOT crash.
    """
    csv_path = tmp_path / "weird.csv"
    _write_csv(csv_path, ["Foo", "Bar", "Baz"], [["a", "b", "c"]])
    out = _parse_op_durations(csv_path)
    assert out == []
