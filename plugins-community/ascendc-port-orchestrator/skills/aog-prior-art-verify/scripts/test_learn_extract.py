# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for learn_extract — Phase 6 of aog-prior-art-verify."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from learn_extract import (  # noqa: E402
    extract, append_candidates,
    _scan_file_for_signals,
)


def _setup(tmp_path: Path, op: str = "op_a",
           verdict: str = "CANDIDATE_PASS",
           kernel_contents: dict = None) -> Path:
    workspace = tmp_path / "ws" / op
    candidate = workspace / ".prior_art_candidate"
    arch35 = candidate / "op_kernel" / "arch35"
    arch35.mkdir(parents=True)
    contents = kernel_contents or {"op_a.h": "// stub"}
    for name, content in contents.items():
        (arch35 / name).write_text(content)
    (workspace / "prior_art_verdict.json").write_text(json.dumps({
        "op": op, "verdict": verdict,
    }))
    return workspace


def test_scan_detects_microapi_signals(tmp_path: Path) -> None:
    f = tmp_path / "kernel.h"
    f.write_text("""
    RegTensor<float> tensor;
    __VEC_SCOPE__ {
        Add(...);
    }
    """)
    sigs = _scan_file_for_signals(f)
    assert sigs.get("MicroAPI_RegTensor") == 1
    assert sigs.get("MicroAPI_VEC_SCOPE") == 1


def test_scan_detects_simt_signals(tmp_path: Path) -> None:
    f = tmp_path / "kernel.h"
    f.write_text("""
    __simt_vf__ void launch() {}
    LAUNCH_BOUND(256);
    __local_mem__ int shared[64];
    Simt::AtomicAdd(...);
    """)
    sigs = _scan_file_for_signals(f)
    assert sigs.get("SIMT_simt_vf") == 1
    assert sigs.get("SIMT_LAUNCH_BOUND") == 1
    assert sigs.get("SIMT_local_mem") == 1
    assert sigs.get("SIMT_AtomicAdd") == 1


def test_scan_detects_v220_guards(tmp_path: Path) -> None:
    """V220 guards in arch35 = anti-pattern."""
    f = tmp_path / "kernel.h"
    f.write_text("""
    #if __CCE_AICORE__ == 220
        // V220-only path
    #endif
    #if __NPU_ARCH__ == 3003
        // BF16-disabled path
    #endif
    """)
    sigs = _scan_file_for_signals(f)
    assert sigs.get("V220_guards_present", 0) >= 2


def test_scan_detects_multi_variant_split(tmp_path: Path) -> None:
    """LayerNorm-style welford+full_load split is a positive pattern signal."""
    f = tmp_path / "op.cpp"
    f.write_text("""
    #include "kernels/arch35/ada_layer_norm_welford.h"
    #include "kernels/arch35/ada_layer_norm_full_load.h"
    """)
    sigs = _scan_file_for_signals(f)
    assert sigs.get("Multi_variant_split", 0) >= 2


def test_extract_pass_tags_signals_positive(tmp_path: Path) -> None:
    workspace = _setup(tmp_path, verdict="CANDIDATE_PASS", kernel_contents={
        "op_a.h": "RegTensor<float> x;\n__VEC_SCOPE__ {}\n",
    })
    rep = extract("op_a", workspace)
    assert rep.verdict == "CANDIDATE_PASS"
    assert rep.candidates_added >= 2
    for d in rep.deltas_extracted:
        if d["signal"] != "V220_guards_present":
            assert d["source"] == "upstream_pass"
            assert d["counter_example"] is False


def test_extract_fail_tags_signals_negative(tmp_path: Path) -> None:
    workspace = _setup(tmp_path, verdict="CANDIDATE_PRECISION_GAP",
                        kernel_contents={
                            "op_a.h": "RegTensor<float> x;\n",
                        })
    rep = extract("op_a", workspace)
    assert rep.verdict == "CANDIDATE_PRECISION_GAP"
    for d in rep.deltas_extracted:
        if d["signal"] != "V220_guards_present":
            assert d["source"] == "upstream_fail"
            assert d["counter_example"] is True


def test_extract_v220_guards_always_anti(tmp_path: Path) -> None:
    """Even on candidate-pass verdict, V220 guards in arch35 are flagged as
    counter_example=yes (they shouldn't be there)."""
    workspace = _setup(tmp_path, verdict="CANDIDATE_PASS", kernel_contents={
        "op_a.h": "#if __CCE_AICORE__ == 220\n#endif\n",
    })
    rep = extract("op_a", workspace)
    guard_deltas = [d for d in rep.deltas_extracted
                     if d["signal"] == "V220_guards_present"]
    assert len(guard_deltas) == 1
    assert guard_deltas[0]["counter_example"] is True
    assert guard_deltas[0]["source"] == "upstream_fail"


def test_extract_writes_learn_md(tmp_path: Path) -> None:
    workspace = _setup(tmp_path, verdict="CANDIDATE_PASS", kernel_contents={
        "op_a.h": "RegTensor<float> x;\n",
    })
    rep = extract("op_a", workspace)
    assert rep.learn_md_path is not None and rep.learn_md_path.exists()
    body = rep.learn_md_path.read_text()
    assert "CANDIDATE_PASS" in body
    assert "MicroAPI_RegTensor" in body


def test_extract_no_candidate_dir(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    rep = extract("op_a", workspace)
    assert any("candidate dir missing" in e for e in rep.errors)


def test_extract_handles_missing_verdict(tmp_path: Path) -> None:
    """If prior_art_verdict.json missing, run with verdict=UNKNOWN; still
    extracts signals."""
    workspace = tmp_path / "ws" / "op_a"
    candidate = workspace / ".prior_art_candidate" / "op_kernel" / "arch35"
    candidate.mkdir(parents=True)
    (candidate / "op_a.h").write_text("RegTensor<float> x;\n")
    rep = extract("op_a", workspace)
    assert rep.verdict == "UNKNOWN"
    assert rep.candidates_added >= 1


def test_append_candidates_writes_block(tmp_path: Path) -> None:
    """Append a CAND-PRIOR-ART block; isolated to tmp_path (don't touch real KB)."""
    workspace = _setup(tmp_path, verdict="CANDIDATE_PASS", kernel_contents={
        "op_a.h": "RegTensor<float> x;\n__VEC_SCOPE__ {}\n",
    })
    rep = extract("op_a", workspace)
    candidates_file = tmp_path / "candidates.md"
    candidates_file.write_text("# Existing candidates\n")
    ok = append_candidates(rep, candidates_path=candidates_file)
    assert ok is True
    body = candidates_file.read_text()
    assert "### CAND-PRIOR-ART-op_a" in body
    assert "MicroAPI_RegTensor" in body


def test_append_candidates_noop_when_no_signals(tmp_path: Path) -> None:
    workspace = _setup(tmp_path, verdict="CANDIDATE_PASS", kernel_contents={
        "op_a.h": "// nothing interesting\n",
    })
    rep = extract("op_a", workspace)
    if rep.candidates_added == 0:
        candidates_file = tmp_path / "candidates.md"
        candidates_file.write_text("# orig\n")
        ok = append_candidates(rep, candidates_path=candidates_file)
        assert ok is False
        assert candidates_file.read_text() == "# orig\n"


def test_append_candidates_missing_parent_silent(tmp_path: Path) -> None:
    """If candidates_path.parent doesn't exist, return False without raising."""
    workspace = _setup(tmp_path, verdict="CANDIDATE_PASS", kernel_contents={
        "op_a.h": "RegTensor<float> x;\n",
    })
    rep = extract("op_a", workspace)
    bogus = tmp_path / "nonexistent" / "candidates.md"
    ok = append_candidates(rep, candidates_path=bogus)
    assert ok is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
