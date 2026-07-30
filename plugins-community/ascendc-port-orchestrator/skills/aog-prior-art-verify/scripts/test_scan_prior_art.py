# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for scan_prior_art.scan() — P0dd default-OFF + Bug A Hybrid B+A.

Test environment: OPGEN_PRESTAGE_ARCH35 is unset (default-OFF) for default-OFF
tests; tests that exercise the opt-in path set OPGEN_PRESTAGE_ARCH35=1
explicitly via monkeypatch.

Coverage matrix:
  - default-OFF + plain only       → upstream_v220_entry (Bug A baseline)
  - default-OFF + plain + apt      → upstream_v220_entry preferred, apt suppressed
                                     (Hybrid B)
  - default-OFF + apt only         → apt suppressed; informational marker only
                                     (Hybrid A)
  - default-OFF + neither          → no prior art (truly fresh)
  - opt-in (=1) + arch35           → upstream_arch35 + shared_common detectors run
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from scan_prior_art import _assess_source_arch_complete, scan  # noqa: E402


def _touch(p: Path, content: str = "// stub") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


# --- opt-in (legacy P0dd OPGEN_PRESTAGE_ARCH35=1) tests ---


def test_optin_mode_a_single_op_arch35(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """OL-141 Mode A under OPT-IN — op's own op_kernel/arch35/ populated."""
    monkeypatch.setenv("OPGEN_PRESTAGE_ARCH35", "1")
    port_source = tmp_path / "cann" / "ops-nn" / "norm" / "ada_layer_norm"
    workspace = tmp_path / "ws" / "ada_layer_norm"
    _touch(port_source / "op_kernel" / "arch35" / "ada_layer_norm.h")
    _touch(port_source / "op_kernel" / "arch35" / "ada_layer_norm.cpp")
    r = scan("ada_layer_norm", port_source, workspace)
    assert r["has_prior_art"] is True
    types = [s["type"] for s in r["sources"]]
    assert "upstream_arch35" in types
    assert r["highest_trust"] == "HIGH"
    assert r["consulted_a5_sources"] is True


def test_optin_mode_b_shared_common(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """OL-141 Mode B under OPT-IN — arch35 lives in sibling <family>_common dir."""
    monkeypatch.setenv("OPGEN_PRESTAGE_ARCH35", "1")
    base = tmp_path / "cann" / "ops-nn" / "pooling"
    port_source = base / "adaptive_avg_pool3d"
    workspace = tmp_path / "ws" / "adaptive_avg_pool3d"
    port_source.mkdir(parents=True)
    common_arch35 = base / "adaptive_pool3d_common" / "op_kernel" / "arch35"
    _touch(common_arch35 / "adaptive_avg_pool3d_big_kernel.h")
    _touch(common_arch35 / "adaptive_avg_pool3d_parall_pool.h")
    _touch(common_arch35 / "adaptive_avg_pool3d_simt.h")
    _touch(common_arch35 / "adaptive_max_pool3d_big_kernel.h")
    r = scan("adaptive_avg_pool3d", port_source, workspace)
    assert r["has_prior_art"] is True
    types = [s["type"] for s in r["sources"]]
    assert "upstream_shared_common" in types
    sc = next(s for s in r["sources"] if s["type"] == "upstream_shared_common")
    assert sc["file_count"] == 3
    assert "adaptive_max_pool3d_big_kernel.h" not in sc["files"]
    assert r["highest_trust"] == "HIGH"


def test_optin_mode_a_and_mode_b_coexist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If an op has BOTH own arch35/ AND shared-common matches, both detectors fire (opt-in)."""
    monkeypatch.setenv("OPGEN_PRESTAGE_ARCH35", "1")
    base = tmp_path / "cann" / "ops-nn" / "pooling"
    port_source = base / "hybrid_op"
    workspace = tmp_path / "ws" / "hybrid_op"
    _touch(port_source / "op_kernel" / "arch35" / "hybrid_op.h")
    common_arch35 = base / "pool_common" / "op_kernel" / "arch35"
    _touch(common_arch35 / "hybrid_op_helper.h")
    r = scan("hybrid_op", port_source, workspace)
    types = [s["type"] for s in r["sources"]]
    assert "upstream_arch35" in types
    assert "upstream_shared_common" in types


def test_default_off_skips_arch35_detectors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """P0dd: default-OFF mode skips arch35 + shared_common detectors entirely."""
    monkeypatch.delenv("OPGEN_PRESTAGE_ARCH35", raising=False)
    port_source = tmp_path / "cann" / "ops-nn" / "norm" / "some_op"
    workspace = tmp_path / "ws" / "some_op"
    _touch(port_source / "op_kernel" / "arch35" / "some_op.h")
    r = scan("some_op", port_source, workspace)
    types = [s["type"] for s in r["sources"]]
    assert "upstream_arch35" not in types
    assert "upstream_shared_common" not in types
    assert r["consulted_a5_sources"] is False


# --- Bug A (2026-05-23) Hybrid B + Hybrid A default-OFF tests ---


def test_bug_a_v220_only_default_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Bug A baseline: upstream has only plain <op>.cpp (V220-pure entry).
    Default-OFF detects it via _detect_upstream_v220_entry. fused_quant_mat_mul-class."""
    monkeypatch.delenv("OPGEN_PRESTAGE_ARCH35", raising=False)
    port_source = tmp_path / "cann" / "ops-nn" / "matmul" / "fused_quant_mat_mul"
    workspace = tmp_path / "ws" / "fused_quant_mat_mul"
    _touch(port_source / "op_kernel" / "fused_quant_mat_mul.cpp")
    r = scan("fused_quant_mat_mul", port_source, workspace)
    assert r["has_prior_art"] is True
    types = [s["type"] for s in r["sources"]]
    assert "upstream_v220_entry" in types
    assert "upstream_apt" not in types  # no apt to suppress
    assert "upstream_apt_suppressed_apt_only_default_off" not in types
    assert r["highest_trust"] == "HIGH"


def test_bug_a_hybrid_b_plain_plus_apt_default_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Bug A Hybrid B: upstream has BOTH plain <op>.cpp AND <op>_apt.cpp.
    Default-OFF must PREFER plain, suppress apt. swi_glu/flat_quant/group_norm_silu-class."""
    monkeypatch.delenv("OPGEN_PRESTAGE_ARCH35", raising=False)
    port_source = tmp_path / "cann" / "ops-nn" / "activation" / "swi_glu"
    workspace = tmp_path / "ws" / "swi_glu"
    _touch(port_source / "op_kernel" / "swi_glu.cpp", '#include "swi_glu_impl.hpp"\n')
    _touch(port_source / "op_kernel" / "swi_glu_apt.cpp", '#include "arch35/swi_glu_impl.hpp"\n')
    r = scan("swi_glu", port_source, workspace)
    assert r["has_prior_art"] is True
    types = [s["type"] for s in r["sources"]]
    assert "upstream_v220_entry" in types
    assert "upstream_apt" not in types  # suppressed under Hybrid B
    assert "upstream_apt_suppressed_apt_only_default_off" not in types  # only fires when v220 absent
    # Picked entry is plain
    entry = next(s for s in r["sources"] if s["type"] == "upstream_v220_entry")
    assert entry["path"].endswith("swi_glu.cpp")
    assert "_apt.cpp" not in entry["path"]


def test_bug_a_hybrid_a_apt_only_default_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Bug A Hybrid A: upstream has ONLY <op>_apt.cpp (no plain <op>.cpp).
    Default-OFF suppresses apt (it has arch35/ includes), records informational
    marker so phase_o25_a3_ref knows to skip prestage + brief tells worker to
    hand-author dispatcher. erfinv/fast_gelu/gelu/elu-class."""
    monkeypatch.delenv("OPGEN_PRESTAGE_ARCH35", raising=False)
    port_source = tmp_path / "cann" / "ops-nn" / "activation" / "erfinv"
    workspace = tmp_path / "ws" / "erfinv"
    _touch(port_source / "op_kernel" / "erfinv_apt.cpp", '#include "arch35/erfinv_impl.hpp"\n')
    r = scan("erfinv", port_source, workspace)
    types = [s["type"] for s in r["sources"]]
    assert "upstream_v220_entry" not in types
    assert "upstream_apt" not in types  # suppressed
    assert "upstream_apt_suppressed_apt_only_default_off" in types  # informational marker
    marker = next(s for s in r["sources"] if s["type"] == "upstream_apt_suppressed_apt_only_default_off")
    assert "arch35" in marker["reason"]
    # has_prior_art should still be True since the marker counts as a source
    # (but highest_trust ranking honors the LOW marker trust)


def test_bug_a_optin_keeps_apt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Bug A opt-in: OPGEN_PRESTAGE_ARCH35=1 disables Hybrid suppression — apt is
    used as before (legacy path). Both plain + apt visible if both exist."""
    monkeypatch.setenv("OPGEN_PRESTAGE_ARCH35", "1")
    port_source = tmp_path / "cann" / "ops-nn" / "activation" / "swi_glu"
    workspace = tmp_path / "ws" / "swi_glu"
    _touch(port_source / "op_kernel" / "swi_glu.cpp")
    _touch(port_source / "op_kernel" / "swi_glu_apt.cpp")
    r = scan("swi_glu", port_source, workspace)
    types = [s["type"] for s in r["sources"]]
    # Under opt-in, BOTH detectors run; no suppression. v220_entry still detected.
    assert "upstream_v220_entry" in types
    assert "upstream_apt" in types
    assert "upstream_apt_suppressed_apt_only_default_off" not in types


def test_bug_a_truly_fresh_op(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Truly fresh op — no plain <op>.cpp, no <op>_apt.cpp, no arch35/. Default-OFF.
    Must return has_prior_art=False (zero sources)."""
    monkeypatch.delenv("OPGEN_PRESTAGE_ARCH35", raising=False)
    port_source = tmp_path / "cann" / "ops-nn" / "novel" / "novel_op"
    workspace = tmp_path / "ws" / "novel_op"
    port_source.mkdir(parents=True)
    (port_source / "op_kernel").mkdir()
    r = scan("novel_op", port_source, workspace)
    assert r["has_prior_art"] is False
    assert r["sources"] == []
    assert r["highest_trust"] is None


def test_mode_b_no_sibling_common(tmp_path: Path) -> None:
    """Negative: sibling *_common dirs without op-prefixed files should NOT match."""
    base = tmp_path / "cann" / "ops-nn" / "pooling"
    port_source = base / "some_op"
    workspace = tmp_path / "ws" / "some_op"
    port_source.mkdir(parents=True)
    # Sibling dir has the right shape but only foreign-op files
    common_arch35 = base / "foo_common" / "op_kernel" / "arch35"
    _touch(common_arch35 / "other_op_kernel.h")
    r = scan("some_op", port_source, workspace)
    assert r["has_prior_art"] is False, "must not match unrelated _common files"


# --- DEBT-165: source-arch completeness (architecture-based port-entry gate) ---
# Mirrors the 4 real cases that anchored the discriminator (2026-06-21):
#   deformable_offsets (pure arch35 shell)  → INCOMPLETE
#   deformable_conv2d  (plain <op>.cpp)      → complete
#   gelu               (apt + atvoss/ algo)  → complete (apt-only != cheat)
#   grid_sample        (no arch35 dir)        → complete

def test_debt165_pure_target_arch_shell_is_incomplete(tmp_path: Path) -> None:
    """deformable_offsets shape: op_kernel/ has ONLY <op>_apt.cpp that #includes ONLY
    arch35/, plus arch35/<op>.h. No source-arch algorithm → INCOMPLETE (reject)."""
    op = "deformable_offsets"
    src = tmp_path / "cann" / "ops-nn" / "conv" / op
    _touch(src / "op_kernel" / f"{op}_apt.cpp", '#include "arch35/deformable_offsets.h"\n')
    _touch(src / "op_kernel" / "arch35" / f"{op}.h", "// real impl only here\n")
    complete, reason = _assess_source_arch_complete(src, op)
    assert complete is False, f"pure arch35 shell must be incomplete; reason={reason}"
    assert "shell" in reason.lower()
    assert scan(op, src, tmp_path / "ws")["source_arch_complete"] is False


def test_debt165_v220_entry_is_complete(tmp_path: Path) -> None:
    """deformable_conv2d shape: plain <op>.cpp present → complete (real source entry)."""
    op = "deformable_conv2d"
    src = tmp_path / "cann" / "ops-nn" / "conv" / op
    _touch(src / "op_kernel" / f"{op}.cpp", "// real V220 algorithm\nint main(){}\n")
    complete, reason = _assess_source_arch_complete(src, op)
    assert complete is True, reason
    assert f"{op}.cpp" in reason


def test_debt165_apt_only_with_nontarget_algorithm_is_complete(tmp_path: Path) -> None:
    """gelu shape — THE key edge case: apt-only is NOT a cheat. <op>_apt.cpp #includes a
    target-arch header AND a non-target-arch algorithm lib (atvoss/) → complete."""
    op = "gelu"
    src = tmp_path / "cann" / "ops-nn" / "activation" / op
    _touch(src / "op_kernel" / f"{op}_apt.cpp",
           '#include "kernel_operator.h"\n'
           '#include "arch35/gelu_dag.h"\n'
           '#include "atvoss/elewise/elewise_sch_with_scalar.h"\n')
    _touch(src / "op_kernel" / "arch35" / "gelu_dag.h", "// arch35 config\n")
    complete, reason = _assess_source_arch_complete(src, op)
    assert complete is True, f"apt-only with non-target algorithm must be complete; {reason}"
    assert "atvoss" in reason


def test_debt165_no_target_arch_dir_is_complete(tmp_path: Path) -> None:
    """grid_sample/canndev shape: a kernel .cpp, no arch35/ dir → source-arch-pure → complete."""
    op = "grid_sample"
    src = tmp_path / "cann" / "canndev" / "ops" / "image" / op
    _touch(src / "op_kernel" / f"{op}_kernel.cpp", "// V220-pure kernel\n")
    complete, reason = _assess_source_arch_complete(src, op)
    assert complete is True, reason


def test_debt165_framework_only_apt_is_incomplete(tmp_path: Path) -> None:
    """Defensive: an apt that includes ONLY framework headers + arch35 (no real algorithm
    source) is still a shell → incomplete (framework headers don't count as algorithm)."""
    op = "shelly"
    src = tmp_path / "cann" / "ops-nn" / "misc" / op
    _touch(src / "op_kernel" / f"{op}_apt.cpp",
           '#include "kernel_operator.h"\n#include "kernel_tiling/kernel_tiling.h"\n'
           '#include "arch35/shelly.h"\n')
    _touch(src / "op_kernel" / "arch35" / f"{op}.h", "// only impl\n")
    complete, _ = _assess_source_arch_complete(src, op)
    assert complete is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
