# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""flash_attention_score-pbh-1 (2026-06-11, owner mandate) regression test —
GE op_host raw-CANN-copy gate.

Background (the reckoning this prevents): 13 port_a3 FA archives shipped a GE
op_host (def/infershape/tiling.cpp) byte-for-byte identical to CANN source
under `~/workspace/cann/ops-transformer/attention/flash_attention_score/op_host/`.
Customers without CANN source can't reproduce those archives. The KB now
carries the generative path (`fa_class/templates/op_host/` skeletons +
`wp_fa_host_tiling.h` arch35 tiling + `GE_HOST_TRANSFORM_RECIPE.md`). The
finalize gate `_check_ge_ophost_raw_cann_copy` makes the recipe-driven path
structurally binding via TWO checks:

  (A) byte check  — md5 of any op_host/*.cpp|*.h == a same-basename CANN
      source file (only where CANN source present).
  (B) structural  — GE tiling.cpp lacks `#include "wp_fa_host_tiling.h"` OR
      has zero `wfh::`/`wp_fa_host::` calls → raw arch35 copy. LOAD-BEARING,
      works WITHOUT CANN source.

These fixtures exercise both directions: a raw-copy tiling.cpp REJECTS, a
recipe-assembled (wfh::) tiling.cpp PASSES.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import finalize_pipeline as fp


# A recipe-assembled GE tiling.cpp: includes the KB shared header + calls wfh::
_GOOD_TILING = (
    '#include <cstdint>\n'
    '#include <register/op_impl_registry.h>\n'
    '#include "wp_fa_host_tiling.h"\n'
    'namespace wfh = wp_fa_host;\n'
    'static ge::graphStatus DoTiling(gert::TilingContext* ctx) {\n'
    '    const int64_t dBasicBlock = wfh::CalcDBasicBlock(D);\n'
    '    const int64_t effSparseMode = wfh::CalcEffSparseMode(sparseMode, hasAtten);\n'
    '    wfh::MultiCoreParams mc = wfh::SetMultiCoreParamsRegbase(B, N2, gSize, S1, s1bb, aic);\n'
    '    return ge::GRAPH_SUCCESS;\n'
    '}\n'
)

# A raw arch35 copy: tiling values inlined, no shared-layer include/call.
_RAW_TILING = (
    '#include <cstdint>\n'
    '#include <register/op_impl_registry.h>\n'
    'static ge::graphStatus DoTiling(gert::TilingContext* ctx) {\n'
    '    int64_t dBasicBlock = (D + 63) / 64 * 64;  // inlined arch35 arithmetic\n'
    '    int64_t s1BasicBlock = 128;\n'
    '    return ge::GRAPH_SUCCESS;\n'
    '}\n'
)

# Fully-qualified namespace form also satisfies the structural check.
_GOOD_TILING_FQ = _GOOD_TILING.replace("namespace wfh = wp_fa_host;\n", "").replace(
    "wfh::", "wp_fa_host::"
)


def _make_port_a3_fa_ws(tmp_path: Path, tiling_text: str, *,
                        op: str = "flash_attention_score",
                        mode: str = "port_a3_to_a5",
                        extra_op_host: dict | None = None) -> Path:
    """Synthetic workspace detect_plugin() resolves to port_a3, op FA-named."""
    ws = tmp_path / op
    (ws / "op_host").mkdir(parents=True, exist_ok=True)
    (ws / ".opgen_state.json").write_text(
        json.dumps({"op": op, "opgen_mode": mode})
    )
    (ws / "op_host" / f"{op}_tiling.cpp").write_text(tiling_text)
    for name, text in (extra_op_host or {}).items():
        (ws / "op_host" / name).write_text(text)
    return ws


# ---------------------------------------------------------------------------
# (B) STRUCTURAL CHECK — the load-bearing, customer-side guard
# ---------------------------------------------------------------------------
def test_structural_raw_tiling_rejected(tmp_path, monkeypatch):
    """tiling.cpp with inlined values (no wfh:: / no shared include) → REJECT."""
    # Point CANN root at a non-existent dir so only (B) can fire.
    monkeypatch.setenv("CANN_SOURCE_ROOT", str(tmp_path / "no_cann"))
    ws = _make_port_a3_fa_ws(tmp_path, _RAW_TILING)
    rationale = getattr(fp, '_check_ge_ophost_raw_cann_copy')(ws)
    assert rationale is not None, "raw-copy tiling.cpp (no shared layer) MUST reject"
    assert "flash_attention_score_tiling.cpp" in rationale
    assert "wp_fa_host_tiling.h" in rationale  # actionable: points at the shared layer
    assert "GE_HOST_TRANSFORM_RECIPE.md" in rationale  # actionable: points at recipe


def test_structural_recipe_tiling_passes(tmp_path, monkeypatch):
    """tiling.cpp that includes wp_fa_host_tiling.h + calls wfh:: → PASS."""
    monkeypatch.setenv("CANN_SOURCE_ROOT", str(tmp_path / "no_cann"))
    ws = _make_port_a3_fa_ws(tmp_path, _GOOD_TILING)
    assert getattr(fp, '_check_ge_ophost_raw_cann_copy')(ws) is None, (
        "recipe-assembled tiling.cpp (wfh:: shared layer) MUST pass"
    )


def test_structural_fully_qualified_namespace_passes(tmp_path, monkeypatch):
    """wp_fa_host:: (fully-qualified, no `wfh` alias) also satisfies (B)."""
    monkeypatch.setenv("CANN_SOURCE_ROOT", str(tmp_path / "no_cann"))
    ws = _make_port_a3_fa_ws(tmp_path, _GOOD_TILING_FQ)
    assert getattr(fp, '_check_ge_ophost_raw_cann_copy')(ws) is None


def test_structural_include_without_call_rejected(tmp_path, monkeypatch):
    """Includes the header but never CALLS wfh:: → still a copy (REJECT)."""
    monkeypatch.setenv("CANN_SOURCE_ROOT", str(tmp_path / "no_cann"))
    tiling = '#include "wp_fa_host_tiling.h"\n' + _RAW_TILING  # include only, no call
    ws = _make_port_a3_fa_ws(tmp_path, tiling)
    rationale = getattr(fp, '_check_ge_ophost_raw_cann_copy')(ws)
    assert rationale is not None
    assert "wfh::/wp_fa_host:: call" in rationale


# ---------------------------------------------------------------------------
# (A) BYTE CHECK — belt+suspenders, only where CANN source present
# ---------------------------------------------------------------------------
def test_byte_identical_to_cann_rejected(tmp_path, monkeypatch):
    """op_host file byte-identical (md5) to a CANN-source same-basename file
    → REJECT, even when (B) would pass (uses the wfh:: shared layer).
    """
    # Build a fake CANN source tree with the attention subtree.
    cann = tmp_path / "cann"
    fa_dir = cann / "ops-transformer" / "attention" / "flash_attention_score" / "op_host"
    fa_dir.mkdir(parents=True)
    # The CANN def.cpp the archive will byte-copy.
    cann_def = "// CANN arch35 def.cpp\nstatic const char kSoc[] = \"ascend910_95\";\n"
    (fa_dir / "flash_attention_score_def.cpp").write_text(cann_def)
    monkeypatch.setenv("CANN_SOURCE_ROOT", str(cann))

    # Archive: tiling.cpp uses wfh:: (passes B), but def.cpp is a byte-copy.
    ws = _make_port_a3_fa_ws(
        tmp_path, _GOOD_TILING,
        extra_op_host={"flash_attention_score_def.cpp": cann_def},
    )
    rationale = getattr(fp, '_check_ge_ophost_raw_cann_copy')(ws)
    assert rationale is not None, "byte-identical def.cpp MUST reject"
    assert "flash_attention_score_def.cpp" in rationale
    assert "md5" in rationale.lower()
    # Sanity: the md5 reported is the real one.
    assert hashlib.md5(cann_def.encode()).hexdigest() in rationale


def test_byte_different_from_cann_passes(tmp_path, monkeypatch):
    """op_host def.cpp present in CANN by basename but content DIFFERS (carried
    + patched per recipe) → PASS (only when (B) also passes).
    """
    cann = tmp_path / "cann"
    fa_dir = cann / "ops-transformer" / "attention" / "flash_attention_score" / "op_host"
    fa_dir.mkdir(parents=True)
    (fa_dir / "flash_attention_score_def.cpp").write_text("// CANN original def\n")
    monkeypatch.setenv("CANN_SOURCE_ROOT", str(cann))

    ws = _make_port_a3_fa_ws(
        tmp_path, _GOOD_TILING,
        extra_op_host={
            "flash_attention_score_def.cpp": "// CARRIED + PATCHED def (differs)\n",
        },
    )
    assert getattr(fp, '_check_ge_ophost_raw_cann_copy')(ws) is None, (
        "byte-different def.cpp with recipe-assembled tiling MUST pass"
    )


def test_no_cann_source_falls_back_to_structural(tmp_path, monkeypatch):
    """No CANN source on disk → (A) no-ops, (B) is the sole guard.

    Recipe-assembled tiling → PASS even though (A) cannot run.
    """
    monkeypatch.setenv("CANN_SOURCE_ROOT", str(tmp_path / "absent_cann"))
    ws = _make_port_a3_fa_ws(tmp_path, _GOOD_TILING)
    assert getattr(fp, '_check_ge_ophost_raw_cann_copy')(ws) is None


# ---------------------------------------------------------------------------
# SCOPING — port_a3 + FA-class only
# ---------------------------------------------------------------------------
def test_non_port_a3_mode_skips_gate(tmp_path, monkeypatch):
    """Backward and an unknown mode are out of scope."""
    monkeypatch.setenv("CANN_SOURCE_ROOT", str(tmp_path / "no_cann"))
    for mode in ("backward", "unsupported"):
        ws = _make_port_a3_fa_ws(tmp_path / mode, _RAW_TILING, mode=mode)
        assert getattr(fp, '_check_ge_ophost_raw_cann_copy')(ws) is None, (
            f"gate must skip non-port_a3 mode {mode!r}"
        )


def test_non_fa_op_skips_gate(tmp_path, monkeypatch):
    """A non-FA-named port_a3 op (no attention tag) is out of scope."""
    monkeypatch.setenv("CANN_SOURCE_ROOT", str(tmp_path / "no_cann"))
    ws = _make_port_a3_fa_ws(tmp_path, _RAW_TILING, op="layer_norm")
    assert getattr(fp, '_check_ge_ophost_raw_cann_copy')(ws) is None, (
        "non-FA op must be out of scope (only FA-class GE op_host is policed)"
    )


def test_no_ge_tiling_skips_gate(tmp_path, monkeypatch):
    """port_a3 FA op with no GE tiling.cpp in op_host/ → nothing to police."""
    monkeypatch.setenv("CANN_SOURCE_ROOT", str(tmp_path / "no_cann"))
    ws = tmp_path / "flash_attention_score"
    (ws / "op_host").mkdir(parents=True)
    (ws / ".opgen_state.json").write_text(
        json.dumps({"op": "flash_attention_score", "opgen_mode": "port_a3_to_a5"})
    )
    # Only a kernel-side file, no *_tiling.cpp.
    (ws / "op_host" / "flash_attention_score_common.h").write_text("// header\n")
    assert getattr(fp, '_check_ge_ophost_raw_cann_copy')(ws) is None


def test_no_op_host_dir_skips_gate(tmp_path, monkeypatch):
    """No op_host/ at all → no GE op_host to guard."""
    monkeypatch.setenv("CANN_SOURCE_ROOT", str(tmp_path / "no_cann"))
    ws = tmp_path / "flash_attention_score"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text(
        json.dumps({"op": "flash_attention_score", "opgen_mode": "port_a3_to_a5"})
    )
    assert getattr(fp, '_check_ge_ophost_raw_cann_copy')(ws) is None


# ---------------------------------------------------------------------------
# Prestage carve-out (flash_attention_score-ghasm-1 2026-06-11): the byte
# check (A) must NOT flag a sanctioned prestaged CANN struct-def header that
# is listed in `.upstream_prestaged.json` — phase_o25_a3_ref pre-stages
# `<op>_tiling_common.h` (byte-identical to CANN by design, sha256-protected
# by the UPSTREAM_PRESTAGE_TAMPER gate). Without the carve-out, the two gates
# collide: prestage says keep-identical, byte-check says reject-because-identical.
# ---------------------------------------------------------------------------
def test_byte_check_skips_prestaged_struct_header(tmp_path, monkeypatch):
    """A prestaged op_host/*.h byte-identical to CANN, listed in
    .upstream_prestaged.json, must NOT trip the byte check.
    """
    # Build a fake CANN tree with a tiling_common.h, point CANN_SOURCE_ROOT at it.
    cann = tmp_path / "cann"
    cann_oph = cann / "ops-transformer" / "attention" / "flash_attention_score" / "op_host"
    cann_oph.mkdir(parents=True)
    common_body = "namespace optiling { struct FlashAttentionScoreCompileInfo { int x; }; }\n"
    (cann_oph / "flash_attention_score_tiling_common.h").write_text(common_body)
    monkeypatch.setenv("CANN_SOURCE_ROOT", str(cann))

    # Recipe-assembled tiling.cpp (passes structural (B)) + a prestaged
    # struct header that is byte-identical to the CANN one above.
    ws = _make_port_a3_fa_ws(
        tmp_path, _GOOD_TILING,
        extra_op_host={"flash_attention_score_tiling_common.h": common_body},
    )
    (ws / ".upstream_prestaged.json").write_text(json.dumps({
        "schema_version": 1,
        "staged_files": {
            "op_host/flash_attention_score_tiling_common.h":
                hashlib.sha256(common_body.encode()).hexdigest(),
        },
    }))
    assert getattr(fp, '_check_ge_ophost_raw_cann_copy')(ws) is None, (
        "prestaged struct-def header (in .upstream_prestaged.json) must be "
        "exempt from the byte check — it's sanctioned, sha256-protected"
    )


def test_byte_check_still_flags_non_prestaged_cann_copy(tmp_path, monkeypatch):
    """Sanity: the carve-out is narrow — a NON-prestaged GE .cpp byte-identical
    to CANN still REJECTS (the real 13-archive regression).
    """
    cann = tmp_path / "cann"
    cann_oph = cann / "ops-transformer" / "attention" / "flash_attention_score" / "op_host"
    cann_oph.mkdir(parents=True)
    # CANN's tiling.cpp uses the shared layer too (so structural (B) passes and
    # we reach the byte check), but it's byte-identical to the workspace copy.
    (cann_oph / "flash_attention_score_tiling.cpp").write_text(_GOOD_TILING)
    monkeypatch.setenv("CANN_SOURCE_ROOT", str(cann))

    ws = _make_port_a3_fa_ws(tmp_path, _GOOD_TILING)  # byte-identical to CANN, NOT prestaged
    # empty prestage manifest → tiling.cpp is not exempt
    (ws / ".upstream_prestaged.json").write_text(json.dumps({"staged_files": {}}))
    rationale = getattr(fp, '_check_ge_ophost_raw_cann_copy')(ws)
    assert rationale is not None, "non-prestaged byte-identical GE .cpp MUST reject"
    assert "BYTE-IDENTICAL" in rationale


# ---------------------------------------------------------------------------
# GateID stability
# ---------------------------------------------------------------------------
def test_gate_id_value_stable():
    """Lock GateID.GE_OPHOST_RAW_CANN_COPY.value for the rollback-loop detector."""
    assert fp.GateID.GE_OPHOST_RAW_CANN_COPY.value == "ge_ophost_raw_cann_copy"
