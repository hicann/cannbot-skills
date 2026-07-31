# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression tests — architecture-class finalize gate (OL-188, design
`PORT_A3_CUBE_CLASS_MIX_ENFORCEMENT_DESIGN.md` §3.3 / §8).

The gate `check_architecture_class` is the structural backstop that rejects a
pure-VEC kernel generated for a cube-required op-class (same anti-cheat tier as
CPU fallback). Before this PR it had ZERO regression coverage and could silently
SKIP — exactly the OL-160 / SAFETY_NET_NAME_COUPLING failure-mode (a gate that
returns "no violation" because it had nothing to scan, with no test to catch the
no-op).

These fixtures pin the load-bearing behaviors (design §3.3):
  - red:                cube-required ref + pure-VEC gen            -> ARCHITECTURAL_HACK
  - green:              cube-required ref + cube-marker gen         -> PASS
  - unresolved-source:  any generated architecture                  -> fail closed
  - target-only source: target-side files cannot replace arch22 evidence

If any of these ever flips, the safety net has regressed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

from checks.architecture_class_check import check_architecture_class  # type: ignore
import finalize_pipeline as fp  # type: ignore

# A cube marker that _has_cube_markers() matches (matmul library instantiation).
_CUBE_SRC = (
    "#include \"lib/matmul_intf.h\"\n"
    "using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, half>;\n"
    "matmul::Matmul<AType, AType, AType, AType> mm;\n"
)
# A pure-VEC kernel: NO cube marker anywhere.
_VEC_SRC = (
    "#include \"kernel_operator.h\"\n"
    "using namespace AscendC;\n"
    "// pure vector: DataCopy + Add, no matmul/cube primitives\n"
    "__aicore__ inline void compute() { /* Add(dst, a, b, n); */ }\n"
)


def _make_ref(tmp_path: Path, *, family_rel: str, cube_marker: bool) -> Path:
    """Build a fake CANN reference dir under a CANN-family path.

    family_rel e.g. 'ops-nn/matmul' (cube-required family) — the path is built so
    `_classify_reference_arch`'s family-root scan finds the `ops-nn`/`ops-transformer`
    segment. cube_marker controls whether op_kernel/*.h grep also signals cube.
    """
    ref = tmp_path / "cann" / family_rel / "some_op"
    ok = ref / "op_kernel" / "arch22"
    ok.mkdir(parents=True, exist_ok=True)
    (ok / "some_op.h").write_text(_CUBE_SRC if cube_marker else _VEC_SRC)
    return ref


def _make_ws(
    tmp_path: Path,
    *,
    gen_cube: bool,
    cube_mix_tag: bool = False,
) -> Path:
    ws = tmp_path / "workspace_op"
    kdir = ws / "kernel"
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "op.cpp").write_text(_CUBE_SRC if gen_cube else _VEC_SRC)
    if cube_mix_tag:
        import json

        (ws / "op_classification.json").write_text(
            json.dumps({"op_class_tags": ["CUBE_MIX"], "op_class": "L4"})
        )
    return ws


def test_red_cube_ref_vec_gen_rejected(tmp_path):
    """cube-required reference + pure-VEC generated kernel -> ARCHITECTURAL_HACK."""
    ref = _make_ref(tmp_path, family_rel="ops-nn/matmul", cube_marker=True)
    ws = _make_ws(tmp_path, gen_cube=False)
    r = check_architecture_class(ws, "some_op", port_source=ref)
    assert r["verdict"] == "ARCHITECTURAL_HACK", r
    assert r["reference_arch"] == "cube-required"
    assert r["generated_arch"] == "vec-only"


def test_green_cube_ref_cube_gen_passes(tmp_path):
    """cube-required reference + cube-marker generated kernel -> PASS."""
    ref = _make_ref(tmp_path, family_rel="ops-nn/matmul", cube_marker=True)
    ws = _make_ws(tmp_path, gen_cube=True)
    r = check_architecture_class(ws, "some_op", port_source=ref)
    assert r["verdict"] == "PASS", r
    assert r["generated_arch"] == "cube-required"


def test_cube_tag_and_vec_gen_still_require_source(tmp_path):
    """A durable cube tag cannot replace live detector-approved arch22 evidence."""
    ws = _make_ws(tmp_path, gen_cube=False, cube_mix_tag=True)
    # port_source unresolvable: pass a non-existent dir AND no .opgen_state.json
    bogus = tmp_path / "does_not_exist"
    r = check_architecture_class(ws, "some_op", port_source=bogus)
    assert r["verdict"] == "SOURCE_ARCH_UNVERIFIED", r
    assert r["reference_arch"] == "unknown"
    assert r["generated_arch"] == "vec-only"


def test_cube_tag_and_cube_gen_still_require_source(tmp_path):
    """Generated cube markers cannot compensate for unavailable arch22 evidence."""
    ws = _make_ws(tmp_path, gen_cube=True, cube_mix_tag=True)
    bogus = tmp_path / "does_not_exist"
    r = check_architecture_class(ws, "some_op", port_source=bogus)
    assert r["verdict"] == "SOURCE_ARCH_UNVERIFIED", r
    assert r["reference_arch"] == "unknown"
    assert r["generated_arch"] == "cube-required"


def test_no_tag_no_source_fails_closed(tmp_path):
    """Unresolved source AND no CUBE_MIX tag fails closed."""
    ws = _make_ws(tmp_path, gen_cube=False, cube_mix_tag=False)
    bogus = tmp_path / "does_not_exist"
    r = check_architecture_class(ws, "some_op", port_source=bogus)
    assert r["verdict"] == "SOURCE_ARCH_UNVERIFIED", r


def test_target_only_source_fails_closed(tmp_path):
    """A target-side tree is never accepted as migration source evidence."""
    ref = tmp_path / "cann" / "ops-nn" / "matmul" / "some_op"
    target_dir = ref / "op_kernel" / "arch35"
    target_dir.mkdir(parents=True)
    (target_dir / "some_op.h").write_text(_CUBE_SRC)
    ws = _make_ws(tmp_path, gen_cube=True, cube_mix_tag=True)

    r = check_architecture_class(ws, "some_op", port_source=ref)

    assert r["verdict"] == "SOURCE_ARCH_UNVERIFIED", r
    assert r["reference_arch"] == "unknown"


def test_backward_finalize_skips_migration_source_arch_gate(tmp_path):
    """Backward truth is CPU fp64 autograd, so absence of arch22 is not a waiver.

    It is simply outside this migration-only gate; other backward provenance,
    precision, and structural gates remain active.
    """
    ws = _make_ws(tmp_path, gen_cube=False)
    (ws / ".opgen_state.json").write_text(
        '{"schema_version": 1, "op": "some_grad", "opgen_mode": "backward"}'
    )

    assert getattr(fp, '_check_architecture_class')(ws) is None


def test_unknown_workflow_does_not_bypass_source_arch_gate(tmp_path):
    """An unowned workspace fails closed instead of inheriting the backward skip."""
    ws = _make_ws(tmp_path, gen_cube=False)

    reason = getattr(fp, '_check_architecture_class')(ws)

    assert reason is not None
    assert "not owned by a supported migration or backward workflow" in reason


def test_waiver_file_does_not_bypass(tmp_path):
    """Legacy waiver filenames do not bypass architecture enforcement."""
    ref = _make_ref(tmp_path, family_rel="ops-nn/matmul", cube_marker=True)
    ws = _make_ws(tmp_path, gen_cube=False)
    (ws / ".workflow_exception_arch_class").write_text("legacy waiver\n")
    r = check_architecture_class(ws, "some_op", port_source=ref)
    assert r["verdict"] == "ARCHITECTURAL_HACK", r


def test_pathstrip_cann_root_agnostic(tmp_path):
    """Family-root scan resolves cube-required for a NON-~/workspace/cann root.

    Pins the design §3.3 path-strip fix: a `/data/cann/...`-style root (here just
    tmp_path/cann) must still resolve `ops-nn/matmul` as cube-required, where the
    old hardcoded `~/workspace/cann/` .replace() would have degraded to vec-default.
    """
    ref = _make_ref(tmp_path, family_rel="ops-nn/matmul", cube_marker=False)  # family signal only
    ws = _make_ws(tmp_path, gen_cube=False)
    r = check_architecture_class(ws, "some_op", port_source=ref)
    # family alone (ops-nn/matmul) makes ref cube-required even without grep marker
    assert r["reference_arch"] == "cube-required", r
    assert r["verdict"] == "ARCHITECTURAL_HACK", r


# ---------------------------------------------------------------------------
# Explicit-vec-only-declaration override (recurrent_gated_delta_rule, 2026-06-18)
# ---------------------------------------------------------------------------
_AIV_ONLY_DECL = (
    "#include \"kernel_operator.h\"\n"
    "using namespace AscendC;\n"
    "extern \"C\" __global__ __aicore__ void recurrent_gated_delta_rule() {\n"
    "    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);\n"
    "    // pure-vector decode-step: Muls / Mul / ReduceSum / Cast, no cube\n"
    "}\n"
)


def _make_ref_with_src(tmp_path: Path, *, family_rel: str, src: str) -> Path:
    ref = tmp_path / "cann" / family_rel / "some_op"
    ok = ref / "op_kernel" / "arch22"
    ok.mkdir(parents=True, exist_ok=True)
    (ok / "some_op_apt.cpp").write_text(src)
    return ref


def test_aiv_only_decl_overrides_cube_family_prefix(tmp_path):
    """recurrent_gated_delta_rule regression: an op under the cube-required
    ops-transformer/attention family that EXPLICITLY declares
    KERNEL_TYPE_AIV_ONLY (and has zero cube markers) classifies vec-only —
    the explicit task-type declaration overrides the coarse family prefix.
    """
    from checks.architecture_class_check import _classify_reference_arch  # type: ignore
    ref = _make_ref_with_src(
        tmp_path, family_rel="ops-transformer/attention", src=_AIV_ONLY_DECL,
    )
    assert _classify_reference_arch(ref) == "vec-only"


def test_aiv_only_decl_does_not_override_when_cube_markers_present(tmp_path):
    """Control: if a cube marker is ALSO present, cube wins regardless of an
    AIV_ONLY declaration (the override fires only with zero cube markers).
    """
    from checks.architecture_class_check import _classify_reference_arch  # type: ignore
    ref = _make_ref_with_src(
        tmp_path, family_rel="ops-transformer/attention",
        src=_AIV_ONLY_DECL + _CUBE_SRC,
    )
    assert _classify_reference_arch(ref) == "cube-required"


def test_attention_family_no_aiv_decl_stays_cube_required(tmp_path):
    """Control: the existing strict-family behavior is preserved — an attention
    op with NEITHER an AIV_ONLY decl NOR cube markers (chunk-like, where cube
    is library-hidden) stays cube-required by family prefix.
    """
    from checks.architecture_class_check import _classify_reference_arch  # type: ignore
    ref = _make_ref(tmp_path, family_rel="ops-transformer/attention", cube_marker=False)
    assert _classify_reference_arch(ref) == "cube-required"
