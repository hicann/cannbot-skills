# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Test deterministic GE op_host assembly for the FA-class fixture.

Background: the port_a3 FA blackbox reaches `done` + promotes, but the archive
`op_host/` ships ONLY shared headers — NO GE def/infershape/tiling.cpp, because
the kw_brief `_fa_ge_host_gen_block` is an LLM-skippable INSTRUCTION, not a hard
gate. The KB already carries the correct, compile-verified, non-CANN GE op_host
templates under `fa_class/templates/op_host/`. The assembler DELIVERS them into
the workspace op_host/ at finalize-prep (before the GE_OPHOST_RAW_CANN_COPY
gate), op-name parameterized, idempotent, with stale-arch35 cleanup.

These tests verify:
  - assembler copies the 3 GE .cpp (def/infershape/tiling) into op_host/
  - assembled tiling.cpp uses `wfh::` and md5 != CANN source
  - op-name parameterization (filename + content substitution for non-FAS op)
  - idempotency: a real worker-emitted GE .cpp is preserved, not overwritten
  - a CANN-byte-copy in the workspace is NOT treated as authoritative (overwritten)
  - stale CANN arch35 header under op_host/ is removed
  - scope: skips non-port_a3 + non-FA-class workspaces
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import finalize_pipeline as fp


_FAS = "flash_attention_score"
_TMPL_DIR = fp._FA_GE_OPHOST_TEMPLATE_DIR
_CANN_TILING = (
    Path.home() / "workspace" / "cann" / "ops-transformer" / "attention"
    / _FAS / "op_host" / f"{_FAS}_tiling.cpp"
)


def _make_ws(tmp_path: Path, *, op: str = _FAS, mode: str = "port_a3_to_a5") -> Path:
    ws = tmp_path / op
    (ws / "op_host").mkdir(parents=True, exist_ok=True)
    (ws / ".opgen_state.json").write_text(json.dumps({"op": op, "opgen_mode": mode}))
    return ws


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def test_template_dir_exists_and_is_kb_authored():
    """Pre-flight: the KB templates exist and are non-CANN (md5 differs)."""
    assert _TMPL_DIR.is_dir(), f"KB template dir absent: {_TMPL_DIR}"
    for suf in ("_def", "_infershape", "_tiling"):
        assert (_TMPL_DIR / f"{_FAS}{suf}.cpp").is_file()
    tiling = (_TMPL_DIR / f"{_FAS}_tiling.cpp").read_text()
    assert "wfh::" in tiling or "wp_fa_host::" in tiling
    if _CANN_TILING.is_file():
        assert _md5(_TMPL_DIR / f"{_FAS}_tiling.cpp") != _md5(_CANN_TILING)


def test_assembler_emits_kb_authored_tiling_common_header(tmp_path):
    """Emit a KB-authored CompileInfo POD header.

    Owner rule 2026-06-13: `<op>_tiling_common.h` is CANN-source-only and is not
    in the CANN library install. The assembler must
    emit a KB-AUTHORED version (built only from installed platform_ascendc API),
    NOT ship/depend on the CANN source copy → makes the GE op_host 100% KB.
    """
    ws = _make_ws(tmp_path)
    rep = fp.assemble_ge_ophost(ws)
    tc = ws / "op_host" / f"{_FAS}_tiling_common.h"
    assert tc.is_file(), f"assembler did not emit {tc.name}; report={rep}"
    assert f"op_host/{_FAS}_tiling_common.h" in rep["assembled"]
    txt = tc.read_text()
    # Authorship is established by the body and installed API dependency; the
    # shipped source still carries the repository's mandatory CANN license.
    assert "FlashAttentionScoreCompileInfo" in txt
    assert "platform_ascendc.h" in txt
    assert "Copyright (c) 2026 Huawei Technologies Co., Ltd." in txt
    assert "CANN Open Software License Agreement Version 2.0" in txt


def test_assembler_tiling_common_parameterized(tmp_path):
    """Parameterize the tiling_common.h filename token by operator name.

    This must generalize to a non-FAS FA-class operator.
    """
    op = "flash_attention_score_v2"
    ws = _make_ws(tmp_path, op=op)
    fp.assemble_ge_ophost(ws)
    assert (ws / "op_host" / f"{op}_tiling_common.h").is_file()


def test_assembler_delivers_ge_cpp_into_op_host(tmp_path):
    ws = _make_ws(tmp_path)
    rep = fp.assemble_ge_ophost(ws)
    assert rep["ran"] is True, rep
    assert not rep["errors"], rep["errors"]
    oph = ws / "op_host"
    for suf in ("_def", "_infershape", "_tiling"):
        assert (oph / f"{_FAS}{suf}.cpp").is_file(), f"missing {suf}.cpp"
        assert f"op_host/{_FAS}{suf}.cpp" in rep["assembled"]


def test_assembled_tiling_uses_wfh_and_differs_from_cann(tmp_path):
    ws = _make_ws(tmp_path)
    fp.assemble_ge_ophost(ws)
    tiling = ws / "op_host" / f"{_FAS}_tiling.cpp"
    text = tiling.read_text()
    assert "wfh::" in text or "wp_fa_host::" in text
    assert '#include "wp_fa_host_tiling.h"' in text
    if _CANN_TILING.is_file():
        assert _md5(tiling) != _md5(_CANN_TILING), "assembled == CANN (raw copy!)"


def test_assembled_tiling_passes_ge_ophost_gate(tmp_path):
    """End-to-end: after assembly, the GE_OPHOST_RAW_CANN_COPY gate accepts."""
    ws = _make_ws(tmp_path)
    fp.assemble_ge_ophost(ws)
    assert fp._check_ge_ophost_raw_cann_copy(ws) is None


def test_op_name_parameterization(tmp_path):
    """Substitute a different FA-class operator name throughout the template.

    Both the filename and content must use the new operator name.
    """
    op = "my_fused_attention"  # FA-named (contains 'attention')
    ws = _make_ws(tmp_path, op=op)
    rep = fp.assemble_ge_ophost(ws)
    assert rep["ran"] is True, rep
    oph = ws / "op_host"
    # filename parameterized
    assert (oph / f"{op}_tiling.cpp").is_file()
    assert not (oph / f"{_FAS}_tiling.cpp").is_file()
    # content parameterized: the original op token no longer appears as a
    # standalone identifier (the new op name took its place)
    def_text = (oph / f"{op}_def.cpp").read_text()
    import re
    assert not re.search(rf"\b{_FAS}\b", def_text), \
        "template op token survived in parameterized content"
    assert op in def_text


def test_identity_substitution_for_fas(tmp_path):
    """Keep substitution for flash_attention_score byte-identical.

    Headers outside the substitution step remain out of scope.
    """
    ws = _make_ws(tmp_path)
    fp.assemble_ge_ophost(ws)
    for suf in ("_def", "_infershape", "_tiling"):
        got = (ws / "op_host" / f"{_FAS}{suf}.cpp").read_text()
        want = (_TMPL_DIR / f"{_FAS}{suf}.cpp").read_text()
        assert got == want, f"{suf}.cpp identity substitution diverged"


def test_idempotent_preserves_worker_emit(tmp_path):
    """Preserve an authoritative worker-emitted tiling.cpp.

    The assembler must not overwrite a file that uses the shared wfh layer.
    """
    ws = _make_ws(tmp_path)
    worker_tiling = (
        '#include <cstdint>\n'
        '#include "wp_fa_host_tiling.h"\n'
        'namespace wfh = wp_fa_host;\n'
        '// WORKER-EMITTED SENTINEL — do not overwrite\n'
        'static int DoTiling() { return (int)wfh::CalcDBasicBlock(64); }\n'
    )
    tiling = ws / "op_host" / f"{_FAS}_tiling.cpp"
    tiling.write_text(worker_tiling)
    rep = fp.assemble_ge_ophost(ws)
    assert tiling.read_text() == worker_tiling, "worker emit was overwritten!"
    assert f"op_host/{_FAS}_tiling.cpp" in rep["preserved"]
    assert f"op_host/{_FAS}_tiling.cpp" not in rep["assembled"]


def test_overwrites_non_shared_tiling(tmp_path):
    """Overwrite a non-authoritative workspace tiling.cpp.

    A file that does not use the shared layer is replaced with the
    KB recipe-assembled version (so the gate then passes).
    """
    ws = _make_ws(tmp_path)
    bad_tiling = (
        '#include <cstdint>\n'
        'static int DoTiling() { int dbb = (64+63)/64*64; return dbb; }\n'
    )
    tiling = ws / "op_host" / f"{_FAS}_tiling.cpp"
    tiling.write_text(bad_tiling)
    rep = fp.assemble_ge_ophost(ws)
    text = tiling.read_text()
    assert "wfh::" in text or "wp_fa_host::" in text, "not replaced by KB template"
    assert f"op_host/{_FAS}_tiling.cpp" in rep["assembled"]


def test_stale_arch35_cleanup(tmp_path):
    """A leftover CANN arch35 header under op_host/ must be removed."""
    ws = _make_ws(tmp_path)
    arch35 = ws / "op_host" / "arch35"
    arch35.mkdir(parents=True, exist_ok=True)
    stale = arch35 / f"{_FAS}_tiling_regbase.h"
    stale.write_text("// CANN arch35 leftover\n")
    rep = fp.assemble_ge_ophost(ws)
    assert not stale.exists(), "stale arch35 header survived"
    assert not arch35.exists(), "empty arch35 dir not removed"
    assert any("arch35" in s for s in rep["stale_arch35_removed"])


def test_scope_skips_non_port_a3(tmp_path):
    ws = _make_ws(tmp_path, mode="backward")
    rep = fp.assemble_ge_ophost(ws)
    assert rep["ran"] is False
    assert "port_a3" in (rep["skipped_reason"] or "")
    assert not (ws / "op_host" / f"{_FAS}_tiling.cpp").exists()


def test_scope_skips_non_fa_class(tmp_path):
    ws = _make_ws(tmp_path, op="elementwise_add")
    rep = fp.assemble_ge_ophost(ws)
    assert rep["ran"] is False
    assert "FA-class" in (rep["skipped_reason"] or "")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
