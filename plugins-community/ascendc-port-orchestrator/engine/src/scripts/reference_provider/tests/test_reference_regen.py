# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""reference_regen (harness-gap #1 cpu_pytorch variant, scan 2026-07-24; main's
`.opgen_state.json` `reference_regen` schema; OL-282 / DEBT-158).

Verifies: cpu_pytorch deterministic regen produces edge_dataset.pt; unsupported
truth sources are ignored; no `reference_regen` block is a no-op; tampered
recipes and non-deterministic inputs fail
LOUD (never silently build a wrong reference).

Run: python3 -m pytest src/scripts/reference_provider/tests/test_reference_regen.py -v
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))  # src/scripts/reference_provider
import reference_regen as rr  # noqa: E402


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_cpu_pytorch_recipe(ws: Path, *, input_bytes: bytes = b"INPUTS_V1",
                              pin_input_sha: bool = True) -> None:
    ig = ws / "input_gen.py"
    ig.write_text(
        "from pathlib import Path\n"
        f"Path('edge_inputs.pt').write_bytes({input_bytes!r})\n"
        "Path('manifest.json').write_text('{}')\n"
    )
    gen = ws / "gen.py"
    gen.write_text("from pathlib import Path\nPath('outputs.pt').write_bytes(b'OUTPUTS')\n")
    comb = ws / "combine.py"
    comb.write_text("from pathlib import Path\nPath('edge_dataset.pt').write_bytes(b'DATASET')\n")
    block = {
        "truth_source": "cpu_pytorch",
        "input_gen": {"path": "input_gen.py", "sha256": _sha(ig.read_bytes())},
        "generator": {"path": "gen.py", "sha256": _sha(gen.read_bytes())},
        "env": {},
        "combine_cmd": [sys.executable, "combine.py"],
    }
    if pin_input_sha:
        block["input_data_sha256"] = _sha(input_bytes)
    (ws / ".opgen_state.json").write_text(json.dumps({"reference_regen": block}))


def test_cpu_pytorch_regen_produces_dataset(tmp_path):
    ws = tmp_path / "op"
    ws.mkdir()
    _write_cpu_pytorch_recipe(ws)
    assert rr.regen_reference(ws) is True
    assert (ws / "edge_dataset.pt").is_file()
    assert (ws / "edge_inputs.pt").read_bytes() == b"INPUTS_V1"


def test_no_reference_regen_block_is_noop(tmp_path):
    """Additive-hook guard: no `reference_regen` → False (no-op), existing behavior kept."""
    ws = tmp_path / "op"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text(json.dumps({"opgen_mode": "backward"}))
    assert rr.regen_reference(ws) is False
    assert not (ws / "edge_dataset.pt").exists()


def test_no_state_file_is_noop(tmp_path):
    ws = tmp_path / "op"
    ws.mkdir()
    assert rr.regen_reference(ws) is False


def test_unsupported_truth_source_is_noop(tmp_path):
    ws = tmp_path / "op"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text(
        json.dumps({"reference_regen": {"truth_source": "unsupported_source"}}))
    assert rr.regen_reference(ws) is False


def test_tampered_recipe_fails_loud(tmp_path):
    ws = tmp_path / "op"
    ws.mkdir()
    _write_cpu_pytorch_recipe(ws)
    # tamper the committed input_gen.py AFTER its sha was pinned
    (ws / "input_gen.py").write_text("# tampered\n")
    try:
        rr.regen_reference(ws)
        assert False, "tampered recipe must raise"
    except RuntimeError as e:
        assert "sha256 mismatch" in str(e)


def test_nondeterministic_input_fails_loud(tmp_path):
    """input_gen produces bytes != committed input_data_sha256 → refuse to build."""
    ws = tmp_path / "op"
    ws.mkdir()
    _write_cpu_pytorch_recipe(ws, input_bytes=b"INPUTS_V1")
    # rewrite input_gen to emit DIFFERENT inputs, and re-pin its OWN sha (recipe intact)
    ig = ws / "input_gen.py"
    ig.write_text(
        "from pathlib import Path\n"
        "Path('edge_inputs.pt').write_bytes(b'DRIFTED_INPUTS')\n"
        "Path('manifest.json').write_text('{}')\n"
    )
    st = json.loads((ws / ".opgen_state.json").read_text())
    st["reference_regen"]["input_gen"]["sha256"] = _sha(ig.read_bytes())  # recipe sha ok
    (ws / ".opgen_state.json").write_text(json.dumps(st))  # input_data_sha256 still pins INPUTS_V1
    try:
        rr.regen_reference(ws)
        assert False, "non-deterministic input must raise"
    except RuntimeError as e:
        assert "input_data_sha256" in str(e) or "drifted" in str(e)
