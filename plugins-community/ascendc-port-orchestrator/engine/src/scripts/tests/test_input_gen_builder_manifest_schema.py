# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Bug B (blue 2026-06-01) regression: /aog-input-gen-builder skill templates
MUST emit manifest cases[].input_stats matching the canonical
input_gen.template.py schema, so scoped reference consumers can build
real-shape probe inputs.

Root of the bug: input_gen.simple.py + input_gen.fused.py had diverged to emit
`"cases": [name_str, ...]` (string names, no input_stats) while the canonical
template emits the full per-case dict. A reference consumer then found no
input_stats, fell back to a 1D input, and model.forward's 2D index raised
IndexError.

This test fills each skill template with a minimal real schema, runs it, and
asserts manifest cases[0].input_stats is a dict with shape+dtype — the exact
contract scoped reference consumers use. Guards against re-divergence.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]  # tests→scripts→src→repo-root
_REF_PROVIDER = _REPO / "src" / "scripts" / "reference_provider"


def _resolve_templates() -> pathlib.Path:
    plugin_root = pathlib.Path(__file__).resolve().parents[4]
    return plugin_root / "skills" / "aog-input-gen-builder" / "templates"


_TPL_DIR = _resolve_templates()


def _fill_simple(text: str) -> str:
    text = (text.replace("<<<OP_NAME>>>", "relu_demo")
                .replace("<<<FORMULA>>>", "y=relu(x)")
                .replace("<<<OUTPUT_NAME>>>", "y")
                .replace("<<<COVERAGE_TIER>>>", "pilot"))
    text = text.replace(
        '"tensor_inputs": [],',
        '"tensor_inputs": [{"name": "x", "role": "operand"}],',
        1,
    )
    return text.replace(
        '"scalar_inputs": [],',
        '"scalar_inputs": [{"name": "alpha", "dtype": "float", '
        '"default": 0.5, "probe_values": [0.0, 0.5, 1.0]}],',
        1,
    )


def _fill_fused(text: str) -> str:
    text = (text.replace("<<<OP_NAME>>>", "fused_demo")
                .replace("<<<FORMULA>>>", "y=f(x)")
                .replace("<<<OUTPUT_NAME>>>", "y")
                .replace("<<<COVERAGE_TIER>>>", "pilot")
                .replace("<<<RANK>>>", "1")
                .replace("<<<DTYPE>>>", "float32"))
    return text.replace(
        '"tensor_inputs": [],',
        '"tensor_inputs": [{"name": "x", "role": "operand"}],',
        1,
    )


@pytest.mark.parametrize("template,filler", [
    ("input_gen.simple.py", _fill_simple),
    ("input_gen.fused.py", _fill_fused),
])
def test_skill_template_manifest_has_input_stats(tmp_path, template, filler):
    filled = filler((_TPL_DIR / template).read_text())
    # NB: don't assert `"<<<" not in filled` — the template's own placeholder
    # guard contains the literal string `"<<<"`. A genuinely-unfilled <<<NAME>>>
    # placeholder makes the template's guard raise → caught by returncode != 0.
    ig = tmp_path / "input_gen.py"
    ig.write_text(filled)

    env = {"PYTHONPATH": str(_REF_PROVIDER)}
    r = subprocess.run([sys.executable, "input_gen.py"], cwd=tmp_path,
                       env={**os.environ, **env},
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"{template} run failed:\n{r.stderr[:600]}"

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    cases = manifest.get("cases")
    assert cases, f"{template}: manifest has no cases"
    c0 = cases[0]
    assert isinstance(c0, dict), (
        f"{template}: cases[0] is {type(c0).__name__}, not a dict — regressed to "
        f"string-name format (Bug B). reference consumers need cases[0].input_stats.")
    stats = c0.get("input_stats")
    assert isinstance(stats, dict) and stats, f"{template}: cases[0].input_stats missing/empty"
    first = next(iter(stats.values()))
    assert "shape" in first and "dtype" in first, (
        f"{template}: input_stats entry lacks shape/dtype: {first}")
