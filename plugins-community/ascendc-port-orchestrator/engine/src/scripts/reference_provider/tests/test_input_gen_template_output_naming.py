# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Test the input_gen.template.py output-naming guard (tensor_output OR tensor_outputs).

E2 (2026-05-29): the template now accepts multi-output ops via `tensor_outputs` (list)
in addition to single `tensor_output` (str, back-compat). Guard extracted to
`_output_naming_errors` for testability.
"""
import importlib.util
import json
import pathlib
import re
import sys

_RP = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RP))  # so the template's `from case_gen import ...` resolves
_spec = importlib.util.spec_from_file_location("_ig_template", _RP / "input_gen.template.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_errs = getattr(_mod, '_output_naming_errors')
_stats = getattr(_mod, '_stringify_tensor_stat')


def test_input_generation_sources_use_canonical_edge_inputs_fixture():
    """Template guidance and implementation must share the fixture name."""
    for source_name in ("input_gen.template.py", "case_gen.py"):
        source = (_RP / source_name).read_text()
        assert "edge_inputs.pt" in source
        assert re.search(r"(?<!edge_)inputs\.pt", source) is None


def test_single_tensor_output_ok():
    assert _errs({"tensor_output": "a"}) == []


def test_multi_tensor_outputs_ok():
    assert _errs({"tensor_outputs": ["attention_out", "softmax_max", "softmax_sum"]}) == []


def test_multi_rescues_replace_single():
    # tensor_output still a placeholder, but tensor_outputs is validly set → OK.
    assert _errs({"tensor_output": "<<<REPLACE-x>>>",
                  "tensor_outputs": ["attention_out", "softmax_max"]}) == []


def test_neither_set_fails():
    assert _errs({}) != []
    assert _errs({"tensor_output": ""}) != []


def test_replace_placeholder_single_fails():
    assert _errs({"tensor_output": "<<<REPLACE-output-tensor-name-e.g.-a>>>"}) != []


def test_empty_tensor_outputs_list_fails():
    assert _errs({"tensor_outputs": []}) != []


def test_tensor_outputs_with_replace_element_fails():
    assert _errs({"tensor_outputs": ["attention_out", "<<<REPLACE-y>>>"]}) != []


def test_tensor_outputs_with_nonstr_element_fails():
    assert _errs({"tensor_outputs": ["attention_out", 123]}) != []


def test_list_of_tensors_stats_are_json_serializable():
    stats = _stats([
        _mod.torch.tensor([1.0, 2.0]),
        _mod.torch.tensor([3.0]),
    ])
    assert stats["length"] == 2
    assert stats["items"][0]["shape"] == [2]
    json.dumps(stats)


def test_unknown_stat_value_falls_back_to_repr():
    stats = _stats({1, 2})
    assert isinstance(stats["value"], str)
    json.dumps(stats)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
