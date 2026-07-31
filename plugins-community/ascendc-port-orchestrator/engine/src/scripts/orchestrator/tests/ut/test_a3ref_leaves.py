# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Characterization UT for the phase_o25_a3_ref decomposition leaves
(a3_ref_common / a3_ref_derive / a3_ref_npu / a3_ref_validate), 2026-07-06.

The bulk of these leaves' public functions are already exercised through the
facade suite (test_phase_o25_a3_ref, test_o25_model_contract,
test_port_a3_reference_capture_validation, test_debt_101_*). This file closes
the remaining DIRECT-coverage gaps for the pure helpers that previously had no
dedicated test: the NPU-delegation static scanner (_strip_py_comments /
_strip_strings_and_comments / _detect_npu_delegation) and the container-path
translation, plus a re-export sanity check that the facade exposes every symbol
external callers + monkeypatch sites depend on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import a3_ref_common as common  # noqa: E402
import a3_ref_validate as validate  # noqa: E402
import phase_o25_a3_ref as facade  # noqa: E402


# ---- a3_ref_common: container-path translation ----------------------------

def test_translate_local_home_to_container(monkeypatch):
    monkeypatch.delenv("A3_CONTAINER_HOME", raising=False)
    # force the default (no env file resolution surprises)
    monkeypatch.setattr(common, "_a3_container_home", lambda: "/home/npu_user")
    p = getattr(common, '_translate_to_a3_container_path')(Path("/home/alice/workspace/cann/foo"))
    assert str(p) == "/home/npu_user/workspace/cann/foo"


def test_translate_non_home_path_unchanged():
    assert getattr(common, '_translate_to_a3_container_path')(Path("/tmp/foo")) == Path("/tmp/foo")
    assert getattr(common, '_translate_to_a3_container_path')(Path("/opt/bar")) == Path("/opt/bar")


def test_translate_bare_home_user_unchanged():
    # /home/<user> with no tail — nothing meaningful to translate
    assert getattr(common, '_translate_to_a3_container_path')(Path("/home/bob")) == Path("/home/bob")


def test_a3_container_home_env_override_wins(monkeypatch):
    monkeypatch.setenv("A3_CONTAINER_HOME", "/home/custom_mount")
    assert getattr(common, '_a3_container_home')() == "/home/custom_mount"


# ---- a3_ref_common: shared case helpers -----------------------------------

def test_case_model_kwargs_unwraps_wrapped_case():
    wrapped = {"idx": 0, "name": "c0", "shape": [4], "inputs": {"x": 1, "y": 2}, "meta": {}}
    assert getattr(common, '_case_model_kwargs')(wrapped) == {"x": 1, "y": 2}


def test_case_model_kwargs_flat_case_unchanged():
    flat = {"x": 1, "y": 2}
    assert getattr(common, '_case_model_kwargs')(flat) == flat
    lst = [1, 2, 3]
    assert getattr(common, '_case_model_kwargs')(lst) is lst


def test_coerce_case_list_int_keyed_dict():
    obj = {0: {"a": 1}, 1: {"b": 2}, 2: {"c": 3}}
    assert getattr(common, '_coerce_case_list')(obj) == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_coerce_case_list_str_int_keys():
    obj = {"0": {"a": 1}, "1": {"b": 2}}
    assert getattr(common, '_coerce_case_list')(obj) == [{"a": 1}, {"b": 2}]


def test_coerce_case_list_documented_schema_untouched():
    obj = {"inputs": [1], "a3_outputs": [2]}
    assert getattr(common, '_coerce_case_list')(obj) is obj


def test_coerce_case_list_non_contiguous_untouched():
    obj = {0: {"a": 1}, 2: {"c": 3}}  # missing 1 → don't guess
    assert getattr(common, '_coerce_case_list')(obj) is obj


def test_coerce_case_list_list_untouched():
    obj = [{"a": 1}]
    assert getattr(common, '_coerce_case_list')(obj) is obj


# ---- a3_ref_validate: NPU-delegation static scanner -----------------------

def test_strip_py_comments_removes_hash_comments():
    src = "x = 1  # inline\n# full line\ny = 2\n"
    out = getattr(validate, '_strip_py_comments')(src)
    assert "inline" not in out
    assert "full line" not in out
    assert "x = 1" in out and "y = 2" in out


def test_detect_npu_delegation_flags_torch_npu_import():
    hits = getattr(validate, '_detect_npu_delegation')("import torch_npu\nx = 1\n")
    assert getattr(validate, '_LABEL_IMPORT_TORCH_NPU') in hits
    assert getattr(validate, '_LABEL_TORCH_NPU') in hits


def test_detect_npu_delegation_flags_npu_call():
    hits = getattr(validate, '_detect_npu_delegation')("y = torch_npu.npu_fusion_attention(q, k, v)\n")
    assert getattr(validate, '_LABEL_NPU_CALL') in hits


def test_detect_npu_delegation_flags_dot_npu_move():
    hits = getattr(validate, '_detect_npu_delegation')("z = x.npu()\n")
    assert getattr(validate, '_LABEL_DOT_NPU') in hits


def test_detect_npu_delegation_clean_cpu_model_has_no_hits():
    src = (
        "import torch\n"
        "class Model(torch.nn.Module):\n"
        "    def forward(self, x):\n"
        "        return torch.relu(x)\n"
    )
    assert getattr(validate, '_detect_npu_delegation')(src) == []


def test_detect_npu_delegation_ignores_delegation_tokens_in_strings_and_comments():
    # The scanner strips strings + comments first, so a mention of torch_npu
    # inside a docstring/comment must NOT be flagged as real delegation.
    src = (
        '"""This model does NOT use torch_npu or .npu() at all."""\n'
        "import torch\n"
        "# note: avoid npu_fusion_attention here\n"
        "def forward(x):\n"
        "    return torch.relu(x)\n"
    )
    assert getattr(validate, '_detect_npu_delegation')(src) == []


def test_strip_strings_and_comments_masks_string_content():
    src = 'a = "torch_npu inside string"\n'
    masked = getattr(validate, '_strip_strings_and_comments')(src)
    assert "torch_npu" not in masked


# ---- facade re-export sanity ----------------------------------------------

def test_facade_reexports_all_external_symbols():
    # External callers + monkeypatch sites depend on these names being resolvable
    # as attributes of the phase_o25_a3_ref facade after the split.
    for name in [
        "provision_a3_reference", "_coerce_case_list",
        "_default_run_remote", "format_block_message", "O25A3Report",
        "provision_native_capture", "validate_model_contract", "derive_aclnn_entry",
        "_a3_container_home", "ModelContractResult", "ensure_edge_inputs",
        "check_a3_npu_busy", "pick_idle_npu_in_range", "_scp_push_dir", "_scp_pull_files",
        "_run_npu_smi", "parse_npu_range", "parse_op_def_signature",
        "_a3_host_workspace_root_from_env",
    ]:
        assert hasattr(facade, name), f"facade missing re-export: {name}"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
